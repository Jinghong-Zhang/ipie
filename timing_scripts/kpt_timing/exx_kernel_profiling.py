import numpy
from numba import jit
import math
from line_profiler import LineProfiler


nk = 27
nchol = 250
nbsf = 32
nocc = 4
nwalkers = 10

def prof_kpt_symmchol_exx_kernel(rchola, rcholbara, Ghalfa_batch, kpq_mat, Qset):
    exx = kpt_symmchol_exx_kernel(rchola, rcholbara, Ghalfa_batch, kpq_mat, Qset)

@jit(nopython=True, fastmath=True)
def kpt_symmchol_exx_kernel(rchola, rcholbara, Ghalfa_batch, kpq_mat, Qset):
    """Compute coulomb contribution for complex rchol with RHF trial.

    Parameters
    ----------
    rchol : :class:`numpy.ndarray`
        Half-rotated cholesky.
    Ghalf : :class:`numpy.ndarray`
        Walker's half-rotated "green's function" shape is nalpha  x nbasis
    kpq_mat : :class:`numpy.ndarray`
        all k + q in fractional coordinates.
    mq_vec : :class:`numpy.ndarray`
        all -q in fractional coordinates.

    Returns
    -------
    ecoul : :class:`numpy.ndarray`
        coulomb contribution for all walkers.
    """
    # sort out cupy later
    zeros = numpy.zeros
    nwalkers = Ghalfa_batch.shape[0]

    # shape of rchola: (naux, nk, nocc, nk, nbsf) (gamma, k, i, q, p)
    # shape of Ghalf: (nw, nk, nocc, nk, nbsf)
    naux = rchola.shape[0]
    nocc = rchola.shape[2]
    nk = rchola.shape[1]
    exx = zeros(nwalkers, dtype=numpy.complex128)
    GhalfaT = Ghalfa_batch.transpose(0, 3, 4, 1, 2)
    GhalfaT = GhalfaT.transpose(1,3,0,2,4).copy()
    GhalfaT2 = Ghalfa_batch.transpose(1,3,0,2,4).copy()
    rcholaT = rchola.transpose(1,3,0,2,4).copy()
    rcholbaraT = rcholbara.transpose(1,3,0,2,4).copy()
    T1 = zeros((nocc, nocc), dtype=numpy.complex128)
    T2 = zeros((nocc, nocc), dtype=numpy.complex128)
    for iq in range(len(Qset)):
        iq_real = Qset[iq]        
        for ik in range(nk):
            for ikprime in range(nk):
                ikpr_pq = kpq_mat[iq_real, ikprime]
                ik_pq = kpq_mat[iq_real, ik]
                Lkq = rcholaT[ik, iq]
                Lbarkpq = rcholbaraT[ikprime, iq]
                for iw in range(nwalkers):
                    Ghalf_kpq_kprpq = GhalfaT[ik_pq, ikpr_pq, iw]
                    Ghalf_k_kp = GhalfaT2[ik,ikprime, iw]
                    for g in range(naux):
                        # T1 = Lkq[g] @ Ghalf_kpq_kprpq
                        # T2 = Ghalf_k_kp @ Lbarkpq[g]
                        # T1real = T1.real.copy()
                        # T1imag = T1.imag.copy()
                        # T2real = T2.real.copy()
                        # T2imag = T2.imag.copy()
                        # exxreal = numpy.sum(T1real*T2real - T1imag*T2imag)
                        # exximag = numpy.sum(T1real*T2imag + T1imag*T2real)
                        # exx[iw] += exxreal + 1j * exximag
                        for i in range(nocc):
                            for j in range(nocc):
                                exx[iw] += - T1[i, j] * T2[i, j]
                        # exx[iw] += -numpy.sum(T1*T2)
    return exx


rchola = numpy.random.rand(nchol, nk, nocc, nk, nbsf) + 1j * numpy.random.rand(nchol, nk, nocc, nk, nbsf)
rcholbara = numpy.random.rand(nchol, nk, nbsf, nk, nocc) + 1j * numpy.random.rand(nchol, nk, nbsf, nk, nocc)

Ghalfa_batch = numpy.random.rand(nwalkers, nk, nocc, nk, nbsf) + 1j * numpy.random.rand(nwalkers, nk, nocc, nk, nbsf)

# kpq mat is a matrix with integers in the range of 0 to nk
kpq_mat = numpy.random.randint(0, nk, (nk, nk))

lenQset = nk // 2 + 1
Qset = numpy.arange(lenQset)

lp = LineProfiler()

profiled_fn = lp(prof_kpt_symmchol_exx_kernel)
# profiled_fn = lp(kpt_symmchol_exx_kernel)
exx = kpt_symmchol_exx_kernel(rchola, rcholbara, Ghalfa_batch, kpq_mat, Qset)
profiled_fn(rchola, rcholbara, Ghalfa_batch, kpq_mat, Qset)
lp.print_stats()