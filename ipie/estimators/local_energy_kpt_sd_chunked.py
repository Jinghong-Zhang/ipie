# from line_profiler import LineProfiler
from math import ceil, sqrt

import numpy
from numba import jit

from ipie.estimators.local_energy import local_energy_G
from ipie.estimators.kernels import exchange_reduction_kpt
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import synchronize
from ipie.config import config

from ipie.systems.generic import Generic
from ipie.hamiltonians.kpt_chunked import KptComplexCholChunked
from ipie.walkers.uhf_walkers import UHFWalkers
from ipie.trial_wavefunction.single_det_kpt import KptSingleDet
from ipie.estimators.local_energy_kpt_sd import kpt_symmchol_ecoul_kernel_uhf, kpt_symmchol_exx_kernel

# from line_profiler import profile

import plum
# Note specialisations occur to because:
# 1. Numba does not allow for mixing types without a warning so need to split
# real and complex components apart when rchol is real. Green's function is
# complex in general.
# Optimize for case when wavefunction is RHF (factor of 2 saving)

@plum.dispatch
def local_energy_kpt_single_det_uhf_chunked(
    system: Generic,
    hamiltonian: KptComplexCholChunked,
    walkers: UHFWalkers,
    trial: KptSingleDet,
):
    """Compute local energy for walker batch (all walkers at once).

    Single determinant UHF case.

    Parameters
    ----------
    system : system object
        System being studied.
    hamiltonian : hamiltonian object
        Hamiltonian being studied.
    walkers : WalkerBatch
        Walkers object.
    trial : trial object
        Trial wavefunction.

    Returns
    -------
    local_energy : np.ndarray
        Total, one-body and two-body energies.
    """
    if config.get_option("use_gpu"):
        return local_energy_kpt_single_det_uhf_batch_chunked_gpu(system, hamiltonian, walkers, trial)
    else:
        return local_energy_kpt_single_det_uhf_chunked_cpu(system, hamiltonian, walkers, trial)

def local_energy_kpt_single_det_uhf_chunked_cpu(
    system: Generic,
    hamiltonian: KptComplexCholChunked,
    walkers: UHFWalkers,
    trial: KptSingleDet,
):
    """Compute local energy for walker batch (all walkers at once).

    Single determinant UHF case.

    Parameters
    ----------
    system : system object
        System being studied.
    hamiltonian : hamiltonian object
        Hamiltonian being studied.
    walkers : WalkerBatch
        Walkers object.
    trial : trial object
        Trial wavefunction.

    Returns
    -------
    local_energy : np.ndarray
        Total, one-body and two-body energies.
    """
    assert hamiltonian.chunked

    nwalkers = walkers.Ghalfa.shape[0]
    nk = hamiltonian.nk
    nalpha = trial.nalpha
    nbeta = trial.nbeta
    nbasis = hamiltonian.nbasis

    ghalfa = walkers.Ghalfa.reshape(nwalkers, nk, nalpha, nk, nbasis)
    ghalfb = walkers.Ghalfb.reshape(nwalkers, nk, nbeta, nk, nbasis)
    ghalfaT = walkers.Ghalfa.transpose(0, 2, 1).reshape(nwalkers, nk, nbasis, nk, nalpha)
    ghalfbT = walkers.Ghalfb.transpose(0, 2, 1).reshape(nwalkers, nk, nbasis, nk, nbeta)

    diagGhalfa = numpy.zeros((nwalkers, nk, nalpha, nbasis), dtype=numpy.complex128)
    diagGhalfb = numpy.zeros((nwalkers, nk, nbeta, nbasis), dtype=numpy.complex128)
    for ik in range(nk):
        diagGhalfa[:, ik, :, :] = ghalfa[:, ik, :, ik, :]
        diagGhalfb[:, ik, :, :] = ghalfb[:, ik, :, ik, :]
    diagGhalfa = diagGhalfa.reshape(nwalkers, nk * nalpha * nbasis)
    diagGhalfb = diagGhalfb.reshape(nwalkers, nk * nbeta * nbasis)
    e1b = diagGhalfa.dot(trial._rH1a.ravel())
    e1b += diagGhalfb.dot(trial._rH1b.ravel())
    e1b /= nk
    e1b += hamiltonian.ecore

    ghalfa = ghalfa.transpose(1, 3, 0, 2, 4).copy() # nk, nk, nw, nalpha, nbasis
    ghalfb = ghalfb.transpose(1, 3, 0, 2, 4).copy() # nk, nk, nw, nbeta, nbasis
    ghalfaTcoul = ghalfaT.transpose(1, 3, 0, 2, 4).copy() # nk, nk, nw, nbasis, nalpha
    ghalfbTcoul = ghalfbT.transpose(1, 3, 0, 2, 4).copy() # nk, nk, nw, nbasis, nbeta
    ghalfaTx = ghalfaT.transpose(1, 3, 2, 4, 0).copy() # nk, nk, nbasis, nalpha, nw
    ghalfbTx = ghalfbT.transpose(1, 3, 2, 4, 0).copy() # nk, nk, nbasis, nbeta, nw

    ghalfa_send = ghalfa.copy()
    ghalfb_send = ghalfb.copy()
    ghalfaTcoul_send = ghalfaTcoul.copy()
    ghalfbTcoul_send = ghalfbTcoul.copy()
    ghalfaTx_send = ghalfaTx.copy()
    ghalfbTx_send = ghalfbTx.copy()

    ghalfa_recv = xp.zeros_like(ghalfa)
    ghalfb_recv = xp.zeros_like(ghalfb)
    ghalfaTcoul_recv = xp.zeros_like(ghalfaTcoul)
    ghalfbTcoul_recv = xp.zeros_like(ghalfbTcoul)
    ghalfaTx_recv = xp.zeros_like(ghalfaTx)
    ghalfbTx_recv = xp.zeros_like(ghalfbTx)

    handler = walkers.mpi_handler
    senders = handler.senders
    receivers = handler.receivers

    rchola_chunk = trial._rchola_chunk
    rcholb_chunk = trial._rcholb_chunk
    rcholbara_chunk = trial._rcholbara_chunk
    rcholbarb_chunk = trial._rcholbarb_chunk

    ecoul_send = kpt_symmchol_ecoul_kernel_uhf(
        rchola_chunk, rcholb_chunk, rcholbara_chunk, rcholbarb_chunk, ghalfa, ghalfb, ghalfaTcoul, ghalfbTcoul, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus
    )

    exx_send = kpt_symmchol_exx_kernel(rchola_chunk, rcholbara_chunk, ghalfa, ghalfaTx, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus) 
    exx_send += kpt_symmchol_exx_kernel(rcholb_chunk, rcholbarb_chunk, ghalfb, ghalfbTx, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus) 

    exx_recv = exx_send.copy()
    ecoul_recv = ecoul_send.copy()

    for _ in range(handler.ssize - 1):
        for isend, sender in enumerate(senders):
            if handler.srank == isend:
                handler.scomm.Send(ghalfa_send, dest=receivers[isend], tag=1)
                handler.scomm.Send(ghalfb_send, dest=receivers[isend], tag=2)
                handler.scomm.Send(ghalfaTcoul_send, dest=receivers[isend], tag=3)
                handler.scomm.Send(ghalfbTcoul_send, dest=receivers[isend], tag=4)
                handler.scomm.Send(ghalfaTx_send, dest=receivers[isend], tag=5)
                handler.scomm.Send(ghalfbTx_send, dest=receivers[isend], tag=6)
                handler.scomm.Send(ecoul_send, dest=receivers[isend], tag=7)
                handler.scomm.Send(exx_send, dest=receivers[isend], tag=8)
            elif handler.srank == receivers[isend]:
                sender = numpy.where(receivers == handler.srank)[0]
                handler.scomm.Recv(ghalfa_recv, source=sender, tag=1)
                handler.scomm.Recv(ghalfb_recv, source=sender, tag=2)
                handler.scomm.Recv(ghalfaTcoul_recv, source=sender, tag=3)
                handler.scomm.Recv(ghalfbTcoul_recv, source=sender, tag=4)
                handler.scomm.Recv(ghalfaTx_recv, source=sender, tag=5)
                handler.scomm.Recv(ghalfbTx_recv, source=sender, tag=6)
                handler.scomm.Recv(ecoul_recv, source=sender, tag=7)
                handler.scomm.Recv(exx_recv, source=sender, tag=8)
        handler.scomm.barrier()

    # prepare sending
        ecoul_send = ecoul_recv.copy()
        ecoul_send += kpt_symmchol_ecoul_kernel_uhf(
        rchola_chunk, rcholb_chunk, rcholbara_chunk, rcholbarb_chunk, ghalfa_recv, ghalfb_recv, ghalfaTcoul_recv, ghalfbTcoul_recv, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus
    )
        exx_send = exx_recv.copy()
        exx_send += kpt_symmchol_exx_kernel(rchola_chunk, rcholbara_chunk, ghalfa_recv, ghalfaTx_recv, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus)
        exx_send += kpt_symmchol_exx_kernel(rcholb_chunk, rcholbarb_chunk, ghalfb_recv, ghalfbTx_recv, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus)
        ghalfa_send = ghalfa_recv.copy()
        ghalfb_send = ghalfb_recv.copy()
        ghalfaTcoul_send = ghalfaTcoul_recv.copy()
        ghalfbTcoul_send = ghalfbTcoul_recv.copy()
        ghalfaTx_send = ghalfaTx_recv.copy()
        ghalfbTx_send = ghalfbTx_recv.copy()


    if len(senders) > 1:
        for isend, sender in enumerate(senders):
            if handler.srank == sender:  # sending 1 xshifted to 0 xshifted_buf
                handler.scomm.Send(ecoul_send, dest=receivers[isend], tag=1)
                handler.scomm.Send(exx_send, dest=receivers[isend], tag=2)
            elif handler.srank == receivers[isend]:
                sender = numpy.where(receivers == handler.srank)[0]
                handler.scomm.Recv(ecoul_recv, source=sender, tag=1)
                handler.scomm.Recv(exx_recv, source=sender, tag=2)
    e2b = ecoul_recv + exx_recv

    energy = xp.zeros((nwalkers, 3), dtype=numpy.complex128)
    energy[:, 0] = e1b + e2b
    energy[:, 1] = e1b
    energy[:, 2] = e2b

    return energy

def kpt_symmchol_ecoul_kernel_batch_uhf_gpu(rchola, rcholb, rcholbara, rcholbarb, Ghalfa, Ghalfb, kpq_mat, Sset, Qplus):
    nwalkers = Ghalfa.shape[2]

    # shape of rchola: (nq, nk, nocc, naux, nbsf) (q, k, gamma, i, p)
    # shape of Ghalf: (nk, nk, nw, nocc, nbsf)
    unique_nq = len(Sset) + len(Qplus)
    nbsf = rchola.shape[4]
    nocc = rchola.shape[2]
    naux = rchola.shape[3]
    nk = rchola.shape[1]
    ecoul = xp.zeros((nwalkers), dtype=numpy.complex128)
    X = xp.zeros((unique_nq, naux, nwalkers), dtype=numpy.complex128)
    Xbar = xp.zeros((unique_nq, naux, nwalkers), dtype=numpy.complex128)
    for iq in range(len(Sset)):
        iq_real = Sset[iq]
        ikpq_vec = kpq_mat[iq_real]
        Ga_kpq = Ghalfa[:, ikpq_vec, :, :, :]
        GaT_kpq = Ghalfa[ikpq_vec]
        Gb_kpq = Ghalfb[:, ikpq_vec, :, :, :]
        GbT_kpq = Ghalfb[ikpq_vec]
        rchola_q = rchola[iq]
        rcholb_q = rcholb[iq]
        rcholbara_q = rcholbara[iq]
        rcholbarb_q = rcholbarb[iq]
        X[iq] = xp.einsum("kixp, kkwip -> xw", rchola_q, Ga_kpq, optimize=True) + xp.einsum("kixp, kkwip -> xw", rcholb_q, Gb_kpq, optimize=True)
        Xbar[iq] = xp.einsum("ksxj, kkwjs -> xw", rcholbara_q, GaT_kpq, optimize=True) + xp.einsum("ksxj, kkwjs -> xw", rcholbarb_q, GbT_kpq, optimize=True)

    for iq in range(len(Sset), unique_nq):
        iq_real = Qplus[iq - len(Sset)]
        ikpq_vec = kpq_mat[iq_real]
        Ga_kpq = Ghalfa[:, ikpq_vec, :, :, :]
        GaT_kpq = Ghalfa[ikpq_vec]
        Gb_kpq = Ghalfb[:, ikpq_vec, :, :, :]
        GbT_kpq = Ghalfb[ikpq_vec]
        rchola_q = rchola[iq]
        rcholb_q = rcholb[iq]
        rcholbara_q = rcholbara[iq]
        rcholbarb_q = rcholbarb[iq]
        X[iq] = xp.sqrt(2) * (xp.einsum("xkip, kkwip -> xw", rchola_q, Ga_kpq, optimize=True) + xp.einsum("xkip, kkwip -> xw", rcholb_q, Gb_kpq, optimize=True))
        Xbar[iq] = xp.sqrt(2) * (xp.einsum("xkqj, kkwjq -> xw", rcholbara_q, GaT_kpq, optimize=True) + xp.einsum("xkqj, kkwjq -> xw", rcholbarb_q, GbT_kpq, optimize=True))
        #TODO: possibly write a kernel for this

    ecoul = xp.einsum("qxw, qxw -> w", X, Xbar, optimize=True)

    return 0.5 * ecoul / nk

def kpt_symmchol_exx_kernel_batch_gpu(rchola_chunk, rcholbara_chunk, Ghalfa, GhalfaT, kpq_mat, Sset, Qplus, buff):
    # shape of rchola: (nq, nk, nocc, naux, nbsf) (q, k, i, gamma, p)
    # shape of rcholbara: (nq, nk, nbsf, naux, nocc) (q, k, p, gamma, i)
    # shape of Ghalf: (nk, nk, nw, nocc, nbsf)
    # shape of GhalfT: (nk, nk, nbsf, nocc, nw)
    # buff size: (nchol_chunk, nocc, nwalkers, nocc)
    nwalkers = Ghalfa.shape[2]
    unique_nq = len(Sset) + len(Qplus)
    nocc = rchola_chunk.shape[2]
    nchol = rchola_chunk.shape[3]
    nk = rchola_chunk.shape[1]
    exx = xp.zeros((nwalkers), dtype=numpy.complex128)
    nchol_chunk_size = buff.shape[0]
    # print(nchol_chunk_size)
    nchol_chunks = ceil(nchol / nchol_chunk_size)
    nchol_left = nchol
    # print(f"nchol_chunks: {nchol_chunks}, nchol_left: {nchol_left}")
    _buff = buff.ravel()
    for i in range(nchol_chunks):
        nchol_chunk = min(nchol_chunk_size, nchol_left)
        nchol_left -= nchol_chunk
        chol_sls = slice(i * nchol_chunk_size, i * nchol_chunk_size + nchol_chunk)
        # print(f"nchol_chunk: {nchol_chunk}, chol_sls: {chol_sls}")
        for iq in range(len(Sset)):
            iq_real = Sset[iq]
            for ik in range(nk):
                for ikprime in range(nk):
                    ikpr_pq = kpq_mat[iq_real, ikprime]
                    ik_pq = kpq_mat[iq_real, ik]
                    size = nwalkers * nchol_chunk * nocc * nocc
                    # alpha-alpha
                    rchola_chunk_ik = rchola_chunk[iq, ik][:, chol_sls, :].copy()
                    Lkq = rchola_chunk_ik.reshape(nchol_chunk * nocc, -1)
                    rcholbara_chunk_ikprime = rcholbara_chunk[iq, ikprime][:, chol_sls, :].copy()
                    Lbarkpq = rcholbara_chunk_ikprime.reshape(-1, nchol_chunk * nocc)
                    Ghalf_kpq_kprpq = GhalfaT[ik_pq, ikpr_pq].reshape(-1, nocc * nwalkers)
                    Ghalf_k_kp = Ghalfa[ik, ikprime].reshape(nwalkers * nocc, -1)
                    Txij = _buff[:size].reshape((nchol_chunk * nocc, nwalkers * nocc))
                    # T1 = Lkq @ Ghalf_kpq_kprpq # (naux * nocc, nocc * nwalkers)
                    xp.dot(Lkq, Ghalf_kpq_kprpq, out=Txij)
                    T1 = Txij.copy().reshape(nocc, nchol_chunk, nocc, nwalkers)
                    Txij = _buff[:size].reshape((nwalkers * nocc, nchol_chunk * nocc))
                    # T2 = Ghalf_k_kp @ Lbarkpq # (nwalkers * nocc, naux * nocc)
                    xp.dot(Ghalf_k_kp, Lbarkpq, out=Txij)
                    T2 = Txij.copy().reshape(nwalkers, nocc, nchol_chunk, nocc)
                    exchange_reduction_kpt(T1, T2, exx)

        for iq in range(len(Sset), unique_nq):
            iq_real = Qplus[iq - len(Sset)]
            for ik in range(nk):
                for ikprime in range(nk):
                    ikpr_pq = kpq_mat[iq_real, ikprime]
                    ik_pq = kpq_mat[iq_real, ik]
                    size = nwalkers * nchol_chunk * nocc * nocc
                    # alpha-alpha
                    rchola_chunk_ik = rchola_chunk[iq, ik][:, chol_sls, :].copy()
                    Lkq = xp.sqrt(2) * rchola_chunk_ik.reshape(nchol_chunk * nocc, -1)
                    rcholbara_chunk_ikprime = rcholbara_chunk[iq, ikprime][:, chol_sls, :].copy()
                    Lbarkpq = xp.sqrt(2) * rcholbara_chunk_ikprime.reshape(-1, nchol_chunk * nocc)
                    Ghalf_kpq_kprpq = GhalfaT[ik_pq, ikpr_pq].reshape(-1, nocc * nwalkers)
                    Ghalf_k_kp = Ghalfa[ik,ikprime].reshape(nwalkers * nocc, -1)
                    Txij = _buff[:size].reshape((nchol_chunk * nocc, nwalkers * nocc))
                    # T1 = Lkq @ Ghalf_kpq_kprpq # (naux * nocc, nocc * nwalkers)
                    xp.dot(Lkq, Ghalf_kpq_kprpq, out=Txij)
                    T1 = Txij.copy().reshape(nchol_chunk, nocc, nocc, nwalkers)
                    Txij = _buff[:size].reshape((nwalkers * nocc, nchol_chunk * nocc))
                    # T2 = Ghalf_k_kp @ Lbarkpq # (nwalkers * nocc, naux * nocc)
                    xp.dot(Ghalf_k_kp, Lbarkpq, out=Txij)
                    T2 = Txij.copy().reshape(nwalkers, nocc, nchol_chunk, nocc)
                    exchange_reduction_kpt(T1, T2, exx)
    return - 0.5 * exx / nk

def local_energy_kpt_single_det_uhf_batch_chunked_gpu(
    system: Generic,
    hamiltonian: KptComplexCholChunked,
    walker_batch: UHFWalkers,
    trial: KptSingleDet,
    max_mem: float = 2.0
):
    """Compute local energy for walker batch (all walkers at once).

    Single determinant case for k point Cholesky, GPU, chunked integrals.

    Parameters
    ----------
    system : system object
        System being studied.
    hamiltonian : hamiltonian object
        Hamiltonian being studied.
    walker_batch : WalkerBatch
        Walkers object.
    trial : trial object
        Trial wavefunctioni.

    Returns
    -------
    local_energy : np.ndarray
        Total, one-body and two-body energies.
    """
    assert hamiltonian.chunked

    nwalkers = walker_batch.Ghalfa.shape[0]
    nk = hamiltonian.nk
    nalpha = trial.nalpha
    nbeta = trial.nbeta
    nbasis = hamiltonian.nbasis

    ghalfa = walker_batch.Ghalfa.reshape(nwalkers, nk, nalpha, nk, nbasis)
    ghalfb = walker_batch.Ghalfb.reshape(nwalkers, nk, nbeta, nk, nbasis)
    ghalfaT = walker_batch.Ghalfa.transpose(0, 2, 1).reshape(nwalkers, nk, nbasis, nk, nalpha)
    ghalfbT = walker_batch.Ghalfb.transpose(0, 2, 1).reshape(nwalkers, nk, nbasis, nk, nbeta)

    diagGhalfa = xp.zeros((nwalkers, nk, nalpha, nbasis), dtype=numpy.complex128)
    diagGhalfb = xp.zeros((nwalkers, nk, nbeta, nbasis), dtype=numpy.complex128)
    for ik in range(nk):
        diagGhalfa[:, ik, :, :] = ghalfa[:, ik, :, ik, :]
        diagGhalfb[:, ik, :, :] = ghalfb[:, ik, :, ik, :]
    diagGhalfa = diagGhalfa.reshape(nwalkers, nk * nalpha * nbasis)
    diagGhalfb = diagGhalfb.reshape(nwalkers, nk * nbeta * nbasis)
    e1b = diagGhalfa.dot(trial._rH1a.ravel())
    e1b += diagGhalfb.dot(trial._rH1b.ravel())
    e1b /= nk
    e1b += hamiltonian.ecore

    ghalfa = ghalfa.transpose(1, 3, 0, 2, 4).copy() # nk, nk, nw, nalpha, nbasis
    ghalfb = ghalfb.transpose(1, 3, 0, 2, 4).copy() # nk, nk, nw, nbeta, nbasis
    ghalfaTcoul = ghalfaT.transpose(1, 3, 0, 2, 4).copy() # nk, nk, nw, nbasis, nalpha
    ghalfbTcoul = ghalfbT.transpose(1, 3, 0, 2, 4).copy() # nk, nk, nw, nbasis, nbeta
    ghalfaTx = ghalfaT.transpose(1, 3, 2, 4, 0).copy() # nk, nk, nbasis, nalpha, nw
    ghalfbTx = ghalfbT.transpose(1, 3, 2, 4, 0).copy() # nk, nk, nbasis, nbeta, nw

    ghalfa_send = ghalfa.copy()
    ghalfb_send = ghalfb.copy()
    ghalfaTcoul_send = ghalfaTcoul.copy()
    ghalfbTcoul_send = ghalfbTcoul.copy()
    ghalfaTx_send = ghalfaTx.copy()
    ghalfbTx_send = ghalfbTx.copy()

    ghalfa_recv = xp.zeros_like(ghalfa)
    ghalfb_recv = xp.zeros_like(ghalfb)
    ghalfaTcoul_recv = xp.zeros_like(ghalfaTcoul)
    ghalfbTcoul_recv = xp.zeros_like(ghalfbTcoul)
    ghalfaTx_recv = xp.zeros_like(ghalfaTx)
    ghalfbTx_recv = xp.zeros_like(ghalfbTx)

    handler = walker_batch.mpi_handler
    senders = handler.senders
    receivers = handler.receivers

    rchola_chunk = trial._rchola_chunk
    rcholb_chunk = trial._rcholb_chunk
    rcholbara_chunk = trial._rcholbara_chunk
    rcholbarb_chunk = trial._rcholbarb_chunk

    # buffer for low on GPU memory usage
    max_nchol = max(trial._rchola_chunk.shape[3], trial._rcholb_chunk.shape[3])
    max_nocc = max(nalpha, nbeta)
    mem_needed = 16 * nwalkers * max_nocc * max_nocc * max_nchol / (1024.0**3.0)
    num_chunks = max(1, ceil(mem_needed / max_mem))
    chunk_size = ceil(max_nchol / num_chunks)
    buff = xp.zeros(shape=(chunk_size, nwalkers * max_nocc * max_nocc), dtype=numpy.complex128)

    ecoul_send = kpt_symmchol_ecoul_kernel_batch_uhf_gpu(
        rchola_chunk, rcholb_chunk, rcholbara_chunk, rcholbarb_chunk, ghalfa, ghalfb, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus
    )

    exx_send = kpt_symmchol_exx_kernel_batch_gpu(rchola_chunk, rcholbara_chunk, ghalfa, ghalfaTx, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus, buff) 
    exx_send += kpt_symmchol_exx_kernel_batch_gpu(rcholb_chunk, rcholbarb_chunk, ghalfb, ghalfbTx, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus, buff) 

    exx_recv = exx_send.copy()
    ecoul_recv = ecoul_send.copy()

    for _ in range(handler.ssize - 1):
        for isend, sender in enumerate(senders):
            if handler.srank == isend:
                handler.scomm.Send(ghalfa_send, dest=receivers[isend], tag=1)
                handler.scomm.Send(ghalfb_send, dest=receivers[isend], tag=2)
                handler.scomm.Send(ghalfaTcoul_send, dest=receivers[isend], tag=3)
                handler.scomm.Send(ghalfbTcoul_send, dest=receivers[isend], tag=4)
                handler.scomm.Send(ghalfaTx_send, dest=receivers[isend], tag=5)
                handler.scomm.Send(ghalfbTx_send, dest=receivers[isend], tag=6)
                handler.scomm.Send(ecoul_send, dest=receivers[isend], tag=7)
                handler.scomm.Send(exx_send, dest=receivers[isend], tag=8)
            elif handler.srank == receivers[isend]:
                sender = numpy.where(receivers == handler.srank)[0]
                handler.scomm.Recv(ghalfa_recv, source=sender, tag=1)
                handler.scomm.Recv(ghalfb_recv, source=sender, tag=2)
                handler.scomm.Recv(ghalfaTcoul_recv, source=sender, tag=3)
                handler.scomm.Recv(ghalfbTcoul_recv, source=sender, tag=4)
                handler.scomm.Recv(ghalfaTx_recv, source=sender, tag=5)
                handler.scomm.Recv(ghalfbTx_recv, source=sender, tag=6)
                handler.scomm.Recv(ecoul_recv, source=sender, tag=7)
                handler.scomm.Recv(exx_recv, source=sender, tag=8)
        handler.scomm.barrier()

    # prepare sending
        ecoul_send = ecoul_recv.copy()
        ecoul_send += kpt_symmchol_ecoul_kernel_batch_uhf_gpu(
        rchola_chunk, rcholb_chunk, rcholbara_chunk, rcholbarb_chunk, ghalfa_recv, ghalfb_recv, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus
    )
        exx_send = exx_recv.copy()
        exx_send += kpt_symmchol_exx_kernel_batch_gpu(rchola_chunk, rcholbara_chunk, ghalfa_recv, ghalfaTx_recv, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus, buff)
        exx_send += kpt_symmchol_exx_kernel_batch_gpu(rcholb_chunk, rcholbarb_chunk, ghalfb_recv, ghalfbTx_recv, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus, buff)
        ghalfa_send = ghalfa_recv.copy()
        ghalfb_send = ghalfb_recv.copy()
        ghalfaTcoul_send = ghalfaTcoul_recv.copy()
        ghalfbTcoul_send = ghalfbTcoul_recv.copy()
        ghalfaTx_send = ghalfaTx_recv.copy()
        ghalfbTx_send = ghalfbTx_recv.copy()


    if len(senders) > 1:
        for isend, sender in enumerate(senders):
            if handler.srank == sender:  # sending 1 xshifted to 0 xshifted_buf
                handler.scomm.Send(ecoul_send, dest=receivers[isend], tag=1)
                handler.scomm.Send(exx_send, dest=receivers[isend], tag=2)
            elif handler.srank == receivers[isend]:
                sender = numpy.where(receivers == handler.srank)[0]
                handler.scomm.Recv(ecoul_recv, source=sender, tag=1)
                handler.scomm.Recv(exx_recv, source=sender, tag=2)

    e2b = ecoul_recv + exx_recv

    energy = xp.zeros((nwalkers, 3), dtype=numpy.complex128)
    energy[:, 0] = e1b + e2b
    energy[:, 1] = e1b
    energy[:, 2] = e2b

    return energy