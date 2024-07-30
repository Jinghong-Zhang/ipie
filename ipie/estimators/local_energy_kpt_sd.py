from math import ceil

import numpy
from numba import jit

from ipie.estimators.local_energy import local_energy_G
from ipie.estimators.kernels import exchange_reduction
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import synchronize

from ipie.systems.generic import Generic
from ipie.hamiltonians.kpt_hamiltonian import KptComplexChol
from ipie.walkers.uhf_walkers import UHFWalkers
from ipie.trial_wavefunction.single_det_kpt import KptSingleDet

from ipie.utils.kpt_conv import find_translated_index, find_inverted_index

import plum
# Note specialisations occur to because:
# 1. Numba does not allow for mixing types without a warning so need to split
# real and complex components apart when rchol is real. Green's function is
# complex in general.
# Optimize for case when wavefunction is RHF (factor of 2 saving)

@jit(nopython=True, fastmath=True)
def kpt_ecoul_kernel_rhf(rchola, Ghalfa_batch, kpq_mat, mq_vec):
    """Compute coulomb contribution for complex rchol with RHF trial.

    Parameters
    ----------
    rchol : :class:`numpy.ndarray`
        Half-rotated cholesky.
    Ghalf : :class:`numpy.ndarray`
        Walker's half-rotated "green's function" shape is nalpha  x nbasis
    kpts : :class:`numpy.ndarray`
        all k-points in fractional coordinates.

    Returns
    -------
    ecoul : :class:`numpy.ndarray`
        coulomb contribution for all walkers.
    """
    # sort out cupy later
    zeros = numpy.zeros
    dot = numpy.dot
    nwalkers = Ghalfa_batch.shape[0]

    # shape of rchola: (naux, nk, nocc, nk, nbsf) (gamma, k, i, q, p)
    # shape of Ghalf: (nw, nk, nocc, nk, nbsf)
    naux = rchola.shape[0]
    nocc = rchola.shape[2]
    nbsf = rchola.shape[4]
    nk = rchola.shape[1]
    ecoul = zeros(nwalkers, dtype=numpy.complex128)
    GhalfaT = Ghalfa_batch.transpose(0, 3, 4, 1, 2)
    X = zeros((nwalkers, naux, nk), dtype=numpy.complex128)
    for iq in range(nk):
        for ik in range(nk):
            ik_pq = kpq_mat[ik, iq]
            i_mq = mq_vec[iq]
            for iw in range(nwalkers):
                for g in range(naux):
                    X[iw, g, iq] += numpy.trace(dot(rchola[g, ik, :, iq, :], GhalfaT[iw, ik_pq, :, ik, :]))

    for iw in range(nwalkers):
        for q in range(nk):
            i_mq = mq_vec[q]
            ecoul[iw] += 2. * dot(X[iw, :, q], X[iw, :, i_mq])
    return ecoul/nk**2

@jit(nopython=True, fastmath=True)
def kpt_exx_kernel(rchola, Ghalfa_batch, kpq_mat, mq_vec):
    """Compute coulomb contribution for complex rchol with RHF trial.

    Parameters
    ----------
    rchol : :class:`numpy.ndarray`
        Half-rotated cholesky.
    Ghalf : :class:`numpy.ndarray`
        Walker's half-rotated "green's function" shape is nalpha  x nbasis
    kpts : :class:`numpy.ndarray`
        all k-points in fractional coordinates.

    Returns
    -------
    ecoul : :class:`numpy.ndarray`
        coulomb contribution for all walkers.
    """
    # sort out cupy later
    zeros = numpy.zeros
    dot = numpy.dot
    nwalkers = Ghalfa_batch.shape[0]

    # shape of rchola: (naux, nk, nocc, nk, nbsf) (gamma, k, i, q, p)
    # shape of Ghalf: (nw, nk, nocc, nk, nbsf)
    naux = rchola.shape[0]
    nocc = rchola.shape[2]
    nbsf = rchola.shape[4]
    nk = rchola.shape[1]
    exx = zeros(nwalkers, dtype=numpy.complex128)
    GhalfaT = Ghalfa_batch.transpose(0, 3, 4, 1, 2)

    T1 = zeros((nwalkers, naux, nocc, nocc), dtype=numpy.complex128)
    T2 = zeros((nwalkers, naux, nocc, nocc), dtype=numpy.complex128)
    for iq in range(nk):
        for ik in range(nk):
            for ikprime in range(nk):
                ikpr_pq = kpq_mat[ikprime, iq]
                ik_pq = kpq_mat[ik, iq]
                i_mq = mq_vec[iq]
                for n in range(nwalkers):
                    for g in range(naux):
                        T1[n, g] = dot(rchola[g, ik, :, iq, :], GhalfaT[n, ik_pq, :, ikpr_pq, :])
                        T2[n, g] = dot(rchola[g, ikpr_pq, :, i_mq, :], GhalfaT[n, ikprime, :, ik, :])
                        exx[n] += -numpy.trace(dot(T1[n, g], T2[n, g]))

    return 0.5 * exx / nk**2

@jit(nopython=True, fastmath=True)
def kpt_ecoul_kernel_uhf(rchola, rcholb, Ghalfa_batch, Ghalfb_batch, kpq_mat, mq_vec):
    """Compute coulomb contribution for real rchol with UHF trial.

    Parameters
    ----------
    rchola : :class:`numpy.ndarray`
        Half-rotated cholesky (alpha).
    rcholb : :class:`numpy.ndarray`
        Half-rotated cholesky (beta).
    Ghalfa : :class:`numpy.ndarray`
        Walker's half-rotated "green's function" shape is nalpha  x nbasis.
    Ghalfb : :class:`numpy.ndarray`
        Walker's half-rotated "green's function" shape is nbeta x nbasis.

    Returns
    -------
    ecoul : :class:`numpy.ndarray`
        coulomb contribution for all walkers.
    """
    # sort out cupy later
    zeros = numpy.zeros
    dot = numpy.dot
    nwalkers = Ghalfa_batch.shape[0]

    # shape of rchola: (naux, nk, nocc, nk, nbsf) (gamma, k, i, q, p)
    # shape of Ghalf: (nw, nk, nocc, nk, nbsf)
    naux = rchola.shape[0]
    nocca = rchola.shape[2]
    noccb = rcholb.shape[2]
    nbsf = rchola.shape[4]
    nk = rchola.shape[1]
    ecoul = zeros(nwalkers, dtype=numpy.complex128)
    GhalfaT = Ghalfa_batch.transpose(0, 3, 4, 1, 2)
    GhalfbT = Ghalfb_batch.transpose(0, 3, 4, 1, 2)
    X = zeros((nwalkers, naux, nk), dtype=numpy.complex128)
    for iq in range(nk):
        for ik in range(nk):
            ik_pq = kpq_mat[ik, iq]
            i_mq = mq_vec[iq]
            for iw in range(nwalkers):
                for g in range(naux):
                    X[iw, g, iq] += numpy.trace(dot(rchola[g, ik, :, iq, :], GhalfaT[iw, ik_pq, :, ik, :])) + numpy.trace(dot(rcholb[g, ik, :, iq, :], GhalfbT[iw, ik_pq, :, ik, :]))

    for iw in range(nwalkers):
        for q in range(nk):
            i_mq = mq_vec[q]
            ecoul[iw] += dot(X[iw, :, q], X[iw, :, i_mq])
    return 0.5 * ecoul / nk**2


@plum.dispatch
def local_energy_kpt_single_det_uhf(
    system: Generic,
    hamiltonian: KptComplexChol,
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
        Trial wavefunctioni.

    Returns
    -------
    local_energy : np.ndarray
        Total, one-body and two-body energies.
    """
    nwalkers = walkers.Ghalfa.shape[0]
    nk = hamiltonian.nk
    nalpha = trial.nalpha
    nbeta = trial.nbeta
    nbasis = hamiltonian.nbasis

    ghalfa = walkers.Ghalfa.reshape(nwalkers, nk, nalpha, nk, nbasis)
    ghalfb = walkers.Ghalfb.reshape(nwalkers, nk, nbeta, nk, nbasis)

    diagGhalfa = numpy.zeros((nwalkers, nk, nalpha, nbasis), dtype=numpy.complex128)
    diagGhalfb = numpy.zeros((nwalkers, nk, nbeta, nbasis), dtype=numpy.complex128)
    for ik in range(nk):
        diagGhalfa[:, ik, :, :] = ghalfa[:, ik, :, ik, :]
        diagGhalfb[:, ik, :, :] = ghalfb[:, ik, :, ik, :]
    e1b = numpy.einsum('wkip, kip -> w', diagGhalfa, trial._rH1a) # Ghalfa.dot(trial._rH1a.ravel())
    e1b += numpy.einsum('wkip, kip -> w', diagGhalfb, trial._rH1b)
    e1b /= nk
    e1b += hamiltonian.ecore

    ecoul = kpt_ecoul_kernel_uhf(
        trial._rchola, trial._rcholb, ghalfa, ghalfb, hamiltonian.ikpq_mat, hamiltonian.imq_vec
    )

    exx = kpt_exx_kernel(
        trial._rchola, ghalfa, hamiltonian.ikpq_mat, hamiltonian.imq_vec
    ) + kpt_exx_kernel(trial._rcholb, ghalfb, hamiltonian.ikpq_mat, hamiltonian.imq_vec)

    e2b = ecoul + exx

    energy = xp.zeros((nwalkers, 3), dtype=numpy.complex128)
    energy[:, 0] = e1b + e2b
    energy[:, 1] = e1b
    energy[:, 2] = e2b

    return energy