from math import ceil, sqrt

import numpy
from numba import jit

from ipie.estimators.local_energy import local_energy_G
from ipie.estimators.kernels import exchange_reduction
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import synchronize
from ipie.config import config
from ipie.utils.contract_gf_cgto import (
    slice_gf_k_kpq_qlis,
    slice_gf_kpq_k_qlis,
    slice_cgto_kpq,
)
from ipie.propagation.force_bias import contract_qkPp_kPr_qkwpr_to_qwP_cupy

from ipie.systems.generic import Generic
from ipie.hamiltonians.kpt_hamiltonian import KptComplexChol, KptComplexCholSymm, KptISDF
from ipie.walkers.uhf_walkers import UHFWalkers
from ipie.trial_wavefunction.single_det_kpt import KptSingleDet

import plum
# Note specialisations occur to because:
# 1. Numba does not allow for mixing types without a warning so need to split
# real and complex components apart when rchol is real. Green's function is
# complex in general.
# Optimize for case when wavefunction is RHF (factor of 2 saving)

@jit(nopython=True, fastmath=True)
def kpt_chol_ecoul_kernel_rhf(rchola, Ghalfa_batch, kpq_mat, mq_vec):
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
    return ecoul / nk

@jit(nopython=True, fastmath=True)
def kpt_chol_exx_kernel(rchola, Ghalfa_batch, kpq_mat, mq_vec):
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
    dot = numpy.dot
    nwalkers = Ghalfa_batch.shape[0]

    # shape of rchola: (naux, nk, nocc, nk, nbsf) (gamma, k, i, q, p)
    # shape of Ghalf: (nw, nk, nocc, nk, nbsf)
    naux = rchola.shape[0]
    nocc = rchola.shape[2]
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

    return 0.5 * exx / nk

@jit(nopython=True, fastmath=True)
def kpt_chol_ecoul_kernel_uhf(rchola, rcholb, Ghalfa_batch, Ghalfb_batch, kpq_mat, mq_vec):
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
    return 0.5 * ecoul  / nk

@jit(nopython=True, fastmath=True)
def kpt_symmchol_ecoul_kernel_rhf(rchola, rcholbara, Ghalfa, GhalfaT, kpq_mat, Sset, Qplus):
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
    multiply = numpy.multiply
    nwalkers = Ghalfa.shape[2]

    # shape of rchola: (nq, nk, nocc, naux, nbsf) (q, k, gamma, i, p)
    # shape of Ghalf: (nk, nk, nw, nocc, nbsf)
    unique_nq = len(Sset) + len(Qplus)
    nbsf = rchola.shape[4]
    nocc = rchola.shape[2]
    naux = rchola.shape[3]
    nk = rchola.shape[1]
    rchola = rchola.transpose(0, 1, 3, 2, 4).copy()
    rcholbara = rcholbara.transpose(0, 1, 3, 2, 4).copy()
    ecoul = zeros(nwalkers, dtype=numpy.complex128)
    X = zeros((unique_nq, nwalkers, naux), dtype=numpy.complex128)
    Xbar = zeros((unique_nq, nwalkers, naux), dtype=numpy.complex128)
    for iq in range(len(Sset)):
        iq_real = Sset[iq]
        Xq = X[iq]
        Xbarq = Xbar[iq]
        for ik in range(nk):
            ik_pq = kpq_mat[iq_real, ik]
            La = rchola[iq, ik].reshape(naux,nocc*nbsf)
            Lbara = rcholbara[iq, ik].reshape(naux,nocc*nbsf)
            for iw in range(nwalkers):
                Ghalfa_k_kpq = Ghalfa[ik, ik_pq, iw].reshape(nocc*nbsf)
                GhalfTa_k_kpq = GhalfaT[ik, ik_pq, iw].reshape(nocc*nbsf)
                Xq[iw] += 2.0 * La @ Ghalfa_k_kpq 
                Xbarq[iw] += 2.0 * Lbara @ GhalfTa_k_kpq

    for iq in range(len(Sset), len(Sset) + len(Qplus)):
        iq_real = Qplus[iq - len(Sset)]
        Xq = X[iq]
        Xbarq = Xbar[iq]
        for ik in range(nk):
            ik_pq = kpq_mat[iq_real, ik]
            La = rchola[iq, ik].reshape(naux,nocc*nbsf)
            Lbara = rcholbara[iq, ik].reshape(naux,nocc*nbsf)
            for iw in range(nwalkers):
                Ghalfa_k_kpq = Ghalfa[ik, ik_pq, iw].reshape(nocc*nbsf)
                GhalfTa_k_kpq = GhalfaT[ik, ik_pq, iw].reshape(nocc*nbsf)
                Xq[iw] += 2.0 * sqrt(2) * La @ Ghalfa_k_kpq
                Xbarq[iw] += 2.0 * sqrt(2) * Lbara @ GhalfTa_k_kpq

    X = X.transpose(1, 0, 2).copy()
    Xbar = Xbar.transpose(1, 0, 2).copy()
    X = X.reshape(nwalkers, naux * unique_nq)
    Xbar = Xbar.reshape(nwalkers, naux * unique_nq)
    for iw in range(nwalkers):
        ecoul[iw] = dot(X[iw], Xbar[iw])
    return 0.5 * ecoul / nk

@jit(nopython=True, fastmath=True) #, parallel=True
def kpt_symmchol_exx_kernel_lowmem(rchola, rcholbara, Ghalfa, GhalfaT, kpq_mat, Sset, Qplus):
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
    nwalkers = Ghalfa.shape[2]

    # shape of rchola: (nq, nk, naux, nocc, nbsf) 
    # shape of Ghalf: (nk, nk, nw, nocc, nbsf)
    naux = rchola.shape[2]
    nocc = rchola.shape[3]
    nk = rchola.shape[1]
    exx = zeros(nwalkers, dtype=numpy.complex128)

    for iq in range(len(Sset)):
        iq_real = Sset[iq]        
        for ik in range(nk):
            ik_pq = kpq_mat[iq_real, ik]
            for ikprime in range(nk):
                ikpr_pq = kpq_mat[iq_real, ikprime]
                for iw in range(nwalkers):
                    Ghalf_kpq_kprpq = GhalfaT[ik_pq, ikpr_pq, iw]
                    Ghalf_k_kp = Ghalfa[ik,ikprime, iw]
                    for g in range(naux):
                        Lkqg = rchola[iq, ik, :, g].transpose(1, 0, 2).copy()
                        Lbarkpqg = rcholbara[iq, ikprime, :, g].transpose(1, 0, 2).copy()
                        T1 = Lkqg @ Ghalf_kpq_kprpq
                        T2 = Ghalf_k_kp @ Lbarkpqg
                        for i in range(nocc):
                            for j in range(nocc):
                                exx[iw] -= T1[i, j] * T2[i, j]

    for iq in range(len(Sset), len(Sset) + len(Qplus)):
        iq_real = Qplus[iq - len(Sset)]
        for ik in range(nk):
            Lkq = rchola[iq, ik].transpose(1, 0, 2).copy()
            ik_pq = kpq_mat[iq_real, ik]
            for ikprime in range(nk):
                ikpr_pq = kpq_mat[iq_real, ikprime]
                Lbarkpq = rcholbara[iq, ikprime].transpose(1, 0, 2).copy()
                for iw in range(nwalkers):
                    Ghalf_kpq_kprpq = GhalfaT[ik_pq, ikpr_pq, iw]
                    Ghalf_k_kp = Ghalfa[ik, ikprime, iw]
                    for g in range(naux):
                        T1 = Lkq[g] @ Ghalf_kpq_kprpq
                        T2 = Ghalf_k_kp @ Lbarkpq[g]
                        for i in range(nocc):
                            for j in range(nocc):
                                exx[iw] -= 2. * T1[i, j] * T2[i, j]


@jit(nopython=True, fastmath=True) #, parallel=True
def kpt_symmchol_exx_kernel(rchola, rcholbara, Ghalfa, GhalfaT, kpq_mat, Sset, Qplus):
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
    nwalkers = Ghalfa.shape[2]

    # shape of rchola: (nq, nk, naux, nocc, nbsf) -> (nq, nk, nocc, naux, nbsf)
    # shape of Ghalf: (nk(nocc), nk(nbsf), nw, nocc, nbsf)
    # shape of GhalfT: (nk(nbsf), nk(nocc), nw, nbsf, nocc) -> (nk, nk, nbsf, nocc, nw)
    naux = rchola.shape[3]
    nocc = rchola.shape[2]
    nk = rchola.shape[1]
    exx = zeros(nwalkers, dtype=numpy.complex128)
    for iq in range(len(Sset)):
        iq_real = Sset[iq]
        for ik in range(nk):
            for ikprime in range(nk):
                ikpr_pq = kpq_mat[iq_real, ikprime]
                ik_pq = kpq_mat[iq_real, ik]
                Lkq = rchola[iq, ik].reshape(naux * nocc, -1)
                Lbarkpq = rcholbara[iq, ikprime].reshape(-1, naux * nocc)
                Ghalf_kpq_kprpq = GhalfaT[ik_pq, ikpr_pq].reshape(-1, nocc * nwalkers)
                Ghalf_k_kp = Ghalfa[ik,ikprime].reshape(nwalkers * nocc, -1)
                T1 = Lkq @ Ghalf_kpq_kprpq # (naux * nocc, nocc * nwalkers)
                T2 = Ghalf_k_kp @ Lbarkpq # (nwalkers * nocc, naux * nocc)
                T1 = T1.reshape(naux * nocc * nocc, nwalkers).T.copy()
                T2 = T2.reshape(nwalkers, naux * nocc * nocc).copy()
                for iw in range(nwalkers):
                    exx[iw] += -T1[iw] @ T2[iw]

    for iq in range(len(Sset), len(Sset) + len(Qplus)):
        iq_real = Qplus[iq - len(Sset)]
        for ik in range(nk):
            for ikprime in range(nk):
                ikpr_pq = kpq_mat[iq_real, ikprime]
                ik_pq = kpq_mat[iq_real, ik]
                Lkq = rchola[iq, ik].reshape(naux * nocc, -1)
                Lbarkpq = rcholbara[iq, ikprime].reshape(-1, naux * nocc)
                Ghalf_kpq_kprpq = GhalfaT[ik_pq, ikpr_pq].reshape(-1, nocc * nwalkers)
                Ghalf_k_kp = Ghalfa[ik,ikprime].reshape(nwalkers * nocc, -1)
                T1 = Lkq @ Ghalf_kpq_kprpq # (naux * nocc, nocc * nwalkers)
                T2 = Ghalf_k_kp @ Lbarkpq # (nwalkers * nocc, naux * nocc)
                T1 = T1.reshape(naux * nocc * nocc, nwalkers).T.copy()
                T2 = T2.reshape(nwalkers, naux * nocc * nocc).copy()
                for iw in range(nwalkers):
                    exx[iw] += - 2. * T1[iw] @ T2[iw]

    return 0.5 * exx / nk

@jit(nopython=True, fastmath=True)
def kpt_symmchol_ecoul_kernel_uhf(rchola, rcholb, rcholbara, rcholbarb, Ghalfa, Ghalfb, GhalfaT, GhalfbT, kpq_mat, Sset, Qplus):
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
    multiply = numpy.multiply
    nwalkers = Ghalfa.shape[2]

    # shape of rchola: (nq, nk, nocc, naux, nbsf) (q, k, gamma, i, p)
    # shape of Ghalf: (nk, nk, nw, nocc, nbsf)
    unique_nq = len(Sset) + len(Qplus)
    nbsf = rchola.shape[4]
    nocc = rchola.shape[2]
    naux = rchola.shape[3]
    nk = rchola.shape[1]
    rchola = rchola.transpose(0, 1, 3, 2, 4).copy()
    rcholb = rcholb.transpose(0, 1, 3, 2, 4).copy()
    rcholbara = rcholbara.transpose(0, 1, 3, 2, 4).copy()
    rcholbarb = rcholbarb.transpose(0, 1, 3, 2, 4).copy()
    ecoul = zeros(nwalkers, dtype=numpy.complex128)
    X = zeros((unique_nq, nwalkers, naux), dtype=numpy.complex128)
    Xbar = zeros((unique_nq, nwalkers, naux), dtype=numpy.complex128)
    for iq in range(len(Sset)):
        iq_real = Sset[iq]
        Xq = X[iq]
        Xbarq = Xbar[iq]
        for ik in range(nk):
            ik_pq = kpq_mat[iq_real, ik]
            La = rchola[iq, ik].reshape(naux,nocc*nbsf)
            Lb = rcholb[iq, ik].reshape(naux,nocc*nbsf)
            Lbara = rcholbara[iq, ik].reshape(naux,nocc*nbsf)
            Lbarb = rcholbarb[iq, ik].reshape(naux,nocc*nbsf)
            for iw in range(nwalkers):
                Ghalfa_k_kpq = Ghalfa[ik, ik_pq, iw].reshape(nocc*nbsf)
                GhalfTa_k_kpq = GhalfaT[ik, ik_pq, iw].reshape(nocc*nbsf)
                Ghalfb_k_kpq = Ghalfb[ik, ik_pq, iw].reshape(nocc*nbsf)
                GhalfTb_k_kpq = GhalfbT[ik, ik_pq, iw].reshape(nocc*nbsf)
                Xq[iw] += La @ Ghalfa_k_kpq + Lb @ Ghalfb_k_kpq 
                Xbarq[iw] += Lbara @ GhalfTa_k_kpq + Lbarb @ GhalfTb_k_kpq

    for iq in range(len(Sset), len(Sset) + len(Qplus)):
        iq_real = Qplus[iq - len(Sset)]
        Xq = X[iq]
        Xbarq = Xbar[iq]
        for ik in range(nk):
            ik_pq = kpq_mat[iq_real, ik]
            La = rchola[iq, ik].reshape(naux,nocc*nbsf)
            Lb = rcholb[iq, ik].reshape(naux,nocc*nbsf)
            Lbara = rcholbara[iq, ik].reshape(naux,nocc*nbsf)
            Lbarb = rcholbarb[iq, ik].reshape(naux,nocc*nbsf)
            for iw in range(nwalkers):
                Ghalfa_k_kpq = Ghalfa[ik, ik_pq, iw].reshape(nocc*nbsf)
                GhalfTa_k_kpq = GhalfaT[ik, ik_pq, iw].reshape(nocc*nbsf)
                Ghalfb_k_kpq = Ghalfb[ik, ik_pq, iw].reshape(nocc*nbsf)
                GhalfTb_k_kpq = GhalfbT[ik, ik_pq, iw].reshape(nocc*nbsf)
                Xq[iw] += sqrt(2) * (La @ Ghalfa_k_kpq + Lb @ Ghalfb_k_kpq)
                Xbarq[iw] += sqrt(2) * (Lbara @ GhalfTa_k_kpq + Lbarb @ GhalfTb_k_kpq)

    X = X.transpose(1, 0, 2).copy()
    Xbar = Xbar.transpose(1, 0, 2).copy()
    X = X.reshape(nwalkers, naux * unique_nq)
    Xbar = Xbar.reshape(nwalkers, naux * unique_nq)
    for iw in range(nwalkers):
        ecoul[iw] = dot(X[iw], Xbar[iw])
    return 0.5 * ecoul / nk

def _available_device_mem_gb():
    """Free device memory in GB, counting CuPy pool blocks as free."""
    if not config.get_option("use_gpu"):
        return 4.0
    free_bytes = xp.cuda.Device().mem_info[0]
    pool_bytes = xp.get_default_memory_pool().free_bytes()
    return (free_bytes + pool_bytes) / 1024**3

def _kpt_isdf_ecoul_batch(MPQ_block, iq_lis, halfrot_lis, cgto, ghalf_lis, kpq_mat, max_mem_gb):
    """Coulomb contribution of a batch of q points (no weight or prefactor).

    halfrot_lis / ghalf_lis hold the per-spin half-rotated cgto (k, P, i) and
    Green's functions (w, k, i, k', p); spins are fused along the occupied
    index so each contraction is a single chunked-GEMM kernel call.
    """
    cgto_kpq = slice_cgto_kpq(cgto, kpq_mat, iq_lis)
    # v1[q, w, P] = sum_{k,i,r} rcgto*[k,P,i] cgto[k+q,P,r] G[w,k,i,k+q,r]
    g_lis = [slice_gf_k_kpq_qlis(g, iq_lis, kpq_mat).transpose(0, 1, 2, 4, 3) for g in ghalf_lis]
    g_k_kpq = g_lis[0] if len(g_lis) == 1 else xp.concatenate(g_lis, axis=-1)
    del g_lis
    h_lis = [h.conj() for h in halfrot_lis]
    halfrot_cat = h_lis[0] if len(h_lis) == 1 else xp.concatenate(h_lis, axis=-1)
    v1 = contract_qkPp_kPr_qkwpr_to_qwP_cupy(cgto_kpq, halfrot_cat, g_k_kpq, max_mem=max_mem_gb)
    del g_k_kpq, halfrot_cat
    # v2[q, w, P] = sum_{k,i,r} rcgto*[k+q,P,i] cgto[k,P,r] G[w,k+q,i,k,r]
    g_lis = [slice_gf_kpq_k_qlis(g, iq_lis, kpq_mat) for g in ghalf_lis]
    g_kpq_k = g_lis[0] if len(g_lis) == 1 else xp.concatenate(g_lis, axis=3)
    del g_lis
    h_lis = [slice_cgto_kpq(h, kpq_mat, iq_lis).conj() for h in halfrot_lis]
    rcgto_kpq_cat = h_lis[0] if len(h_lis) == 1 else xp.concatenate(h_lis, axis=-1)
    v2 = contract_qkPp_kPr_qkwpr_to_qwP_cupy(rcgto_kpq_cat, cgto, g_kpq_k, max_mem=max_mem_gb)
    del g_kpq_k, rcgto_kpq_cat
    return xp.sum(xp.matmul(v1, MPQ_block) * v2, axis=(0, 2))

def _kpt_isdf_ecoul_all_q(MPQ, halfrot_lis, cgto, ghalf_lis, kpq_mat, Sset, Qplus, max_mem_gb):
    nk = cgto.shape[0]
    nbsf = cgto.shape[-1]
    nwalkers = ghalf_lis[0].shape[0]
    nocc_tot = sum(g.shape[2] for g in ghalf_lis)
    if max_mem_gb is None:
        max_mem_gb = 0.25 * _available_device_mem_gb()
    budget_bytes = int(max(float(max_mem_gb), 0.05) * 1024**3)
    per_q_bytes = 4 * nk * nwalkers * nocc_tot * nbsf * 16
    nq_chunk = max(1, budget_bytes // per_q_bytes)
    ecoul = xp.zeros(nwalkers, dtype=numpy.complex128)
    nS = len(Sset)
    for start in range(0, nS, nq_chunk):
        stop = min(start + nq_chunk, nS)
        ecoul += _kpt_isdf_ecoul_batch(
            MPQ[start:stop], Sset[start:stop], halfrot_lis, cgto, ghalf_lis, kpq_mat, max_mem_gb
        )
    for start in range(0, len(Qplus), nq_chunk):
        stop = min(start + nq_chunk, len(Qplus))
        ecoul += 2.0 * _kpt_isdf_ecoul_batch(
            MPQ[nS + start : nS + stop], Qplus[start:stop], halfrot_lis, cgto, ghalf_lis, kpq_mat, max_mem_gb
        )
    return ecoul

def kpt_isdf_ecoul_kernel_gpu(MPQ, halfrot_cgtoa, halfrot_cgtob, cgto, Ghalfa_batch, Ghalfb_batch, kpq_mat, Sset, Qplus, max_mem_gb=None):
    nk = cgto.shape[0]
    ecoul = _kpt_isdf_ecoul_all_q(
        MPQ,
        [halfrot_cgtoa, halfrot_cgtob],
        cgto,
        [Ghalfa_batch, Ghalfb_batch],
        kpq_mat,
        Sset,
        Qplus,
        max_mem_gb,
    )
    return 0.5 * ecoul / nk

def kpt_isdf_ecoul_rhf_kernel_gpu(MPQ, halfrot_cgtoa, cgto, Ghalfa_batch, kpq_mat, Sset, Qplus, max_mem_gb=None):
    nk = cgto.shape[0]
    ecoul = _kpt_isdf_ecoul_all_q(
        MPQ, [halfrot_cgtoa], cgto, [Ghalfa_batch], kpq_mat, Sset, Qplus, max_mem_gb
    )
    return 2. * ecoul / nk

def kpt_isdf_exx_largek_q(halfrot_cgtoa, phikr_kpq, M_PQ_iq, phiki_kpq, cgto, Ghalfa_batch, GkpqT, max_mem_gb=4.0):
    """Exchange contribution of one q point (large-nk algorithm).

    Evaluates 'kPi, kPp, PQ, KQj, KQq, wkiKq, wKjkp -> w' with the
    walker-independent GEMM A = (rcgto* x cgto_kpq) @ M hoisted out of the
    walker loop.

    GkpqT : (K, j, w, k, p) contiguous transpose of
        Ghalf[w, kpq[K], j, kpq[k], p].
    """
    nw, nk, nocc, _, nbsf = Ghalfa_batch.shape
    nisdf = M_PQ_iq.shape[0]
    itemsize = 16
    budget = int(max(float(max_mem_gb), 0.05) * 1024**3)
    exx = xp.zeros(nw, dtype=xp.complex128)
    Ga_flat = Ghalfa_batch.reshape(nw, -1)

    # Q chunk sized by the walker-independent intermediates (A, its transpose
    # copy and the rho chunks); P chunk bounds the rho build the same way
    nQ_chunk = max(1, min(nisdf, budget // (4 * nk * nocc * nbsf * itemsize)))
    nP_chunk = nQ_chunk
    # walker chunk sized by B (and its transpose copy) plus the G slice and D
    per_w_bytes = nk * nk * nbsf * (2 * nQ_chunk + 3 * nocc) * itemsize
    nw_chunk = max(1, min(nw, budget // max(per_w_bytes, 1)))

    for qstart in range(0, nisdf, nQ_chunk):
        qstop = min(qstart + nQ_chunk, nisdf)
        nQ = qstop - qstart
        # A[kip, Q] = sum_P rcgto*[k,P,i] cgto_kpq[k,P,p] M[P,Q]
        A = xp.zeros((nk * nocc * nbsf, nQ), dtype=xp.complex128)
        for pstart in range(0, nisdf, nP_chunk):
            pstop = min(pstart + nP_chunk, nisdf)
            rho = halfrot_cgtoa[:, pstart:pstop, :, None].conj() * phikr_kpq[:, pstart:pstop, None, :]  # k, P, i, p
            rho = rho.transpose(0, 2, 3, 1).reshape(nk * nocc * nbsf, pstop - pstart)
            A += rho @ M_PQ_iq[pstart:pstop, qstart:qstop].astype(xp.complex128, copy=False)
            del rho
        A = A.reshape(nk, nocc, nbsf, nQ).transpose(0, 3, 2, 1).reshape(nk * nQ, nbsf, nocc)  # kQ, p, i
        psi_kQj = xp.ascontiguousarray(phiki_kpq[:, qstart:qstop, :].conj())  # K, Q, j
        cgto_Q = cgto[:, qstart:qstop, :]  # K, Q, q
        for wstart in range(0, nw, nw_chunk):
            wstop = min(wstart + nw_chunk, nw)
            wchunk = wstop - wstart
            Gk = xp.ascontiguousarray(GkpqT[:, :, wstart:wstop]).reshape(nk, nocc, wchunk * nk * nbsf)
            B = xp.matmul(psi_kQj, Gk)  # K, Q, wkp
            del Gk
            B = B.reshape(nk, nQ, wchunk, nk, nbsf).transpose(3, 1, 2, 0, 4).reshape(nk * nQ, wchunk * nk, nbsf)  # kQ, wK, p
            C = xp.matmul(B, A)  # kQ, wK, i
            del B
            C = C.reshape(nk, nQ, wchunk, nk, nocc).transpose(3, 0, 2, 4, 1).reshape(nk, nk * wchunk * nocc, nQ)  # K, kwi, Q
            D = xp.matmul(C, cgto_Q)  # K, kwi, q
            del C
            D = D.reshape(nk, nk, wchunk, nocc, nbsf).transpose(2, 1, 3, 0, 4).reshape(wchunk, -1)  # w, kiKq
            exx[wstart:wstop] += xp.sum(D * Ga_flat[wstart:wstop], axis=-1)
            del D
    return exx

def _psi_G_psi_stage1(GT, psi_qQ, nw, nk, nocc):
    """First GEMM of psi+ . G . psi: (k', wki, q) @ (k', q, Q) -> (k, wQk', i)."""
    nQ = psi_qQ.shape[-1]
    Gpsi = xp.matmul(GT, psi_qQ)  # k', wki, Q
    Gpsi = Gpsi.reshape(nk, nw, nk, nocc, nQ).transpose(2, 1, 4, 0, 3).reshape(nk, nw * nQ * nk, nocc)
    return Gpsi  # k, wQk', i

def kpt_isdf_exx_lowk_q(halfrot_cgtoa, phikr_kpq, M_PQ_iq, phiki_kpq, cgto, GaT, GkpqT, nw, nocc, max_mem_gb=4.0):
    """Exchange contribution of one q point (low-nk algorithm).

    The psi+ . G . psi contractions are split so the GEMM that depends on only
    one ISDF chunk index is hoisted out of the double chunk loop, and the
    (P, Q) reduction is fused into two elementwise passes plus a sum.

    GaT : (k', w*k*i, q) contiguous transpose of Ghalf (q-independent).
    GkpqT : same layout for the doubly-shifted Green's function of this q.
    """
    nk = cgto.shape[0]
    nisdf = M_PQ_iq.shape[0]
    itemsize = 16
    budget = int(max(float(max_mem_gb), 0.05) * 1024**3)
    exx = xp.zeros(nw, dtype=xp.complex128)

    # chunk both ISDF indices so the (k, w, Q, k', P) intermediates fit
    nchunk = max(1, min(nisdf, int(sqrt(budget / max(4 * nk * nk * nw * itemsize, 1)))))
    slices_isdf = [slice(s, min(s + nchunk, nisdf)) for s in range(0, nisdf, nchunk)]

    # the stage-1 GEMM of T_PQ depends only on the Q chunk; cache it across
    # the P loop when the full set fits in the budget
    stage1_bytes = nk * nk * nw * nocc * nisdf * itemsize
    cache_stage1 = stage1_bytes <= budget
    TPQ_stage1 = None
    if cache_stage1:
        TPQ_stage1 = [
            _psi_G_psi_stage1(GaT, cgto[:, Q, :].transpose(0, 2, 1), nw, nk, nocc)
            for Q in slices_isdf
        ]

    for P in slices_isdf:
        nP = P.stop - P.start
        psi_iP_k = halfrot_cgtoa[:, P, :].transpose(0, 2, 1).conj()  # k, i, P
        phip_kpq_P = phikr_kpq[:, P, :].transpose(0, 2, 1)  # k+q, p, P
        TQP_s1 = _psi_G_psi_stage1(GkpqT, phip_kpq_P, nw, nk, nocc)  # K, wPK', j
        for ib, Q in enumerate(slices_isdf):
            nQ = Q.stop - Q.start
            if cache_stage1:
                TPQ_s1 = TPQ_stage1[ib]
            else:
                TPQ_s1 = _psi_G_psi_stage1(GaT, cgto[:, Q, :].transpose(0, 2, 1), nw, nk, nocc)
            TPQ = xp.matmul(TPQ_s1, psi_iP_k)  # k, wQk', P
            phij_kpq_Q = phiki_kpq[:, Q, :].transpose(0, 2, 1).conj()  # k'+q, j, Q
            TQP = xp.matmul(TQP_s1, phij_kpq_Q)  # K, wPK', Q
            # exx[w] += sum TPQ[k,w,Q,k',P] M[P,Q] TQP[k',w,P,k,Q]
            # (contiguous transpose copies keep the product coalesced)
            TPQ = TPQ.reshape(nk, nw, nQ, nk, nP).transpose(1, 0, 3, 4, 2).reshape(nw, nk * nk, nP, nQ)
            TQP = TQP.reshape(nk, nw, nP, nk, nQ).transpose(1, 3, 0, 2, 4).reshape(nw, nk * nk, nP, nQ)
            Tsq = xp.sum(TPQ * TQP, axis=1)  # w, P, Q
            M_sliced = M_PQ_iq[P, Q].astype(xp.complex128, copy=False)
            exx += Tsq.reshape(nw, nP * nQ) @ M_sliced.ravel()
            del Tsq, TPQ, TQP
    return exx

def kpt_isdf_exx_kernel_gpu(MPQ, halfrot_cgtoa, cgto, Ghalfa_batch, kpq_mat, Sset, Qplus, algo=None, max_mem_gb=None):
    nwalker, nk, nocc, _, nbsf = Ghalfa_batch.shape
    nisdf = MPQ.shape[-1]
    if max_mem_gb is None:
        max_mem_gb = 0.25 * _available_device_mem_gb()
    if algo is None:
        # per-q cost model: the large-k path pays a walker-independent
        # nk*nocc*nbsf*nisdf^2 GEMM but only 3 walker-dependent GEMMs of
        # nk^2*nw*nocc*nbsf*nisdf; the low-k path scales as nisdf^2 per walker.
        # The low-k reduction also streams the (nw, nk^2, nP, nQ) product
        # tensors through memory (bandwidth-bound, ~48*nw*nk^2*nisdf^2 bytes),
        # which dominates its runtime well past the pure-FLOP crossover; charge
        # it as flop-equivalents with a coefficient calibrated on an H200 sweep
        # (nk 8-125, nbsf 20-1000; feasible range 88-1231, mispick <= 8%).
        flops_largek = nk * nocc * nbsf * nisdf**2 + 3 * nk**2 * nwalker * nocc * nbsf * nisdf
        flops_lowk = 2 * nk**2 * nwalker * nocc * nisdf**2 + 2 * nk**2 * nwalker * nocc * nbsf * nisdf
        bw_penalty_lowk = 330 * nwalker * nk**2 * nisdf**2
        algo = "largek" if flops_largek <= flops_lowk + bw_penalty_lowk else "lowk"

    exx = xp.zeros(nwalker, dtype=numpy.complex128)
    GaT = None
    if algo == "lowk":
        GaT = xp.ascontiguousarray(Ghalfa_batch.transpose(3, 0, 1, 2, 4)).reshape(
            nk, nwalker * nk * nocc, nbsf
        )
    nS = len(Sset)
    for iq in range(nS + len(Qplus)):
        iq_real = Sset[iq] if iq < nS else Qplus[iq - nS]
        weight = 1.0 if iq < nS else 2.0
        ikpq = kpq_mat[iq_real]
        phikr_kpq = cgto[ikpq]
        phiki_kpq = halfrot_cgtoa[ikpq]
        # G_kpq[w, K, j, k, p] = Ghalf[w, kpq[K], j, kpq[k], p]
        G_kpq = Ghalfa_batch.take(ikpq, axis=1).take(ikpq, axis=3)
        if algo == "largek":
            GkpqT = xp.ascontiguousarray(G_kpq.transpose(1, 2, 0, 3, 4))  # K, j, w, k, p
            del G_kpq
            exx -= weight * kpt_isdf_exx_largek_q(
                halfrot_cgtoa, phikr_kpq, MPQ[iq], phiki_kpq, cgto, Ghalfa_batch, GkpqT, max_mem_gb
            )
        else:
            GkpqT = xp.ascontiguousarray(G_kpq.transpose(3, 0, 1, 2, 4)).reshape(
                nk, nwalker * nk * nocc, nbsf
            )
            del G_kpq
            exx -= weight * kpt_isdf_exx_lowk_q(
                halfrot_cgtoa, phikr_kpq, MPQ[iq], phiki_kpq, cgto, GaT, GkpqT, nwalker, nocc, max_mem_gb
            )
        del GkpqT
    return 0.5 * exx / nk


def kpt_isdf_ecoul_kernel_rhf():
    raise NotImplementedError("CPU ISDF Coulomb kernel for RHF not implemented yet.")
    

@jit(nopython=True, fastmath=True)
def kpt_isdf_ecoul_kernel_uhf():
    raise NotImplementedError("CPU ISDF Coulomb kernel for UHF not implemented yet.")

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
        Trial wavefunction.

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

    ecoul = kpt_chol_ecoul_kernel_uhf(
        trial._rchola, trial._rcholb, ghalfa, ghalfb, hamiltonian.ikpq_mat, hamiltonian.imq_vec
    )

    exx = kpt_chol_exx_kernel(
        trial._rchola, ghalfa, hamiltonian.ikpq_mat, hamiltonian.imq_vec
    ) + kpt_chol_exx_kernel(trial._rcholb, ghalfb, hamiltonian.ikpq_mat, hamiltonian.imq_vec)

    e2b = ecoul + exx

    energy = xp.zeros((nwalkers, 3), dtype=numpy.complex128)
    energy[:, 0] = e1b + e2b
    energy[:, 1] = e1b
    energy[:, 2] = e2b

    return energy

@plum.dispatch
def local_energy_kpt_single_det_uhf(
    system: Generic,
    hamiltonian: KptComplexCholSymm,
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
    nwalkers = walkers.Ghalfa.shape[0]
    nk = hamiltonian.nk
    nalpha = trial.nalpha
    nbeta = trial.nbeta
    nbasis = hamiltonian.nbasis

    if walkers.rhf:
        ghalfa = walkers.Ghalfa.reshape(nwalkers, nk, nalpha, nk, nbasis)
        ghalfaT = walkers.Ghalfa.transpose(0, 2, 1).reshape(nwalkers, nk, nbasis, nk, nalpha)

        diagGhalfa = numpy.zeros((nwalkers, nk, nalpha, nbasis), dtype=numpy.complex128)
        for ik in range(nk):
            diagGhalfa[:, ik, :, :] = ghalfa[:, ik, :, ik, :]
        e1b = 2.0 * numpy.einsum('wkip, kip -> w', diagGhalfa, trial._rH1a) # Ghalfa.dot(trial._rH1a.ravel())
        e1b /= nk
        e1b += hamiltonian.ecore

        ghalfa = ghalfa.transpose(1, 3, 0, 2, 4).copy() # nk, nk, nw, nalpha, nbasis
        ghalfaTcoul = ghalfaT.transpose(1, 3, 0, 2, 4).copy() # nk, nk, nw, nbasis, nalpha
        ghalfaTx = ghalfaT.transpose(1, 3, 2, 4, 0).copy() # nk, nk, nbasis, nalpha, nw

        ecoul = kpt_symmchol_ecoul_kernel_rhf(
            trial._rchola, trial._rcholbara, ghalfa, ghalfaTcoul, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus
        )

        exx = 2.0 * kpt_symmchol_exx_kernel(trial._rchola, trial._rcholbara, ghalfa, ghalfaTx, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus) 

        e2b = ecoul + exx

        energy = xp.zeros((nwalkers, 3), dtype=numpy.complex128)
        energy[:, 0] = e1b + e2b
        energy[:, 1] = e1b
        energy[:, 2] = e2b
    else:
        ghalfa = walkers.Ghalfa.reshape(nwalkers, nk, nalpha, nk, nbasis)
        ghalfb = walkers.Ghalfb.reshape(nwalkers, nk, nbeta, nk, nbasis)
        ghalfaT = walkers.Ghalfa.transpose(0, 2, 1).reshape(nwalkers, nk, nbasis, nk, nalpha)
        ghalfbT = walkers.Ghalfb.transpose(0, 2, 1).reshape(nwalkers, nk, nbasis, nk, nbeta)

        diagGhalfa = numpy.zeros((nwalkers, nk, nalpha, nbasis), dtype=numpy.complex128)
        diagGhalfb = numpy.zeros((nwalkers, nk, nbeta, nbasis), dtype=numpy.complex128)
        for ik in range(nk):
            diagGhalfa[:, ik, :, :] = ghalfa[:, ik, :, ik, :]
            diagGhalfb[:, ik, :, :] = ghalfb[:, ik, :, ik, :]
        e1b = numpy.einsum('wkip, kip -> w', diagGhalfa, trial._rH1a) # Ghalfa.dot(trial._rH1a.ravel())
        e1b += numpy.einsum('wkip, kip -> w', diagGhalfb, trial._rH1b)
        e1b /= nk
        e1b += hamiltonian.ecore

        ghalfa = ghalfa.transpose(1, 3, 0, 2, 4).copy() # nk, nk, nw, nalpha, nbasis
        ghalfb = ghalfb.transpose(1, 3, 0, 2, 4).copy() # nk, nk, nw, nbeta, nbasis
        ghalfaTcoul = ghalfaT.transpose(1, 3, 0, 2, 4).copy() # nk, nk, nw, nbasis, nalpha
        ghalfbTcoul = ghalfbT.transpose(1, 3, 0, 2, 4).copy() # nk, nk, nw, nbasis, nbeta
        ghalfaTx = ghalfaT.transpose(1, 3, 2, 4, 0).copy() # nk, nk, nbasis, nalpha, nw
        ghalfbTx = ghalfbT.transpose(1, 3, 2, 4, 0).copy() # nk, nk, nbasis, nbeta, nw

        ecoul = kpt_symmchol_ecoul_kernel_uhf(
            trial._rchola, trial._rcholb, trial._rcholbara, trial._rcholbarb, ghalfa, ghalfb, ghalfaTcoul, ghalfbTcoul, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus
        )

        exxa = kpt_symmchol_exx_kernel(trial._rchola, trial._rcholbara, ghalfa, ghalfaTx, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus) 
        exxb = kpt_symmchol_exx_kernel(trial._rcholb, trial._rcholbarb, ghalfb, ghalfbTx, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus)


        e2b = ecoul + exxa + exxb

        energy = xp.zeros((nwalkers, 3), dtype=numpy.complex128)
        energy[:, 0] = e1b + e2b
        energy[:, 1] = e1b
        energy[:, 2] = e2b

    return energy


@plum.dispatch
def local_energy_kpt_single_det_uhf(
    system: Generic,
    hamiltonian: KptISDF,
    walkers: UHFWalkers,
    trial: KptSingleDet,
):
    """Compute local energy for walker batch (all walkers at once).

    Single determinant RHF case.

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

    if config.get_option("use_gpu"):
        return local_energy_kpt_single_det_uhf_isdf_gpu(system, hamiltonian, walkers, trial)
    else:
        raise NotImplementedError("CPU ISDF Coulomb kernel for UHF not implemented yet.")
    

def local_energy_kpt_single_det_uhf_isdf_gpu(system, hamiltonian, walkers, trial):
    """Compute local energy for walker batch (all walkers at once).

    Single determinant RHF case.

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

    kdiag = xp.arange(nk)
    if walkers.rhf:
        ghalfa = walkers.Ghalfa.reshape(nwalkers, nk, nalpha, nk, nbasis)
        # advanced indexing puts the k axis first: (nk, nw, nocc, nbsf)
        diagGhalfa = ghalfa[:, kdiag, :, kdiag, :].transpose(1, 0, 2, 3)
        diagGhalfa = diagGhalfa.reshape(nwalkers, nk * nalpha * nbasis)
        e1b = 2. * diagGhalfa.dot(trial._rH1a.ravel())
        e1b /= nk
        e1b += hamiltonian.ecore

        ecoul = kpt_isdf_ecoul_rhf_kernel_gpu(hamiltonian.MPQ, trial._rcgtoa, hamiltonian.cgto, ghalfa, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus)

        exxa = 2.0 * kpt_isdf_exx_kernel_gpu(hamiltonian.MPQ, trial._rcgtoa, hamiltonian.cgto, ghalfa, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus)

        e2b = ecoul + exxa
    else:
        ghalfa = walkers.Ghalfa.reshape(nwalkers, nk, nalpha, nk, nbasis)
        ghalfb = walkers.Ghalfb.reshape(nwalkers, nk, nbeta, nk, nbasis)

        diagGhalfa = ghalfa[:, kdiag, :, kdiag, :].transpose(1, 0, 2, 3)
        diagGhalfb = ghalfb[:, kdiag, :, kdiag, :].transpose(1, 0, 2, 3)
        diagGhalfa = diagGhalfa.reshape(nwalkers, nk * nalpha * nbasis)
        diagGhalfb = diagGhalfb.reshape(nwalkers, nk * nbeta * nbasis)
        e1b = diagGhalfa.dot(trial._rH1a.ravel())
        e1b += diagGhalfb.dot(trial._rH1b.ravel())
        e1b /= nk
        e1b += hamiltonian.ecore

        ecoul = kpt_isdf_ecoul_kernel_gpu(hamiltonian.MPQ, trial._rcgtoa, trial._rcgtob, hamiltonian.cgto, ghalfa, ghalfb, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus)

        spin_degenerate = getattr(trial, "_rcgto_spin_degenerate", None)
        if spin_degenerate is None:
            spin_degenerate = trial._rcgtoa is trial._rcgtob or (
                trial._rcgtoa.shape == trial._rcgtob.shape
                and bool(xp.all(trial._rcgtoa == trial._rcgtob))
            )
            trial._rcgto_spin_degenerate = spin_degenerate
        if spin_degenerate:
            # same half-rotated orbitals for both spins: one exx call on the
            # walker-concatenated Green's function instead of two
            ghalf_both = xp.concatenate((ghalfa, ghalfb), axis=0)
            exx_both = kpt_isdf_exx_kernel_gpu(hamiltonian.MPQ, trial._rcgtoa, hamiltonian.cgto, ghalf_both, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus)
            exxa = exx_both[:nwalkers]
            exxb = exx_both[nwalkers:]
        else:
            exxa = kpt_isdf_exx_kernel_gpu(hamiltonian.MPQ, trial._rcgtoa, hamiltonian.cgto, ghalfa, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus)
            exxb = kpt_isdf_exx_kernel_gpu(hamiltonian.MPQ, trial._rcgtob, hamiltonian.cgto, ghalfb, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus)

        e2b = ecoul + exxa + exxb

    energy = xp.zeros((nwalkers, 3), dtype=numpy.complex128)
    energy[:, 0] = e1b + e2b
    energy[:, 1] = e1b
    energy[:, 2] = e2b

    return energy