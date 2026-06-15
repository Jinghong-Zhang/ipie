import numpy
import pytest

from ipie.config import config
from ipie.estimators.local_energy_kpt_sd import (
    kpt_symmchol_ecoul_kernel_uhf,
    kpt_symmchol_exx_kernel,
    local_energy_kpt_single_det_uhf,
)
from ipie.hamiltonians.kpt_hamiltonian import KptComplexChol, KptComplexCholSymm
from ipie.hamiltonians.kpt_isdf_hamiltonian import KptISDF
from ipie.systems.generic import Generic
from ipie.trial_wavefunction.single_det_kpt import KptSingleDet
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import to_host
from ipie.utils.mpi import MPIHandler
from ipie.utils.testing import (
    _blockdiag_k_orbitals,
    expand_chol_symm_to_full,
    gen_random_test_input_kpt,
    gen_random_test_input_kpt_isdf,
    shaped_normal,
)
from ipie.walkers.uhf_walkers import UHFWalkers


def _build_kpt_walkers(trial, nwalkers):
    init_walker = numpy.concatenate(
        [_blockdiag_k_orbitals(trial.psi0a), _blockdiag_k_orbitals(trial.psi0b)], axis=1
    )
    walkers = UHFWalkers(
        init_walker,
        trial.nk * trial.nalpha,
        trial.nk * trial.nbeta,
        trial.nk * trial.nbasis,
        nwalkers,
        MPIHandler(),
    )
    return walkers


def _local_energy_kptcholsymm_uhf_cpu(system, hamiltonian, walkers, trial):
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
    e1b = numpy.einsum("wkip, kip -> w", diagGhalfa, trial._rH1a)
    e1b += numpy.einsum("wkip, kip -> w", diagGhalfb, trial._rH1b)
    e1b /= nk
    e1b += hamiltonian.ecore

    ghalfa = ghalfa.transpose(1, 3, 0, 2, 4).copy()
    ghalfb = ghalfb.transpose(1, 3, 0, 2, 4).copy()
    ghalfaTcoul = ghalfaT.transpose(1, 3, 0, 2, 4).copy()
    ghalfbTcoul = ghalfbT.transpose(1, 3, 0, 2, 4).copy()
    ghalfaTx = ghalfaT.transpose(1, 3, 2, 4, 0).copy()
    ghalfbTx = ghalfbT.transpose(1, 3, 2, 4, 0).copy()

    ecoul = kpt_symmchol_ecoul_kernel_uhf(
        trial._rchola,
        trial._rcholb,
        trial._rcholbara,
        trial._rcholbarb,
        ghalfa,
        ghalfb,
        ghalfaTcoul,
        ghalfbTcoul,
        hamiltonian.ikpq_mat,
        hamiltonian.Sset,
        hamiltonian.Qplus,
    )
    exxa = kpt_symmchol_exx_kernel(
        trial._rchola,
        trial._rcholbara,
        ghalfa,
        ghalfaTx,
        hamiltonian.ikpq_mat,
        hamiltonian.Sset,
        hamiltonian.Qplus,
    )
    exxb = kpt_symmchol_exx_kernel(
        trial._rcholb,
        trial._rcholbarb,
        ghalfb,
        ghalfbTx,
        hamiltonian.ikpq_mat,
        hamiltonian.Sset,
        hamiltonian.Qplus,
    )
    e2b = ecoul + exxa + exxb

    energy = numpy.zeros((nwalkers, 3), dtype=numpy.complex128)
    energy[:, 0] = e1b + e2b
    energy[:, 1] = e1b
    energy[:, 2] = e2b
    return energy


@pytest.mark.unit
def test_local_energy_kptchol():
    kmesh = (2, 1, 1)
    nk = numpy.prod(kmesh)
    nbasis = 6
    naux = 3 * nbasis
    nalpha, nbeta = (2, 2)
    nwalkers = 3

    h1e, chol_packed, wfn, kpts = gen_random_test_input_kpt(
        kmesh, nbasis, (nalpha, nbeta), naux, seed=7
    )
    ham_symm = KptComplexCholSymm(
        h1e=numpy.array([h1e, h1e], dtype=numpy.complex128),
        chol=numpy.array(chol_packed, dtype=numpy.complex128),
        kpts=kpts,
    )
    chol_full = expand_chol_symm_to_full(
        chol_packed,
        kpts,
        ham_symm.Sset,
        ham_symm.Qplus,
        ham_symm.ikpq_mat,
        ham_symm.imq_vec,
    )
    ham = KptComplexChol(
        h1e=numpy.array([h1e, h1e], dtype=numpy.complex128),
        chol=numpy.array(chol_full, dtype=numpy.complex128),
        kpts=kpts,
    )

    system = Generic(nelec=(nalpha, nbeta))
    trial = KptSingleDet(wfn, nk, (nalpha, nbeta), nbasis)
    trial.half_rotate(ham)
    walkers = _build_kpt_walkers(trial, nwalkers)
    walkers.Ghalfa = shaped_normal((nwalkers, nk * nalpha, nk * nbasis), cmplx=True, seed=11)
    walkers.Ghalfb = shaped_normal((nwalkers, nk * nbeta, nk * nbasis), cmplx=True, seed=13)

    energy = local_energy_kpt_single_det_uhf(system, ham, walkers, trial)
    assert energy.shape == (nwalkers, 3)
    assert numpy.all(numpy.isfinite(energy.real))


@pytest.mark.unit
def test_local_energy_kptcholsymm():
    kmesh = (2, 1, 1)
    nk = numpy.prod(kmesh)
    nbasis = 6
    naux = 3 * nbasis
    nalpha, nbeta = (2, 2)
    nwalkers = 3

    h1e, chol_packed, wfn, kpts = gen_random_test_input_kpt(
        kmesh, nbasis, (nalpha, nbeta), naux, seed=17
    )
    ham = KptComplexCholSymm(
        h1e=numpy.array([h1e, h1e], dtype=numpy.complex128),
        chol=numpy.array(chol_packed, dtype=numpy.complex128),
        kpts=kpts,
    )

    system = Generic(nelec=(nalpha, nbeta))
    trial = KptSingleDet(wfn, nk, (nalpha, nbeta), nbasis)
    trial.half_rotate(ham)
    walkers = _build_kpt_walkers(trial, nwalkers)
    walkers.Ghalfa = shaped_normal((nwalkers, nk * nalpha, nk * nbasis), cmplx=True, seed=19)
    walkers.Ghalfb = shaped_normal((nwalkers, nk * nbeta, nk * nbasis), cmplx=True, seed=23)

    energy = local_energy_kpt_single_det_uhf(system, ham, walkers, trial)
    assert energy.shape == (nwalkers, 3)
    assert numpy.all(numpy.isfinite(energy.real))


@pytest.mark.gpu
def test_local_energy_kptisdf_gpu():
    if not config.get_option("use_gpu"):
        pytest.skip("Requires GPU backend. Set IPIE_USE_GPU=1 to run this test.")

    kmesh = (2, 1, 1)
    nk = numpy.prod(kmesh)
    nbasis = 6
    naux = 3 * nbasis
    nalpha, nbeta = (2, 2)
    nwalkers = 3

    h1e, MPQ_in, cgto, wfn, kpts = gen_random_test_input_kpt_isdf(
        kmesh, nbasis, (nalpha, nbeta), naux, seed=31
    )
    nq = MPQ_in.shape[0]
    nisdf = cgto.shape[1]
    cholM = shaped_normal((nq, nisdf, naux), cmplx=True, seed=37)
    MPQ = numpy.einsum("qPg,qRg->qPR", cholM, cholM.conj(), optimize=True)

    ham = KptISDF(
        h1e=numpy.array([h1e, h1e], dtype=numpy.complex128),
        MPQ=numpy.array(MPQ, dtype=numpy.complex128),
        cholM=numpy.array(cholM, dtype=numpy.complex128),
        cgto=numpy.array(cgto, dtype=numpy.complex128),
        kpts=kpts,
        h1e_mod=numpy.zeros((2, nk, nbasis, nbasis), dtype=numpy.complex128),
    )

    system = Generic(nelec=(nalpha, nbeta))
    trial = KptSingleDet(wfn, nk, (nalpha, nbeta), nbasis)
    trial.half_rotate(ham)
    walkers = _build_kpt_walkers(trial, nwalkers)
    walkers.Ghalfa = shaped_normal((nwalkers, nk * nalpha, nk * nbasis), cmplx=True, seed=41)
    walkers.Ghalfb = shaped_normal((nwalkers, nk * nbeta, nk * nbasis), cmplx=True, seed=43)

    ham.cast_to_cupy()
    trial.cast_to_cupy()
    walkers.cast_to_cupy()

    energy = local_energy_kpt_single_det_uhf(system, ham, walkers, trial)
    energy_h = to_host(energy)
    assert energy_h.shape == (nwalkers, 3)
    assert numpy.all(numpy.isfinite(energy_h.real))


@pytest.mark.unit
def test_kptisdf_exchange_gpu_selector_regimes():
    from ipie.estimators.local_energy_kpt_sd_isdf import _select_kpt_isdf_exx_kernel_gpu

    assert _select_kpt_isdf_exx_kernel_gpu(1, 500, 4)[0] == "dense_cupy"
    assert _select_kpt_isdf_exx_kernel_gpu(1, 1000, 4)[0] == "lowk_cupy"
    assert _select_kpt_isdf_exx_kernel_gpu(1, 1000, 8)[0] == "original_cuquantum"
    assert _select_kpt_isdf_exx_kernel_gpu(1, 1000, 16)[0] == "lowk_cupy"
    assert _select_kpt_isdf_exx_kernel_gpu(1, 2000, 4)[0] == "lowk_cupy"
    assert _select_kpt_isdf_exx_kernel_gpu(1, 1000, 16, nwalker=4)[0] == "lowk_cupy"
    assert _select_kpt_isdf_exx_kernel_gpu(1, 1000, 8, nwalker=4)[0] == "original_cuquantum"
    assert _select_kpt_isdf_exx_kernel_gpu(1, 2000, 4, nwalker=4)[0] == "dense_cupy"
    assert _select_kpt_isdf_exx_kernel_gpu(1, 500, 4, nwalker=64)[0] == "dense_cupy"
    assert _select_kpt_isdf_exx_kernel_gpu(1, 500, 4, nwalker=512)[0] == "dense_cupy"
    assert _select_kpt_isdf_exx_kernel_gpu(1, 500, 4, nwalker=1024)[0] == "dense_cupy"
    assert _select_kpt_isdf_exx_kernel_gpu(1, 1000, 4, nwalker=64)[0] == "dense_cupy"
    assert _select_kpt_isdf_exx_kernel_gpu(1, 1000, 16, nwalker=64)[0] == "dense_cupy"
    assert _select_kpt_isdf_exx_kernel_gpu(8, 100, 25)[0] == "dense_cupy"
    assert _select_kpt_isdf_exx_kernel_gpu(8, 200, 4)[0] == "dense_cupy"
    assert _select_kpt_isdf_exx_kernel_gpu(8, 200, 16)[0] == "lowk_cupy"
    assert _select_kpt_isdf_exx_kernel_gpu(8, 200, 16, nwalker=4)[0] == "dense_cupy"
    assert _select_kpt_isdf_exx_kernel_gpu(8, 200, 16, nwalker=64)[0] == "dense_cupy"
    assert _select_kpt_isdf_exx_kernel_gpu(8, 200, 4, nwalker=64)[0] == "original_cuquantum"
    assert _select_kpt_isdf_exx_kernel_gpu(8, 100, 25, nwalker=256)[0] == "dense_cupy"
    assert _select_kpt_isdf_exx_kernel_gpu(32, 100, 25)[0] == "dense_cupy"
    assert _select_kpt_isdf_exx_kernel_gpu(32, 100, 25, nwalker=4)[0] == "dense_cupy"
    assert _select_kpt_isdf_exx_kernel_gpu(27, 200, 4)[0] == "dense_cupy"
    assert _select_kpt_isdf_exx_kernel_gpu(27, 100, 25, nwalker=64)[0] == "original_cuquantum"
    assert _select_kpt_isdf_exx_kernel_gpu(64, 100, 4)[0] == "dense_cupy"
    assert _select_kpt_isdf_exx_kernel_gpu(64, 100, 8)[0] == "dense_cupy"
    assert _select_kpt_isdf_exx_kernel_gpu(64, 100, 16)[0] == "original_cuquantum"
    assert _select_kpt_isdf_exx_kernel_gpu(64, 100, 25)[0] == "original_cuquantum"
    assert _select_kpt_isdf_exx_kernel_gpu(64, 100, 64)[0] == "original_cuquantum"
    assert _select_kpt_isdf_exx_kernel_gpu(64, 100, 4, nwalker=4)[0] == "original_cuquantum"
    assert _select_kpt_isdf_exx_kernel_gpu(64, 200, 32, nwalker=1)[0] == "dense_cupy"
    assert (
        _select_kpt_isdf_exx_kernel_gpu(64, 200, 4, nwalker=2)[0] == "original_cuquantum"
    )
    assert (
        _select_kpt_isdf_exx_kernel_gpu(64, 200, 16, nwalker=2)[0] == "original_cuquantum"
    )
    assert _select_kpt_isdf_exx_kernel_gpu(125, 200, 4, nwalker=1)[0] == "dense_cupy"
    assert _select_kpt_isdf_exx_kernel_gpu(125, 100, 8, nwalker=1)[0] == "dense_cupy"
    assert _select_kpt_isdf_exx_kernel_gpu(125, 100, 25, nwalker=1)[0] == "original_cuquantum"
    assert _select_kpt_isdf_exx_kernel_gpu(125, 100, 4, nwalker=2)[0] == "original_cuquantum"


@pytest.mark.unit
def test_kptisdf_dense_cupy_chunk_estimate_bounds_large_k_memory():
    from ipie.estimators.local_energy_kpt_sd_isdf import (
        _choose_exx_dense_optimized_chunks,
        _estimate_exx_dense_optimized_bytes,
    )

    itemsize = numpy.dtype(numpy.complex128).itemsize
    max_mem = 32.0
    q_chunk, p_chunk, k_chunk = _choose_exx_dense_optimized_chunks(
        nwalker=2,
        nk=64,
        nocc=4,
        nbsf=200,
        nisdf=3600,
        itemsize=itemsize,
        max_mem=max_mem,
    )
    estimate = _estimate_exx_dense_optimized_bytes(
        nwalker=2,
        nk=64,
        nocc=4,
        nbsf=200,
        nisdf=3600,
        q_chunk=q_chunk,
        p_chunk=p_chunk,
        k_chunk=k_chunk,
        itemsize=itemsize,
    )

    assert 1 <= k_chunk < 64
    assert 1 <= q_chunk <= 3600
    assert 1 <= p_chunk <= 3600
    assert estimate <= max_mem * 1024**3


@pytest.mark.gpu
def test_kptisdf_exchange_lowk_cupy_matches_original_cuquantum_gpu():
    if not config.get_option("use_gpu"):
        pytest.skip("Requires GPU backend. Set IPIE_USE_GPU=1 to run this test.")

    from ipie.estimators.local_energy_kpt_sd_isdf import (
        kpt_isdf_exx_kernel_gpu_cutn_path_cupy,
        kpt_isdf_exx_kernel_gpu_dense_cupy,
        kpt_isdf_exx_kernel_gpu_lowk_cupy,
        kpt_isdf_exx_kernel_gpu_original_cuquantum,
    )

    kmesh = (2, 2, 1)
    nk = numpy.prod(kmesh)
    nbasis = 8
    naux = 3 * nbasis
    nalpha, nbeta = (2, 2)
    nwalkers = 2

    h1e, MPQ_in, cgto, wfn, kpts = gen_random_test_input_kpt_isdf(
        kmesh, nbasis, (nalpha, nbeta), naux, seed=47
    )
    nq = MPQ_in.shape[0]
    nisdf = cgto.shape[1]
    cholM = shaped_normal((nq, nisdf, naux), cmplx=True, seed=49)
    MPQ = numpy.einsum("qPg,qRg->qPR", cholM, cholM.conj(), optimize=True)

    ham = KptISDF(
        h1e=numpy.array([h1e, h1e], dtype=numpy.complex128),
        MPQ=numpy.array(MPQ, dtype=numpy.complex128),
        cholM=numpy.array(cholM, dtype=numpy.complex128),
        cgto=numpy.array(cgto, dtype=numpy.complex128),
        kpts=kpts,
        h1e_mod=numpy.zeros((2, nk, nbasis, nbasis), dtype=numpy.complex128),
    )
    trial = KptSingleDet(wfn, nk, (nalpha, nbeta), nbasis)
    trial.half_rotate(ham)
    ghalfa = shaped_normal((nwalkers, nk, nalpha, nk, nbasis), cmplx=True, seed=59)

    ham.cast_to_cupy()
    trial.cast_to_cupy()
    ghalfa = xp.asarray(ghalfa)

    original = kpt_isdf_exx_kernel_gpu_original_cuquantum(
        ham.MPQ,
        trial._rcgtoa,
        ham.cgto,
        ghalfa,
        ham.ikpq_mat,
        ham.Sset,
        ham.Qplus,
    )
    lowk = kpt_isdf_exx_kernel_gpu_lowk_cupy(
        ham.MPQ,
        trial._rcgtoa,
        ham.cgto,
        ghalfa,
        ham.ikpq_mat,
        ham.Sset,
        ham.Qplus,
        max_mem=0.01,
    )
    dense = kpt_isdf_exx_kernel_gpu_dense_cupy(
        ham.MPQ,
        trial._rcgtoa,
        ham.cgto,
        ghalfa,
        ham.ikpq_mat,
        ham.Sset,
        ham.Qplus,
        max_mem=0.01,
    )
    cutn_path = kpt_isdf_exx_kernel_gpu_cutn_path_cupy(
        ham.MPQ,
        trial._rcgtoa,
        ham.cgto,
        ghalfa,
        ham.ikpq_mat,
        ham.Sset,
        ham.Qplus,
        max_mem=0.01,
    )

    numpy.testing.assert_allclose(to_host(lowk), to_host(original), atol=1e-8, rtol=1e-8)
    numpy.testing.assert_allclose(to_host(dense), to_host(original), atol=1e-8, rtol=1e-8)
    numpy.testing.assert_allclose(to_host(cutn_path), to_host(original), atol=1e-8, rtol=1e-8)


@pytest.mark.gpu
def test_kptisdf_vs_kptcholsymm_reconstructed_chol_gpu():
    if not config.get_option("use_gpu"):
        pytest.skip("Requires GPU backend. Set IPIE_USE_GPU=1 to run this test.")

    kmesh = (2, 1, 1)
    nk = numpy.prod(kmesh)
    nbasis = 6
    naux = 3 * nbasis
    nalpha, nbeta = (2, 2)
    nwalkers = 2

    h1e, MPQ_in, cgto, wfn, kpts = gen_random_test_input_kpt_isdf(
        kmesh, nbasis, (nalpha, nbeta), naux, seed=51
    )
    nq = MPQ_in.shape[0]
    nisdf = cgto.shape[1]
    cholM = shaped_normal((nq, nisdf, naux), cmplx=True, seed=53)
    MPQ = numpy.einsum("qPg,qRg->qPR", cholM, cholM.conj(), optimize=True)

    ham_isdf = KptISDF(
        h1e=numpy.array([h1e, h1e], dtype=numpy.complex128),
        MPQ=numpy.array(MPQ, dtype=numpy.complex128),
        cholM=numpy.array(cholM, dtype=numpy.complex128),
        cgto=numpy.array(cgto, dtype=numpy.complex128),
        kpts=kpts,
        h1e_mod=numpy.zeros((2, nk, nbasis, nbasis), dtype=numpy.complex128),
    )

    chol_symm = numpy.zeros((naux, nk, nbasis, ham_isdf.unique_nk, nbasis), dtype=numpy.complex128)
    unique_qs = numpy.concatenate((ham_isdf.Sset, ham_isdf.Qplus))
    for iq, iq_real in enumerate(unique_qs):
        for ik in range(nk):
            ikpq = ham_isdf.ikpq_mat[iq_real, ik]
            chol_symm[:, ik, :, iq, :] = numpy.einsum(
                "Pp,Pr,Pg->gpr",
                cgto[ik].conj(),
                cgto[ikpq],
                cholM[iq],
                optimize=True,
            )

    ham_symm = KptComplexCholSymm(
        h1e=numpy.array([h1e, h1e], dtype=numpy.complex128),
        chol=numpy.array(chol_symm, dtype=numpy.complex128),
        kpts=kpts,
    )

    system = Generic(nelec=(nalpha, nbeta))
    trial_symm = KptSingleDet(wfn, nk, (nalpha, nbeta), nbasis)
    trial_symm.half_rotate(ham_symm)
    walkers_symm = _build_kpt_walkers(trial_symm, nwalkers)

    trial_isdf = KptSingleDet(wfn, nk, (nalpha, nbeta), nbasis)
    trial_isdf.half_rotate(ham_isdf)
    walkers_isdf = _build_kpt_walkers(trial_isdf, nwalkers)

    ghalfa = shaped_normal((nwalkers, nk * nalpha, nk * nbasis), cmplx=True, seed=67)
    ghalfb = shaped_normal((nwalkers, nk * nbeta, nk * nbasis), cmplx=True, seed=71)
    walkers_symm.Ghalfa = ghalfa.copy()
    walkers_symm.Ghalfb = ghalfb.copy()
    walkers_isdf.Ghalfa = ghalfa.copy()
    walkers_isdf.Ghalfb = ghalfb.copy()

    energy_symm = _local_energy_kptcholsymm_uhf_cpu(system, ham_symm, walkers_symm, trial_symm)

    ham_isdf.cast_to_cupy()
    trial_isdf.cast_to_cupy()
    walkers_isdf.cast_to_cupy()
    energy_isdf = local_energy_kpt_single_det_uhf(system, ham_isdf, walkers_isdf, trial_isdf)

    numpy.testing.assert_allclose(to_host(energy_isdf), energy_symm, atol=1e-8)
