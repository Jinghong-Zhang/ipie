import numpy
import pytest

from ipie.estimators.local_energy_kpt_sd import (
    kpt_isdf_ecoul_kernel_gpu,
    kpt_isdf_ecoul_rhf_kernel_gpu,
    kpt_isdf_exx_kernel_gpu,
    local_energy_kpt_single_det_uhf_isdf_gpu,
)


def complex_rand(rng, shape):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def kpoint_map(nk):
    """Synthetic symmetric k+q index map: kpq_mat[iq, ik] = (ik + iq) % nk."""
    kpq_mat = numpy.zeros((nk, nk), dtype=numpy.int64)
    for iq in range(nk):
        for ik in range(nk):
            kpq_mat[iq, ik] = (ik + iq) % nk
    return kpq_mat


def build_inputs(rng, nk=4, nw=3, nocca=2, noccb=2, nbsf=5, nisdf=8):
    kpq_mat = kpoint_map(nk)
    Sset = numpy.array([0, 2])
    Qplus = numpy.array([1, 3])
    nq = len(Sset) + len(Qplus)
    MPQ = complex_rand(rng, (nq, nisdf, nisdf))
    cgto = complex_rand(rng, (nk, nisdf, nbsf))
    rcgtoa = complex_rand(rng, (nk, nisdf, nocca))
    rcgtob = complex_rand(rng, (nk, nisdf, noccb))
    Ga = complex_rand(rng, (nw, nk, nocca, nk, nbsf))
    Gb = complex_rand(rng, (nw, nk, noccb, nk, nbsf))
    return MPQ, rcgtoa, rcgtob, cgto, Ga, Gb, kpq_mat, Sset, Qplus


def ref_ecoul_v1v2(MPQ_iq, rcgto_lis, cgto, G_lis, kpq_mat, iq_real):
    kpq = kpq_mat[iq_real]
    v1 = 0.0
    v2 = 0.0
    for rcgto, G in zip(rcgto_lis, G_lis):
        g1 = G[:, numpy.arange(G.shape[1]), :, kpq, :].transpose(1, 0, 2, 3)  # w, k, i, r(k+q)
        v1 = v1 + numpy.einsum("kPi,kPr,wkir->wP", rcgto.conj(), cgto[kpq], g1, optimize=True)
        g2 = G[:, kpq, :, numpy.arange(G.shape[1]), :].transpose(1, 0, 2, 3)  # w, k(k+q), i, r
        v2 = v2 + numpy.einsum("kPi,kPr,wkir->wP", rcgto[kpq].conj(), cgto, g2, optimize=True)
    return numpy.sum((v1 @ MPQ_iq) * v2, axis=1)


def ref_ecoul(MPQ, rcgto_lis, cgto, G_lis, kpq_mat, Sset, Qplus):
    nw = G_lis[0].shape[0]
    ecoul = numpy.zeros(nw, dtype=numpy.complex128)
    for iq, iq_real in enumerate(Sset):
        ecoul += ref_ecoul_v1v2(MPQ[iq], rcgto_lis, cgto, G_lis, kpq_mat, iq_real)
    for iq, iq_real in enumerate(Qplus):
        ecoul += 2.0 * ref_ecoul_v1v2(
            MPQ[len(Sset) + iq], rcgto_lis, cgto, G_lis, kpq_mat, iq_real
        )
    return ecoul


def ref_exx(MPQ, rcgto, cgto, G, kpq_mat, Sset, Qplus):
    nw, nk = G.shape[0], G.shape[1]
    exx = numpy.zeros(nw, dtype=numpy.complex128)
    for iq, (iq_real, weight) in enumerate(
        [(q, 1.0) for q in Sset] + [(q, 2.0) for q in Qplus]
    ):
        kpq = kpq_mat[iq_real]
        G_kpq = G.take(kpq, axis=1).take(kpq, axis=3)  # w, K, j, k, p
        exx -= weight * numpy.einsum(
            "kPi,kPp,PQ,KQj,KQq,wkiKq,wKjkp->w",
            rcgto.conj(),
            cgto[kpq],
            MPQ[iq],
            rcgto[kpq].conj(),
            cgto,
            G,
            G_kpq,
            optimize=True,
        )
    return 0.5 * exx / nk


@pytest.mark.unit
def test_kpt_isdf_ecoul_uhf_parity():
    rng = numpy.random.default_rng(7)
    MPQ, rcgtoa, rcgtob, cgto, Ga, Gb, kpq_mat, Sset, Qplus = build_inputs(rng)
    nk = cgto.shape[0]
    ref = 0.5 * ref_ecoul(MPQ, [rcgtoa, rcgtob], cgto, [Ga, Gb], kpq_mat, Sset, Qplus) / nk
    out = kpt_isdf_ecoul_kernel_gpu(MPQ, rcgtoa, rcgtob, cgto, Ga, Gb, kpq_mat, Sset, Qplus)
    numpy.testing.assert_allclose(out, ref, atol=1e-10)
    # tiny budget forces both the q-chunking and the kernel-internal chunking
    out_chunked = kpt_isdf_ecoul_kernel_gpu(
        MPQ, rcgtoa, rcgtob, cgto, Ga, Gb, kpq_mat, Sset, Qplus, max_mem_gb=1e-9
    )
    numpy.testing.assert_allclose(out_chunked, ref, atol=1e-10)


@pytest.mark.unit
def test_kpt_isdf_ecoul_uhf_unequal_occupancy():
    rng = numpy.random.default_rng(11)
    MPQ, rcgtoa, rcgtob, cgto, Ga, Gb, kpq_mat, Sset, Qplus = build_inputs(
        rng, nocca=3, noccb=2
    )
    nk = cgto.shape[0]
    ref = 0.5 * ref_ecoul(MPQ, [rcgtoa, rcgtob], cgto, [Ga, Gb], kpq_mat, Sset, Qplus) / nk
    out = kpt_isdf_ecoul_kernel_gpu(MPQ, rcgtoa, rcgtob, cgto, Ga, Gb, kpq_mat, Sset, Qplus)
    numpy.testing.assert_allclose(out, ref, atol=1e-10)


@pytest.mark.unit
def test_kpt_isdf_ecoul_rhf_parity():
    rng = numpy.random.default_rng(13)
    MPQ, rcgtoa, _, cgto, Ga, _, kpq_mat, Sset, Qplus = build_inputs(rng)
    nk = cgto.shape[0]
    ref = 2.0 * ref_ecoul(MPQ, [rcgtoa], cgto, [Ga], kpq_mat, Sset, Qplus) / nk
    out = kpt_isdf_ecoul_rhf_kernel_gpu(MPQ, rcgtoa, cgto, Ga, kpq_mat, Sset, Qplus)
    numpy.testing.assert_allclose(out, ref, atol=1e-10)
    out_chunked = kpt_isdf_ecoul_rhf_kernel_gpu(
        MPQ, rcgtoa, cgto, Ga, kpq_mat, Sset, Qplus, max_mem_gb=1e-9
    )
    numpy.testing.assert_allclose(out_chunked, ref, atol=1e-10)


@pytest.mark.unit
@pytest.mark.parametrize("algo", ["largek", "lowk", None])
def test_kpt_isdf_exx_parity(algo):
    rng = numpy.random.default_rng(17)
    MPQ, rcgtoa, _, cgto, Ga, _, kpq_mat, Sset, Qplus = build_inputs(rng)
    ref = ref_exx(MPQ, rcgtoa, cgto, Ga, kpq_mat, Sset, Qplus)
    out = kpt_isdf_exx_kernel_gpu(MPQ, rcgtoa, cgto, Ga, kpq_mat, Sset, Qplus, algo=algo)
    numpy.testing.assert_allclose(out, ref, atol=1e-10)


@pytest.mark.unit
@pytest.mark.parametrize("algo", ["largek", "lowk"])
def test_kpt_isdf_exx_forced_chunking(algo):
    rng = numpy.random.default_rng(19)
    MPQ, rcgtoa, _, cgto, Ga, _, kpq_mat, Sset, Qplus = build_inputs(rng)
    ref = ref_exx(MPQ, rcgtoa, cgto, Ga, kpq_mat, Sset, Qplus)
    # tiny budget forces single-element ISDF/walker chunks and disables the
    # low-k stage-1 cache
    out = kpt_isdf_exx_kernel_gpu(
        MPQ, rcgtoa, cgto, Ga, kpq_mat, Sset, Qplus, algo=algo, max_mem_gb=1e-9
    )
    numpy.testing.assert_allclose(out, ref, atol=1e-10)


@pytest.mark.unit
def test_kpt_isdf_exx_walker_concat():
    rng = numpy.random.default_rng(23)
    MPQ, rcgtoa, _, cgto, Ga, Gb, kpq_mat, Sset, Qplus = build_inputs(rng)
    nw = Ga.shape[0]
    both = kpt_isdf_exx_kernel_gpu(
        MPQ, rcgtoa, cgto, numpy.concatenate((Ga, Gb), axis=0), kpq_mat, Sset, Qplus
    )
    sep_a = kpt_isdf_exx_kernel_gpu(MPQ, rcgtoa, cgto, Ga, kpq_mat, Sset, Qplus)
    sep_b = kpt_isdf_exx_kernel_gpu(MPQ, rcgtoa, cgto, Gb, kpq_mat, Sset, Qplus)
    numpy.testing.assert_allclose(both[:nw], sep_a, atol=1e-10)
    numpy.testing.assert_allclose(both[nw:], sep_b, atol=1e-10)


class _Fake:
    pass


@pytest.mark.unit
@pytest.mark.parametrize("spin_degenerate", [True, False])
def test_local_energy_isdf_driver(spin_degenerate):
    rng = numpy.random.default_rng(29)
    nk, nw, nocc, nbsf, nisdf = 4, 3, 2, 5, 8
    MPQ, rcgtoa, rcgtob, cgto, Ga, Gb, kpq_mat, Sset, Qplus = build_inputs(
        rng, nk=nk, nw=nw, nocca=nocc, noccb=nocc, nbsf=nbsf, nisdf=nisdf
    )
    if spin_degenerate:
        rcgtob = rcgtoa.copy()

    hamiltonian = _Fake()
    hamiltonian.nk = nk
    hamiltonian.nbasis = nbsf
    hamiltonian.ecore = 0.17
    hamiltonian.MPQ = MPQ
    hamiltonian.cgto = cgto
    hamiltonian.ikpq_mat = kpq_mat
    hamiltonian.Sset = Sset
    hamiltonian.Qplus = Qplus

    trial = _Fake()
    trial.nalpha = nocc
    trial.nbeta = nocc
    trial._rcgtoa = rcgtoa
    trial._rcgtob = rcgtob
    trial._rH1a = complex_rand(rng, (nk, nocc, nbsf))
    trial._rH1b = complex_rand(rng, (nk, nocc, nbsf))

    walkers = _Fake()
    walkers.rhf = False
    walkers.Ghalfa = Ga.copy()
    walkers.Ghalfb = Gb.copy()

    energy = local_energy_kpt_single_det_uhf_isdf_gpu(None, hamiltonian, walkers, trial)

    diaga = numpy.stack([Ga[:, ik, :, ik, :] for ik in range(nk)], axis=1)
    diagb = numpy.stack([Gb[:, ik, :, ik, :] for ik in range(nk)], axis=1)
    e1b_ref = diaga.reshape(nw, -1).dot(trial._rH1a.ravel())
    e1b_ref += diagb.reshape(nw, -1).dot(trial._rH1b.ravel())
    e1b_ref = e1b_ref / nk + hamiltonian.ecore

    nk_ = cgto.shape[0]
    ecoul_ref = (
        0.5 * ref_ecoul(MPQ, [rcgtoa, rcgtob], cgto, [Ga, Gb], kpq_mat, Sset, Qplus) / nk_
    )
    exx_ref = ref_exx(MPQ, rcgtoa, cgto, Ga, kpq_mat, Sset, Qplus)
    exx_ref = exx_ref + ref_exx(MPQ, rcgtob, cgto, Gb, kpq_mat, Sset, Qplus)
    e2b_ref = ecoul_ref + exx_ref

    numpy.testing.assert_allclose(energy[:, 1], e1b_ref, atol=1e-10)
    numpy.testing.assert_allclose(energy[:, 2], e2b_ref, atol=1e-10)
    numpy.testing.assert_allclose(energy[:, 0], e1b_ref + e2b_ref, atol=1e-10)
    assert trial._rcgto_spin_degenerate is spin_degenerate


if __name__ == "__main__":
    test_kpt_isdf_ecoul_uhf_parity()
    test_kpt_isdf_ecoul_rhf_parity()
    for algo in ("largek", "lowk", None):
        test_kpt_isdf_exx_parity(algo)
    for algo in ("largek", "lowk"):
        test_kpt_isdf_exx_forced_chunking(algo)
    test_kpt_isdf_exx_walker_concat()
    for sd in (True, False):
        test_local_energy_isdf_driver(sd)
