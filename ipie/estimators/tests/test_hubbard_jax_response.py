from types import SimpleNamespace
import os
import tempfile

import numpy
import pytest
import scipy.linalg

jax = pytest.importorskip("jax")

from ipie.estimators import hubbard_jax_ad
from ipie.estimators.hubbard_jax_ad import HubbardJAXADBlockRunner
from ipie.estimators.hubbard_jax_response import HubbardJAX1RDMResponseEstimator
from ipie.config import MPI
from ipie.hamiltonians.hubbard import Hubbard
from ipie.propagation.hirsch_base import construct_hirsch_auxiliaries
from ipie.propagation.hubbard_jax import (
    _mpi_allgather_ad,
    as_jax_array,
    block_until_ready,
    hashable_mpi_comm,
    hubbard_ad_block_raw,
    hubbard_ad_block_value,
    hubbard_response_objective,
    hubbard_uhf_trial_from_densities,
)
from ipie.qmc.afqmc import AFQMC
from ipie.trial_wavefunction.single_det import SingleDet
from ipie.utils.mpi import MPIHandler
from ipie.walkers.walkers_dispatch import UHFWalkersTrial


def _hubbard_inputs(nelec=(2, 1), nbasis=4, nwalkers=4, rhf=False):
    if rhf:
        assert nelec[0] == nelec[1]
    h1 = numpy.zeros((nbasis, nbasis), dtype=numpy.float64)
    for i in range(nbasis - 1):
        h1[i, i + 1] = h1[i + 1, i] = -0.7
    h1 += numpy.diag(numpy.linspace(-0.35, 0.45, nbasis))

    psi = numpy.zeros((nbasis, sum(nelec)), dtype=numpy.complex128)
    psi[:, : nelec[0]] = numpy.eye(nbasis, dtype=numpy.complex128)[:, : nelec[0]]
    if rhf:
        psi[:, nelec[0] :] = psi[:, : nelec[0]]
    else:
        psi[:, nelec[0] :] = numpy.eye(nbasis, dtype=numpy.complex128)[:, -nelec[1] :]

    ham = Hubbard(numpy.array([h1, h1]), U=2.0)
    trial = SingleDet(psi, nelec, nbasis)
    trial.build()
    trial.half_rotate(ham)
    return ham, trial, psi, nwalkers


def _hubbard_2x2_inputs(nwalkers=2):
    nbasis = 4
    nelec = (2, 1)
    h1 = numpy.zeros((nbasis, nbasis), dtype=numpy.float64)

    def site(x, y):
        return x + 2 * y

    for x in range(2):
        for y in range(2):
            if x + 1 < 2:
                i, j = site(x, y), site(x + 1, y)
                h1[i, j] = h1[j, i] = -0.7
            if y + 1 < 2:
                i, j = site(x, y), site(x, y + 1)
                h1[i, j] = h1[j, i] = -0.5
    h1 += numpy.diag([-0.31, -0.07, 0.13, 0.29])
    h1_up = h1.copy()
    h1_down = h1 + numpy.diag([0.04, -0.02, 0.01, -0.03])

    psi = numpy.zeros((nbasis, sum(nelec)), dtype=numpy.complex128)
    psi[:, : nelec[0]] = numpy.eye(nbasis, dtype=numpy.complex128)[:, : nelec[0]]
    psi[:, nelec[0] :] = numpy.eye(nbasis, dtype=numpy.complex128)[:, -nelec[1] :]

    ham = Hubbard(numpy.array([h1_up, h1_down]), U=2.0)
    trial = SingleDet(psi, nelec, nbasis)
    trial.build()
    trial.half_rotate(ham)
    return ham, trial, psi, nwalkers, nelec


def _new_walkers(trial, initial_walker, nelec, nbasis, nwalkers, rhf=False):
    walkers = UHFWalkersTrial(
        trial, initial_walker, nelec[0], nelec[1], nbasis, nwalkers, MPIHandler()
    )
    walkers.build(trial)
    walkers.rhf = rhf
    return walkers


def _walker_view(walkers, start, stop):
    return SimpleNamespace(
        nwalkers=stop - start,
        nup=walkers.nup,
        ndown=walkers.ndown,
        nbasis=walkers.nbasis,
        rhf=walkers.rhf,
        phia=walkers.phia[start:stop],
        phib=None if walkers.phib is None else walkers.phib[start:stop],
        weight=walkers.weight[start:stop],
        log_shift=walkers.log_shift[start:stop],
    )


def _gauge_fix_numpy(vecs):
    vecs = vecs.astype(numpy.complex128)
    max_abs = numpy.argmax(numpy.abs(vecs), axis=0)
    vals = vecs[max_abs, numpy.arange(vecs.shape[1])]
    phase = numpy.where(numpy.abs(vals) > 0.0, vals / numpy.abs(vals), 1.0 + 0.0j)
    return vecs / phase[numpy.newaxis, :]


def _objective_value(estimator, walkers, hamiltonian, trial, random_fields, coupling):
    _, _, aux_wfac, delta = construct_hirsch_auxiliaries(
        hamiltonian, estimator.timestep, spin_decomp=estimator.spin_decomp
    )
    phib = walkers.phib
    if phib is None:
        phib = numpy.zeros((walkers.nwalkers, walkers.nbasis, 0), dtype=numpy.complex128)
    values = hubbard_response_objective(
        as_jax_array(coupling, dtype="float64"),
        as_jax_array(walkers.phia, dtype="complex128"),
        as_jax_array(phib, dtype="complex128"),
        as_jax_array(walkers.weight, dtype="float64"),
        as_jax_array(walkers.log_shift, dtype="float64"),
        as_jax_array(trial.psi0a, dtype="complex128"),
        as_jax_array(trial.psi0b, dtype="complex128"),
        as_jax_array(hamiltonian.T, dtype="complex128"),
        as_jax_array(hamiltonian.U, dtype="complex128"),
        as_jax_array(hamiltonian.ecore, dtype="complex128"),
        as_jax_array(delta, dtype="complex128"),
        as_jax_array(aux_wfac, dtype="complex128"),
        as_jax_array(random_fields, dtype="float64"),
        estimator.timestep,
        walkers.nup,
        walkers.ndown,
        rhf=bool(walkers.rhf),
        trial_response=estimator.trial_response,
        uhf_n_scf=estimator.uhf_n_scf,
        uhf_mixing=estimator.uhf_mixing,
        uhf_ueff=estimator.uhf_ueff,
    )
    values = numpy.asarray(block_until_ready(values))
    return values[0] / values[1]


def _ratio_gradient(raw):
    return (
        raw["dENumer"] / raw["EDenom"][0]
        - (raw["ENumer"][0] / raw["EDenom"][0]) * raw["dEDenom"] / raw["EDenom"][0]
    )


def _comm_sum(value, comm):
    send = numpy.asarray(value)
    recv = numpy.zeros_like(send)
    comm.Allreduce(send, recv, op=MPI.SUM)
    return recv


def _reduce_raw(raw, comm):
    return {key: _comm_sum(value, comm) for key, value in raw.items()}


def _ad_block_raw_for_coupling(
    walkers,
    hamiltonian,
    trial,
    coupling,
    random_fields,
    pop_control_randoms,
    comm,
    hcomm,
):
    _, _, aux_wfac, delta = construct_hirsch_auxiliaries(hamiltonian, 0.01, spin_decomp=True)
    phib = walkers.phib
    if phib is None:
        phib = numpy.zeros((walkers.nwalkers, walkers.nbasis, 0), dtype=numpy.complex128)
    result = hubbard_ad_block_raw(
        as_jax_array(coupling, dtype="float64", force_host=True),
        as_jax_array(walkers.phia, dtype="complex128", force_host=True),
        as_jax_array(phib, dtype="complex128", force_host=True),
        as_jax_array(walkers.weight, dtype="float64", force_host=True),
        as_jax_array(walkers.log_shift, dtype="float64", force_host=True),
        as_jax_array(trial.psi0a, dtype="complex128", force_host=True),
        as_jax_array(trial.psi0b, dtype="complex128", force_host=True),
        as_jax_array(hamiltonian.T, dtype="complex128", force_host=True),
        as_jax_array(hamiltonian.U, dtype="complex128", force_host=True),
        as_jax_array(hamiltonian.ecore, dtype="complex128", force_host=True),
        as_jax_array(delta, dtype="complex128", force_host=True),
        as_jax_array(aux_wfac, dtype="complex128", force_host=True),
        as_jax_array(random_fields, dtype="float64", force_host=True),
        as_jax_array(pop_control_randoms, dtype="float64", force_host=True),
        0.01,
        walkers.nup,
        walkers.ndown,
        rhf=bool(walkers.rhf),
        trial_response=False,
        uhf_n_scf=1,
        stabilize_freq=99,
        pop_control_freq=1,
        measure_freq=2,
        local_pop_control=False,
        global_pop_control=True,
        mpi_comm=hcomm,
        mpi_rank=comm.rank,
        mpi_size=comm.size,
        checkpoint_steps=False,
        coupling_mode="diagonal",
    )
    result = block_until_ready(result)
    enumer, edenom, d_enumer, d_edenom = result[:4]
    return {
        "ENumer": numpy.asarray(enumer, dtype=numpy.complex128).reshape(1),
        "EDenom": numpy.asarray(edenom, dtype=numpy.complex128).reshape(1),
        "dENumer": numpy.asarray(d_enumer, dtype=numpy.complex128).ravel(),
        "dEDenom": numpy.asarray(d_edenom, dtype=numpy.complex128).ravel(),
    }


def _ad_block_value_for_coupling(
    walkers,
    hamiltonian,
    trial,
    coupling,
    random_fields,
    pop_control_randoms,
    comm,
    hcomm,
):
    _, _, aux_wfac, delta = construct_hirsch_auxiliaries(hamiltonian, 0.01, spin_decomp=True)
    phib = walkers.phib
    if phib is None:
        phib = numpy.zeros((walkers.nwalkers, walkers.nbasis, 0), dtype=numpy.complex128)
    result = hubbard_ad_block_value(
        as_jax_array(coupling, dtype="float64", force_host=True),
        as_jax_array(walkers.phia, dtype="complex128", force_host=True),
        as_jax_array(phib, dtype="complex128", force_host=True),
        as_jax_array(walkers.weight, dtype="float64", force_host=True),
        as_jax_array(walkers.log_shift, dtype="float64", force_host=True),
        as_jax_array(trial.psi0a, dtype="complex128", force_host=True),
        as_jax_array(trial.psi0b, dtype="complex128", force_host=True),
        as_jax_array(hamiltonian.T, dtype="complex128", force_host=True),
        as_jax_array(hamiltonian.U, dtype="complex128", force_host=True),
        as_jax_array(hamiltonian.ecore, dtype="complex128", force_host=True),
        as_jax_array(delta, dtype="complex128", force_host=True),
        as_jax_array(aux_wfac, dtype="complex128", force_host=True),
        as_jax_array(random_fields, dtype="float64", force_host=True),
        as_jax_array(pop_control_randoms, dtype="float64", force_host=True),
        0.01,
        walkers.nup,
        walkers.ndown,
        rhf=bool(walkers.rhf),
        trial_response=False,
        uhf_n_scf=1,
        stabilize_freq=99,
        pop_control_freq=1,
        measure_freq=2,
        local_pop_control=False,
        global_pop_control=True,
        mpi_comm=hcomm,
        mpi_rank=comm.rank,
        mpi_size=comm.size,
        checkpoint_steps=False,
        coupling_mode="diagonal",
    )
    result = block_until_ready(result)
    enumer, edenom = result
    return {
        "ENumer": numpy.asarray(enumer, dtype=numpy.complex128).reshape(1),
        "EDenom": numpy.asarray(edenom, dtype=numpy.complex128).reshape(1),
    }


@pytest.mark.unit
@pytest.mark.parametrize("rhf", [False, True])
def test_hubbard_jax_response_estimator_shape_and_finite(rhf):
    nelec = (2, 2) if rhf else (2, 1)
    ham, trial, initial_walker, nwalkers = _hubbard_inputs(nelec=nelec, rhf=rhf)
    walkers = _new_walkers(trial, initial_walker, nelec, ham.nbasis, nwalkers, rhf=rhf)
    random_fields = numpy.random.default_rng(7).random((1, ham.nbasis, nwalkers))
    estimator = HubbardJAX1RDMResponseEstimator(
        ham=ham, num_steps=1, timestep=0.01, random_fields=random_fields
    )

    data = estimator.compute_estimator(None, walkers, ham, trial)
    response = estimator["RDMResponse"].reshape(2, ham.nbasis, ham.nbasis)
    weight_response = estimator["WeightResponse"].reshape(2, ham.nbasis, ham.nbasis)

    assert estimator.shape == (2, ham.nbasis, ham.nbasis)
    assert data.size == estimator.size
    assert numpy.all(numpy.isfinite(response))
    assert numpy.all(numpy.isfinite(weight_response))


@pytest.mark.unit
def test_hubbard_jax_response_walker_batching_matches_unbatched_raw_response():
    nelec = (2, 1)
    ham, trial, initial_walker, nwalkers = _hubbard_inputs(nelec=nelec, nwalkers=4)
    walkers = _new_walkers(trial, initial_walker, nelec, ham.nbasis, nwalkers)
    random_fields = numpy.random.default_rng(13).random((2, ham.nbasis, nwalkers))
    estimator = HubbardJAX1RDMResponseEstimator(
        ham=ham, num_steps=2, timestep=0.01, walker_batch_size=None
    )
    batched_estimator = HubbardJAX1RDMResponseEstimator(
        ham=ham, num_steps=2, timestep=0.01, walker_batch_size=2
    )

    raw = estimator._compute_raw_response(walkers, ham, trial, random_fields=random_fields)
    raw_batched = batched_estimator._compute_raw_response(
        walkers, ham, trial, random_fields=random_fields
    )

    for key in raw:
        numpy.testing.assert_allclose(raw_batched[key], raw[key], atol=1e-9, rtol=1e-9)


@pytest.mark.unit
def test_hubbard_jax_ad_block_matches_response_raw_without_pop_control(monkeypatch):
    nelec = (2, 1)
    ham, trial, initial_walker, nwalkers = _hubbard_inputs(nelec=nelec, nwalkers=4)
    walkers_ref = _new_walkers(trial, initial_walker, nelec, ham.nbasis, nwalkers)
    walkers_ad = _new_walkers(trial, initial_walker, nelec, ham.nbasis, nwalkers)
    random_fields = numpy.random.default_rng(15).random((2, ham.nbasis, nwalkers))
    pop_randoms = numpy.random.default_rng(16).random((2,))
    draws = iter([random_fields, pop_randoms])
    monkeypatch.setattr(
        hubbard_jax_ad, "_random_uniform", lambda shape: numpy.asarray(next(draws)).copy()
    )

    estimator = HubbardJAX1RDMResponseEstimator(ham=ham, num_steps=2, timestep=0.01)
    raw = estimator._compute_raw_response(walkers_ref, ham, trial, random_fields=random_fields)
    runner = HubbardJAXADBlockRunner(
        ham,
        trial,
        timestep=0.01,
        ad_block_size=2,
        measure_freq=2,
        stabilize_freq=99,
        pop_control_freq=99,
        local_pop_control=False,
    )
    result = runner.run_block(walkers_ad)

    numpy.testing.assert_allclose(result.ENumer, raw["ENumer"], atol=1e-10, rtol=1e-10)
    numpy.testing.assert_allclose(result.EDenom, raw["EDenom"], atol=1e-10, rtol=1e-10)
    numpy.testing.assert_allclose(result.dENumer, raw["dENumer"], atol=1e-10, rtol=1e-10)
    numpy.testing.assert_allclose(result.dEDenom, raw["dEDenom"], atol=1e-10, rtol=1e-10)
    assert numpy.all(numpy.isfinite(result.RDMResponse))


@pytest.mark.unit
def test_hubbard_jax_ad_block_local_pop_control_resets_chunk_weights(monkeypatch):
    nelec = (2, 1)
    ham, trial, initial_walker, nwalkers = _hubbard_inputs(nelec=nelec, nwalkers=4)
    walkers = _new_walkers(trial, initial_walker, nelec, ham.nbasis, nwalkers)
    random_fields = numpy.random.default_rng(17).random((2, ham.nbasis, nwalkers))
    pop_randoms = numpy.random.default_rng(18).random((2,))
    draws = iter([random_fields[:, :, :2], pop_randoms, random_fields[:, :, 2:], pop_randoms])
    monkeypatch.setattr(
        hubbard_jax_ad, "_random_uniform", lambda shape: numpy.asarray(next(draws)).copy()
    )

    runner = HubbardJAXADBlockRunner(
        ham,
        trial,
        timestep=0.01,
        ad_block_size=2,
        measure_freq=2,
        stabilize_freq=99,
        pop_control_freq=2,
        walker_batch_size=2,
        local_pop_control=True,
    )
    result = runner.run_block(walkers)

    numpy.testing.assert_allclose(numpy.asarray(walkers.weight), numpy.ones(nwalkers))
    assert numpy.all(numpy.isfinite(result.RDMResponse))


@pytest.mark.unit
def test_hubbard_jax_ad_block_diagonal_response_matches_full_diagonal(monkeypatch):
    nelec = (2, 1)
    ham, trial, initial_walker, nwalkers = _hubbard_inputs(nelec=nelec, nwalkers=4)
    walkers_full = _new_walkers(trial, initial_walker, nelec, ham.nbasis, nwalkers)
    walkers_diag = _new_walkers(trial, initial_walker, nelec, ham.nbasis, nwalkers)
    random_fields = numpy.random.default_rng(19).random((2, ham.nbasis, nwalkers))
    pop_randoms = numpy.random.default_rng(20).random((2,))
    draws = iter([random_fields, pop_randoms, random_fields, pop_randoms])
    monkeypatch.setattr(
        hubbard_jax_ad, "_random_uniform", lambda shape: numpy.asarray(next(draws)).copy()
    )
    common = {
        "timestep": 0.01,
        "ad_block_size": 2,
        "measure_freq": 2,
        "stabilize_freq": 99,
        "pop_control_freq": 99,
        "local_pop_control": False,
    }

    result_full = HubbardJAXADBlockRunner(ham, trial, response_mode="full", **common).run_block(
        walkers_full
    )
    result_diag = HubbardJAXADBlockRunner(ham, trial, response_mode="diagonal", **common).run_block(
        walkers_diag
    )

    full = result_full.RDMResponse.reshape(2, ham.nbasis, ham.nbasis)
    diag = result_diag.RDMResponse.reshape(2, ham.nbasis, ham.nbasis)
    idx = numpy.arange(ham.nbasis)
    mask = numpy.ones_like(diag, dtype=bool)
    mask[:, idx, idx] = False
    numpy.testing.assert_allclose(diag[:, idx, idx], full[:, idx, idx], atol=1e-10, rtol=1e-10)
    numpy.testing.assert_allclose(diag[mask], 0.0, atol=1e-12)


@pytest.mark.mpi
def test_mpi4jax_allgather_ad_moves_cotangents_between_ranks():
    pytest.importorskip("mpi4jax")
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires at least two MPI ranks")

    hcomm = hashable_mpi_comm(comm)
    source_rank = 1 if comm.rank == 0 else 0

    @jax.jit
    def remote_objective(x):
        gathered = _mpi_allgather_ad(x, hcomm, comm.rank, comm.size).reshape((comm.size,))
        return 2.0 * gathered[source_rank]

    grad = jax.grad(remote_objective)(as_jax_array(float(comm.rank + 1), dtype="float64"))
    grad = numpy.asarray(block_until_ready(grad))
    expected = 2.0 * (comm.size - 1) if comm.rank == 0 else (2.0 if comm.rank == 1 else 0.0)

    numpy.testing.assert_allclose(grad, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.mpi
def test_hubbard_jax_mpi4jax_ad_block_response_matches_finite_difference_after_afqmc():
    pytest.importorskip("mpi4jax")
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("requires at least two MPI ranks")

    mpi_handler = MPIHandler()
    ham, trial, _, _, nelec = _hubbard_2x2_inputs(nwalkers=10)
    afqmc = AFQMC.build(
        nelec,
        ham,
        trial,
        num_walkers=10,
        num_steps_per_block=1,
        num_blocks=0,
        timestep=0.01,
        stabilize_freq=99,
        pop_control_freq=99,
        num_eq_blocks=1,
        eq_num_steps_per_block=1,
        eq_stabilize_freq=99,
        eq_pop_control_freq=99,
        seed=71,
        propagator_backend="jax",
        verbose=False,
        mpi_handler=mpi_handler,
    )

    if comm.rank == 0:
        tmp = tempfile.NamedTemporaryFile(delete=False)
        estimator_filename = tmp.name
        tmp.close()
    else:
        estimator_filename = None
    estimator_filename = comm.bcast(estimator_filename, root=0)
    try:
        afqmc.run(
            estimator_filename=estimator_filename, verbose=False, discard_weights_aftereq=True
        )
    finally:
        comm.Barrier()
        if comm.rank == 0 and estimator_filename is not None:
            try:
                os.unlink(estimator_filename)
            except OSError:
                pass
        comm.Barrier()

    if comm.rank == 0:
        biased_weights = numpy.linspace(4.0, 5.0, afqmc.walkers.nwalkers)
    else:
        biased_weights = numpy.linspace(0.01, 0.02, afqmc.walkers.nwalkers)
    afqmc.walkers.weight[:] = biased_weights
    afqmc.walkers.unscaled_weight = afqmc.walkers.weight.copy()

    random_fields = numpy.random.default_rng(200 + comm.rank).random(
        (2, ham.nbasis, afqmc.walkers.nwalkers)
    )
    pop_control_randoms = numpy.array([0.17, 0.31], dtype=numpy.float64)
    hcomm = hashable_mpi_comm(comm)

    coupling = numpy.zeros((2, ham.nbasis), dtype=numpy.float64)
    raw = _reduce_raw(
        _ad_block_raw_for_coupling(
            afqmc.walkers,
            ham,
            trial,
            coupling,
            random_fields,
            pop_control_randoms,
            comm,
            hcomm,
        ),
        comm,
    )
    ad_response = _ratio_gradient(raw).reshape(2, ham.nbasis)

    eps = 1.0e-5
    coupling_p = coupling.copy()
    coupling_m = coupling.copy()
    coupling_p[0, 0] = eps
    coupling_m[0, 0] = -eps
    raw_p = _reduce_raw(
        _ad_block_value_for_coupling(
            afqmc.walkers,
            ham,
            trial,
            coupling_p,
            random_fields,
            pop_control_randoms,
            comm,
            hcomm,
        ),
        comm,
    )
    raw_m = _reduce_raw(
        _ad_block_value_for_coupling(
            afqmc.walkers,
            ham,
            trial,
            coupling_m,
            random_fields,
            pop_control_randoms,
            comm,
            hcomm,
        ),
        comm,
    )
    energy_p = raw_p["ENumer"][0] / raw_p["EDenom"][0]
    energy_m = raw_m["ENumer"][0] / raw_m["EDenom"][0]
    finite_difference = (energy_p - energy_m) / (2.0 * eps)

    numpy.testing.assert_allclose(
        ad_response[0, 0].real,
        finite_difference.real,
        atol=8.0e-5,
        rtol=8.0e-5,
    )


@pytest.mark.unit
def test_hubbard_jax_response_matches_finite_difference_fixed_trial():
    nelec = (2, 1)
    ham, trial, initial_walker, nwalkers = _hubbard_inputs(nelec=nelec)
    walkers = _new_walkers(trial, initial_walker, nelec, ham.nbasis, nwalkers)
    random_fields = numpy.random.default_rng(11).random((1, ham.nbasis, nwalkers))
    estimator = HubbardJAX1RDMResponseEstimator(ham=ham, num_steps=1, timestep=0.01)

    raw = estimator._compute_raw_response(walkers, ham, trial, random_fields=random_fields)
    response = _ratio_gradient(raw).reshape(2, ham.nbasis, ham.nbasis)

    eps = 1.0e-5
    coupling_p = numpy.zeros((2, ham.nbasis, ham.nbasis), dtype=numpy.float64)
    coupling_m = numpy.zeros_like(coupling_p)
    coupling_p[0, 1, 2] = eps
    coupling_m[0, 1, 2] = -eps
    fd = (
        _objective_value(estimator, walkers, ham, trial, random_fields, coupling_p)
        - _objective_value(estimator, walkers, ham, trial, random_fields, coupling_m)
    ) / (2.0 * eps)

    numpy.testing.assert_allclose(response[0, 1, 2].real, fd.real, atol=2e-5, rtol=2e-5)


@pytest.mark.unit
def test_hubbard_jax_response_full_alpha_1rdm_matches_fixed_trial_finite_difference():
    ham, trial, initial_walker, nwalkers, nelec = _hubbard_2x2_inputs()
    walkers = _new_walkers(trial, initial_walker, nelec, ham.nbasis, nwalkers)
    random_fields = numpy.random.default_rng(31).random((1, ham.nbasis, nwalkers))
    estimator = HubbardJAX1RDMResponseEstimator(
        ham=ham,
        num_steps=1,
        timestep=0.01,
        trial_response=False,
    )

    raw = estimator._compute_raw_response(walkers, ham, trial, random_fields=random_fields)
    ad_response = _ratio_gradient(raw).reshape(2, ham.nbasis, ham.nbasis)

    eps = 1.0e-5
    spin = 0
    fd_response = numpy.zeros((ham.nbasis, ham.nbasis), dtype=numpy.float64)
    for p in range(ham.nbasis):
        for q in range(ham.nbasis):
            coupling_p = numpy.zeros((2, ham.nbasis, ham.nbasis), dtype=numpy.float64)
            coupling_m = numpy.zeros_like(coupling_p)
            coupling_p[spin, p, q] = eps
            coupling_m[spin, p, q] = -eps
            fd_response[p, q] = (
                _objective_value(estimator, walkers, ham, trial, random_fields, coupling_p)
                - _objective_value(estimator, walkers, ham, trial, random_fields, coupling_m)
            ).real / (2.0 * eps)

    numpy.testing.assert_allclose(ad_response[spin].imag, 0.0, atol=1e-12)
    numpy.testing.assert_allclose(ad_response[spin].real, fd_response, atol=3e-5, rtol=3e-5)


@pytest.mark.unit
def test_hubbard_jax_response_zero_weight_walkers_are_finite():
    nelec = (2, 1)
    ham, trial, initial_walker, nwalkers = _hubbard_inputs(nelec=nelec)
    walkers = _new_walkers(trial, initial_walker, nelec, ham.nbasis, nwalkers)
    walkers.weight[:] = 0.0
    random_fields = numpy.random.default_rng(13).random((1, ham.nbasis, nwalkers))
    estimator = HubbardJAX1RDMResponseEstimator(
        ham=ham, num_steps=1, timestep=0.01, random_fields=random_fields
    )

    estimator.compute_estimator(None, walkers, ham, trial)

    numpy.testing.assert_allclose(estimator["EDenom"], 0.0)
    numpy.testing.assert_allclose(estimator["RDMResponse"], 0.0)
    assert numpy.all(numpy.isfinite(estimator["dENumer"]))
    assert numpy.all(numpy.isfinite(estimator["dEDenom"]))


@pytest.mark.unit
def test_hubbard_jax_response_post_reduce_matches_unsplit_walkers():
    nelec = (2, 1)
    ham, trial, initial_walker, nwalkers = _hubbard_inputs(nelec=nelec, nwalkers=4)
    walkers = _new_walkers(trial, initial_walker, nelec, ham.nbasis, nwalkers)
    random_fields = numpy.random.default_rng(17).random((1, ham.nbasis, nwalkers))
    estimator = HubbardJAX1RDMResponseEstimator(ham=ham, num_steps=1, timestep=0.01)

    full = estimator._compute_raw_response(walkers, ham, trial, random_fields=random_fields)
    left = estimator._compute_raw_response(
        _walker_view(walkers, 0, 2), ham, trial, random_fields=random_fields[:, :, 0:2]
    )
    right = estimator._compute_raw_response(
        _walker_view(walkers, 2, 4), ham, trial, random_fields=random_fields[:, :, 2:4]
    )

    for key in ("ENumer", "EDenom", "dENumer", "dEDenom"):
        numpy.testing.assert_allclose(left[key] + right[key], full[key], atol=1e-10, rtol=1e-10)

    combined = estimator.data.copy()
    combined[estimator._slice("ENumer")] = left["ENumer"] + right["ENumer"]
    combined[estimator._slice("EDenom")] = left["EDenom"] + right["EDenom"]
    combined[estimator._slice("dENumer")] = left["dENumer"] + right["dENumer"]
    combined[estimator._slice("dEDenom")] = left["dEDenom"] + right["dEDenom"]
    estimator.post_reduce_hook(combined)

    numpy.testing.assert_allclose(
        combined[estimator._slice("RDMResponse")], _ratio_gradient(full), atol=1e-10, rtol=1e-10
    )


@pytest.mark.unit
def test_hubbard_jax_uhf_one_iteration_matches_legacy_equations():
    nelec = (2, 1)
    ham, trial, _, _ = _hubbard_inputs(nelec=nelec)
    niup = numpy.diag(trial.psi0a @ trial.psi0a.conj().T).real
    nidown = numpy.diag(trial.psi0b @ trial.psi0b.conj().T).real

    psi0a, psi0b = hubbard_uhf_trial_from_densities(
        as_jax_array(ham.T, dtype="complex128"),
        ham.U,
        nelec[0],
        nelec[1],
        as_jax_array(niup, dtype="float64"),
        as_jax_array(nidown, dtype="float64"),
        n_scf=1,
        mixing=0.5,
    )
    psi0a = numpy.asarray(block_until_ready(psi0a))
    psi0b = numpy.asarray(block_until_ready(psi0b))

    _, ev_up = scipy.linalg.eigh(ham.T[0] + numpy.diag(ham.U * nidown))
    _, ev_down = scipy.linalg.eigh(ham.T[1] + numpy.diag(ham.U * niup))
    ev_up = _gauge_fix_numpy(ev_up)
    ev_down = _gauge_fix_numpy(ev_down)

    numpy.testing.assert_allclose(psi0a, ev_up[:, : nelec[0]], atol=1e-10, rtol=1e-10)
    numpy.testing.assert_allclose(psi0b, ev_down[:, : nelec[1]], atol=1e-10, rtol=1e-10)


@pytest.mark.unit
def test_hubbard_jax_response_matches_finite_difference_with_uhf_trial_response():
    nelec = (2, 1)
    ham, trial, initial_walker, nwalkers = _hubbard_inputs(nelec=nelec)
    walkers = _new_walkers(trial, initial_walker, nelec, ham.nbasis, nwalkers)
    random_fields = numpy.random.default_rng(23).random((1, ham.nbasis, nwalkers))
    estimator = HubbardJAX1RDMResponseEstimator(
        ham=ham,
        num_steps=1,
        timestep=0.01,
        trial_response="jax_uhf",
        uhf_n_scf=2,
    )

    raw = estimator._compute_raw_response(walkers, ham, trial, random_fields=random_fields)
    response = _ratio_gradient(raw).reshape(2, ham.nbasis, ham.nbasis)

    eps = 1.0e-5
    coupling_p = numpy.zeros((2, ham.nbasis, ham.nbasis), dtype=numpy.float64)
    coupling_m = numpy.zeros_like(coupling_p)
    coupling_p[0, 1, 1] = eps
    coupling_m[0, 1, 1] = -eps
    fd = (
        _objective_value(estimator, walkers, ham, trial, random_fields, coupling_p)
        - _objective_value(estimator, walkers, ham, trial, random_fields, coupling_m)
    ) / (2.0 * eps)

    numpy.testing.assert_allclose(response[0, 1, 1].real, fd.real, atol=5e-5, rtol=5e-5)
