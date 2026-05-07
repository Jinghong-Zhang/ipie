import numpy
import pytest

from ipie.addons.self_consistent.trial_generation import (
    MixedOneRDMAccumulator,
    MixedOneRDMElementEstimator,
    MixedOneRDMEstimator,
    build_ipie_single_det_trial,
    compute_mixed_1rdm_from_ipie_walkers,
    generate_self_consistent_trial,
    green_to_density,
    mixed_1rdm_numerators_from_ipie_walkers,
    natural_orbitals_from_rho,
)
from ipie.hamiltonians.hubbard import Hubbard
from ipie.trial_wavefunction.single_det import SingleDet
from ipie.utils.mpi import MPIHandler
from ipie.walkers.uhf_walkers import UHFWalkers


def _build_tiny_hubbard_case(nelec=(2, 1), nbasis=4, nwalkers=4):
    h1e = numpy.zeros((2, nbasis, nbasis))
    hamiltonian = Hubbard(h1e, U=1.0)
    psi = numpy.concatenate(
        [numpy.eye(nbasis)[:, : nelec[0]], numpy.eye(nbasis)[:, : nelec[1]]], axis=1
    )
    trial = SingleDet(psi, nelec, nbasis, verbose=False)
    trial.half_rotate(hamiltonian)
    walkers = UHFWalkers(psi, nelec[0], nelec[1], nbasis, nwalkers, MPIHandler())
    walkers.build(trial)
    return hamiltonian, trial, walkers


@pytest.mark.unit
def test_green_to_density_conventions():
    G = numpy.array([[0.2, 0.3], [0.4, 0.7]])
    eye = numpy.eye(2)

    numpy.testing.assert_allclose(green_to_density(G, "ipie_default"), G)
    numpy.testing.assert_allclose(green_to_density(G, "direct"), G)
    numpy.testing.assert_allclose(green_to_density(G, "transpose"), G.T)
    numpy.testing.assert_allclose(green_to_density(G, "one_minus_direct"), eye - G)
    numpy.testing.assert_allclose(green_to_density(G, "one_minus_transpose"), eye - G.T)

    with pytest.raises(ValueError):
        green_to_density(G, "unknown")


@pytest.mark.unit
def test_natural_orbital_update_recovers_known_projector():
    theta = 0.37
    u_occ = numpy.array(
        [
            [numpy.cos(theta), 0.0],
            [numpy.sin(theta), 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ]
    )
    rho = u_occ @ u_occ.T

    phi, occ = natural_orbitals_from_rho(rho, 2, hermitize=True)
    projector = phi @ phi.conj().T

    numpy.testing.assert_allclose(projector, rho, atol=1.0e-12)
    numpy.testing.assert_allclose(occ[:2], [1.0, 1.0], atol=1.0e-12)
    numpy.testing.assert_allclose(occ[2:], [0.0, 0.0], atol=1.0e-12)


@pytest.mark.unit
def test_mixed_1rdm_trace_from_zero_temperature_ipie_walkers():
    nelec = (2, 1)
    _hamiltonian, trial, walkers = _build_tiny_hubbard_case(nelec=nelec)

    rho_a, rho_b = compute_mixed_1rdm_from_ipie_walkers(
        trial, walkers, nelec, green_convention="ipie_default"
    )

    assert numpy.trace(rho_a).real == pytest.approx(nelec[0])
    assert numpy.trace(rho_b).real == pytest.approx(nelec[1])


@pytest.mark.unit
def test_mixed_1rdm_accumulator_matches_repeated_walker_average():
    nelec = (2, 1)
    _hamiltonian, trial, walkers = _build_tiny_hubbard_case(nelec=nelec)

    rho_a, rho_b = compute_mixed_1rdm_from_ipie_walkers(
        trial, walkers, nelec, green_convention="ipie_default"
    )
    accumulator = MixedOneRDMAccumulator(nelec, green_convention="ipie_default")
    accumulator.update(trial, walkers)
    accumulator.update(trial, walkers)
    rho_a_acc, rho_b_acc = accumulator.finalize()

    assert accumulator.num_samples == 2
    numpy.testing.assert_allclose(rho_a_acc, rho_a)
    numpy.testing.assert_allclose(rho_b_acc, rho_b)


@pytest.mark.unit
def test_mixed_1rdm_estimator_outputs_block_numerators():
    nelec = (2, 1)
    _hamiltonian, trial, walkers = _build_tiny_hubbard_case(nelec=nelec)
    estimator = MixedOneRDMEstimator(
        nelec,
        nbasis=trial.nbasis,
        green_convention="ipie_default",
    )

    data = estimator.compute_estimator(walkers=walkers, trial=trial)
    rho_a_num, rho_b_num, denom = mixed_1rdm_numerators_from_ipie_walkers(
        trial,
        walkers,
        nelec,
        green_convention="ipie_default",
    )

    matrix_size = trial.nbasis * trial.nbasis
    assert data.shape == (2 * matrix_size + 1,)
    numpy.testing.assert_allclose(data[:matrix_size].reshape(trial.nbasis, trial.nbasis), rho_a_num)
    numpy.testing.assert_allclose(
        data[matrix_size : 2 * matrix_size].reshape(trial.nbasis, trial.nbasis),
        rho_b_num,
    )
    numpy.testing.assert_allclose(data[-1], denom)


@pytest.mark.unit
def test_mixed_1rdm_element_estimator_outputs_normalized_element():
    nelec = (2, 1)
    _hamiltonian, trial, walkers = _build_tiny_hubbard_case(nelec=nelec)
    estimator = MixedOneRDMElementEstimator(
        nelec,
        element=(0, 0),
        green_convention="ipie_default",
    )

    data = estimator.compute_estimator(walkers=walkers, trial=trial)
    estimator.post_reduce_hook(data)
    rho_a, rho_b = compute_mixed_1rdm_from_ipie_walkers(
        trial,
        walkers,
        nelec,
        green_convention="ipie_default",
    )

    assert list(estimator.names) == [
        "RhoA00Numer",
        "RhoB00Numer",
        "RhoDenom",
        "RhoA00",
        "RhoB00",
    ]
    numpy.testing.assert_allclose(data[3], rho_a[0, 0])
    numpy.testing.assert_allclose(data[4], rho_b[0, 0])


@pytest.mark.unit
def test_self_consistency_wrapper_with_mocked_afqmc_run():
    nelec = (2, 1)
    nmo = 4
    hamiltonian, trial, walkers = _build_tiny_hubbard_case(nelec=nelec, nbasis=nmo)
    target_a = numpy.eye(nmo)[:, : nelec[0]]
    target_b = numpy.eye(nmo)[:, : nelec[1]]

    class Result:
        def __init__(self, walkers, energy_mean):
            self.walkers = walkers
            self.energy_mean = energy_mean

    calls = []

    def run_afqmc_once(hamiltonian, trial, afqmc_options):
        calls.append(trial)
        walkers.Ga[:] = target_a @ target_a.T
        walkers.Gb[:] = target_b @ target_b.T
        walkers.weight[:] = 1.0
        return Result(walkers, -1.0)

    final_trial, history = generate_self_consistent_trial(
        hamiltonian,
        trial,
        run_afqmc_once,
        {},
        nelec,
        max_iter=3,
        rho_tol=1.0e-12,
        subspace_tol=1.0e-12,
        verbose=False,
    )

    assert isinstance(final_trial, SingleDet)
    assert len(history) == 2
    assert len(calls) == 2
    assert history[0]["energy_mean"] == -1.0
    assert history[-1]["rho_change"] == pytest.approx(0.0)
    assert history[-1]["subspace_change"] == pytest.approx(0.0)
    numpy.testing.assert_allclose(final_trial.psi0a @ final_trial.psi0a.T, target_a @ target_a.T)
    numpy.testing.assert_allclose(final_trial.psi0b @ final_trial.psi0b.T, target_b @ target_b.T)


@pytest.mark.unit
def test_self_consistency_wrapper_prefers_averaged_rho_result():
    nelec = (2, 1)
    nmo = 4
    hamiltonian, trial, _walkers = _build_tiny_hubbard_case(nelec=nelec, nbasis=nmo)
    target_a = numpy.eye(nmo)[:, [1, 2]]
    target_b = numpy.eye(nmo)[:, [3]]

    class Result:
        def __init__(self):
            self.rho_a = target_a @ target_a.T
            self.rho_b = target_b @ target_b.T
            self.energy_mean = -1.0
            self.rho_num_samples = 7

    calls = []

    def run_afqmc_once(hamiltonian, trial, afqmc_options):
        calls.append(trial)
        return Result()

    final_trial, history = generate_self_consistent_trial(
        hamiltonian,
        trial,
        run_afqmc_once,
        {},
        nelec,
        max_iter=2,
        rho_tol=1.0e-12,
        subspace_tol=1.0e-12,
        verbose=False,
    )

    assert len(calls) == 2
    assert history[0]["rho_num_samples"] == 7
    numpy.testing.assert_allclose(final_trial.psi0a @ final_trial.psi0a.T, target_a @ target_a.T)
    numpy.testing.assert_allclose(final_trial.psi0b @ final_trial.psi0b.T, target_b @ target_b.T)


@pytest.mark.unit
def test_returned_single_det_trial_works_with_zero_temperature_walkers():
    nelec = (2, 1)
    nmo = 4
    hamiltonian, old_trial, walkers = _build_tiny_hubbard_case(nelec=nelec, nbasis=nmo)

    phi_a = numpy.eye(nmo)[:, : nelec[0]]
    phi_b = numpy.eye(nmo)[:, : nelec[1]]
    trial = build_ipie_single_det_trial(
        phi_a, phi_b, old_trial, hamiltonian, nelec, verbose=False
    )

    ovlp = trial.calc_greens_function(walkers, build_full=True)

    assert isinstance(trial, SingleDet)
    assert ovlp.shape == (walkers.nwalkers,)
    assert trial.energy is not None
