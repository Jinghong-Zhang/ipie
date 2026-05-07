import numpy
import pytest

from ipie.hamiltonians.hubbard import Hubbard
from ipie.trial_wavefunction.hubbard_ueff import (
    extract_qmc_1rdm,
    fit_ueff_to_qmc_density,
    update_trial_by_ueff_density_match,
)
from ipie.trial_wavefunction.hubbard_uhf import (
    build_slater_trial_from_uhf_solution,
    solve_hubbard_uhf_fixed_ueff,
)
from ipie.trial_wavefunction.single_det import SingleDet


def _chain_hopping(nsites=4):
    K = numpy.zeros((nsites, nsites), dtype=numpy.float64)
    for i in range(nsites - 1):
        K[i, i + 1] = K[i + 1, i] = -1.0
    K += numpy.diag(numpy.linspace(-0.4, 0.35, nsites))
    return K


def _uhf_options():
    return {"max_iter": 500, "mixing": 0.4, "tol": 1e-11}


def _trial_from_solution(solution, nelec):
    psi = build_slater_trial_from_uhf_solution(solution)
    return SingleDet(psi, nelec, psi.shape[0])


def _rho_from_solution(solution):
    return numpy.asarray([solution["rho_up"], solution["rho_down"]])


@pytest.mark.unit
def test_fit_ueff_to_qmc_density_recovers_known_uhf_target_density():
    K = _chain_hopping()
    nup = ndown = 2
    target_ueff = 1.7
    target = solve_hubbard_uhf_fixed_ueff(
        K, nup, ndown, target_ueff, **_uhf_options()
    )

    fit = fit_ueff_to_qmc_density(
        K,
        nup,
        ndown,
        target["density_up"],
        target["density_down"],
        (0.0, 3.0),
        grid_size=9,
        optimizer="grid_then_brent",
        uhf_options=_uhf_options(),
    )

    assert fit["best_delta"] < 1e-7
    assert abs(fit["best_ueff"] - target_ueff) < 1e-3
    numpy.testing.assert_allclose(
        fit["best_solution"]["density_up"], target["density_up"], atol=1e-7, rtol=0.0
    )
    numpy.testing.assert_allclose(
        fit["best_solution"]["density_down"], target["density_down"], atol=1e-7, rtol=0.0
    )


@pytest.mark.unit
def test_grid_objective_is_smallest_near_generating_ueff_closed_shell():
    K = _chain_hopping()
    nup = ndown = 2
    target_ueff = 1.5
    target = solve_hubbard_uhf_fixed_ueff(
        K, nup, ndown, target_ueff, **_uhf_options()
    )

    fit = fit_ueff_to_qmc_density(
        K,
        nup,
        ndown,
        target["density_up"],
        target["density_down"],
        (0.0, 3.0),
        grid_size=7,
        optimizer="grid",
        uhf_options=_uhf_options(),
    )
    grid_history = {round(item["ueff"], 12): item["delta"] for item in fit["history"]}

    assert fit["best_ueff"] == pytest.approx(target_ueff)
    assert grid_history[1.5] < grid_history[1.0]
    assert grid_history[1.5] < grid_history[2.0]


@pytest.mark.unit
def test_fit_ueff_warns_on_inconsistent_qmc_density_trace_and_keeps_raw_trace():
    K = _chain_hopping()
    nup = ndown = 2
    target = solve_hubbard_uhf_fixed_ueff(K, nup, ndown, 1.0, **_uhf_options())
    bad_up = 0.95 * target["density_up"]

    with pytest.warns(RuntimeWarning, match="n_qmc_up trace"):
        fit = fit_ueff_to_qmc_density(
            K,
            nup,
            ndown,
            bad_up,
            target["density_down"],
            (0.0, 2.0),
            grid_size=5,
            optimizer="grid",
            uhf_options=_uhf_options(),
        )

    assert numpy.isfinite(fit["best_delta"])
    assert fit["qmc_trace_up"] == pytest.approx(0.95 * nup)
    assert fit["qmc_trace_down"] == pytest.approx(ndown)


@pytest.mark.unit
def test_update_trial_by_ueff_density_match_returns_valid_single_det_trial():
    K = _chain_hopping()
    nup = ndown = 2
    ham = Hubbard(numpy.asarray([K, K]), U=3.0)
    previous = solve_hubbard_uhf_fixed_ueff(K, nup, ndown, 0.5, **_uhf_options())
    previous_trial = _trial_from_solution(previous, (nup, ndown))
    target = solve_hubbard_uhf_fixed_ueff(K, nup, ndown, 1.25, **_uhf_options())
    rho = _rho_from_solution(target)

    trial, diagnostics = update_trial_by_ueff_density_match(
        {"RDMResponse": rho.ravel(), "shape": rho.shape},
        ham,
        previous_trial,
        ueff_bounds=(0.0, 3.0),
        grid_size=7,
        uhf_options=_uhf_options(),
    )

    assert isinstance(trial, SingleDet)
    assert trial.nalpha == nup
    assert trial.nbeta == ndown
    assert trial.psi0a.shape == (K.shape[0], nup)
    assert trial.psi0b.shape == (K.shape[0], ndown)
    assert trial.half_rotated
    assert diagnostics["delta"] < 1e-6
    assert diagnostics["qmc_trace_up"] == pytest.approx(nup)
    assert diagnostics["ip_trace_down"] == pytest.approx(ndown)
    numpy.testing.assert_allclose(diagnostics["n_qmc_up"], numpy.diag(rho[0]).real)
    assert diagnostics["natural_occupations_up"].shape == (K.shape[0],)


@pytest.mark.unit
def test_update_trial_by_ueff_density_match_smoke_two_outer_iterations():
    K = _chain_hopping()
    nup = ndown = 2
    ham = Hubbard(numpy.asarray([K, K]), U=4.0)
    previous_solution = solve_hubbard_uhf_fixed_ueff(
        K, nup, ndown, 0.5, **_uhf_options()
    )
    trial = _trial_from_solution(previous_solution, (nup, ndown))

    for target_ueff in (1.0, 1.6):
        target = solve_hubbard_uhf_fixed_ueff(
            K, nup, ndown, target_ueff, **_uhf_options()
        )
        trial, diagnostics = update_trial_by_ueff_density_match(
            {"rho_qmc": _rho_from_solution(target)},
            ham,
            trial,
            ueff_bounds=(0.0, 3.0),
            grid_size=7,
            uhf_options=_uhf_options(),
        )

        assert isinstance(trial, SingleDet)
        assert trial.psi.shape == (K.shape[0], nup + ndown)
        assert numpy.isfinite(diagnostics["best_ueff"])
        assert numpy.isfinite(diagnostics["delta"])
        assert len(diagnostics["history"]) >= 7


@pytest.mark.unit
def test_extract_qmc_1rdm_supports_estimator_like_object():
    rho = numpy.arange(32, dtype=numpy.float64).reshape(2, 4, 4)

    class EstimatorLike:
        shape = (2, 4, 4)

        def __getitem__(self, name):
            if name == "RDMResponse":
                return rho.ravel()
            raise RuntimeError(f"Unknown estimator {name}")

    estimator_like = EstimatorLike()

    extracted = extract_qmc_1rdm(estimator_like, nbasis=4)

    numpy.testing.assert_allclose(extracted, rho)
