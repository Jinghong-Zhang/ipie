from types import SimpleNamespace

import numpy
import pytest

from ipie.legacy.trial_wavefunction.hubbard_uhf import HubbardUHF as LegacyHubbardUHF
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


def _complex_chain_hopping():
    K = _chain_hopping(4).astype(numpy.complex128)
    phase = numpy.exp(0.23j)
    K[0, 1] = -0.8 * phase
    K[1, 0] = K[0, 1].conj()
    K[2, 3] = -0.6 * phase.conj()
    K[3, 2] = K[2, 3].conj()
    return K


def _gauge_fix(orbitals):
    orbitals = numpy.array(orbitals, copy=True)
    max_abs = numpy.argmax(numpy.abs(orbitals), axis=0)
    values = orbitals[max_abs, numpy.arange(orbitals.shape[1])]
    phase = numpy.where(numpy.abs(values) > 0.0, values / numpy.abs(values), 1.0 + 0.0j)
    return orbitals / phase[numpy.newaxis, :]


def _assert_projector(rho, nocc):
    numpy.testing.assert_allclose(rho, rho.conj().T, atol=1e-10, rtol=1e-10)
    numpy.testing.assert_allclose(rho @ rho, rho, atol=1e-10, rtol=1e-10)
    numpy.testing.assert_allclose(numpy.trace(rho).real, nocc, atol=1e-10, rtol=0.0)


def _legacy_hubbard_uhf_loop(K, nup, ndown, ueff, initial_density, max_iter, mixing, tol):
    legacy = object.__new__(LegacyHubbardUHF)
    legacy.trial = numpy.zeros((K.shape[0], nup + ndown), dtype=numpy.complex128)
    system = SimpleNamespace(nup=nup, ndown=ndown)
    hamiltonian = SimpleNamespace(T=numpy.array([K, K]), nbasis=K.shape[0], U=ueff)
    density_up, density_down = (numpy.array(d, dtype=numpy.float64) for d in initial_density)
    legacy_alpha = 1.0 - mixing

    for iteration in range(1, max_iter + 1):
        old_up = density_up
        old_down = density_down
        density_up, density_down, eigs_up, eigs_down = legacy.diagonalise_mean_field(
            system, hamiltonian, ueff, density_up, density_down
        )
        residual = numpy.sqrt(
            numpy.mean((density_up - old_up) ** 2 + (density_down - old_down) ** 2)
        )
        if residual < tol:
            break
        density_up = legacy.mix_density(density_up, old_up, legacy_alpha)
        density_down = legacy.mix_density(density_down, old_down, legacy_alpha)

    phi_up = legacy.trial[:, :nup]
    phi_down = legacy.trial[:, nup:]
    return {
        "phi_up": phi_up,
        "phi_down": phi_down,
        "density_up": numpy.sum(numpy.abs(phi_up) ** 2, axis=1).real,
        "density_down": numpy.sum(numpy.abs(phi_down) ** 2, axis=1).real,
        "rho_up": phi_up @ phi_up.conj().T,
        "rho_down": phi_down @ phi_down.conj().T,
        "eigenvalues_up": eigs_up,
        "eigenvalues_down": eigs_down,
        "niter": iteration,
        "residual": residual,
    }


@pytest.mark.unit
def test_hubbard_uhf_ueff_zero_reproduces_free_particle_determinant():
    K = _chain_hopping()
    nup, ndown = 2, 1
    solution = solve_hubbard_uhf_fixed_ueff(K, nup, ndown, ueff=0.0)
    eigs, eigv = numpy.linalg.eigh(K)
    eigv = _gauge_fix(eigv)

    numpy.testing.assert_allclose(solution["phi_up"], eigv[:, :nup], atol=1e-12, rtol=1e-12)
    numpy.testing.assert_allclose(solution["phi_down"], eigv[:, :ndown], atol=1e-12, rtol=1e-12)
    numpy.testing.assert_allclose(solution["eigenvalues_up"], eigs, atol=1e-12, rtol=1e-12)
    numpy.testing.assert_allclose(solution["eigenvalues_down"], eigs, atol=1e-12, rtol=1e-12)


@pytest.mark.unit
def test_hubbard_uhf_matches_legacy_fixed_ueff_mean_field_iteration():
    K = _chain_hopping()
    nup, ndown = 2, 1
    ueff = 1.2
    mixing = 0.35
    tol = 1e-10
    initial_density = (
        numpy.array([0.65, 0.55, 0.45, 0.35]),
        numpy.array([0.35, 0.30, 0.20, 0.15]),
    )

    solution = solve_hubbard_uhf_fixed_ueff(
        K,
        nup,
        ndown,
        ueff,
        initial_density=initial_density,
        max_iter=200,
        mixing=mixing,
        tol=tol,
    )
    legacy = _legacy_hubbard_uhf_loop(
        K,
        nup,
        ndown,
        ueff,
        initial_density=initial_density,
        max_iter=200,
        mixing=mixing,
        tol=tol,
    )

    assert solution["converged"]
    numpy.testing.assert_allclose(solution["density_up"], legacy["density_up"], atol=1e-10, rtol=1e-10)
    numpy.testing.assert_allclose(
        solution["density_down"], legacy["density_down"], atol=1e-10, rtol=1e-10
    )
    numpy.testing.assert_allclose(solution["rho_up"], legacy["rho_up"], atol=1e-10, rtol=1e-10)
    numpy.testing.assert_allclose(solution["rho_down"], legacy["rho_down"], atol=1e-10, rtol=1e-10)
    numpy.testing.assert_allclose(
        solution["eigenvalues_up"], legacy["eigenvalues_up"], atol=1e-10, rtol=1e-10
    )
    numpy.testing.assert_allclose(
        solution["eigenvalues_down"], legacy["eigenvalues_down"], atol=1e-10, rtol=1e-10
    )


@pytest.mark.unit
def test_hubbard_uhf_density_matrices_are_hermitian_projectors_complex_hopping():
    K = _complex_chain_hopping()
    solution = solve_hubbard_uhf_fixed_ueff(
        K,
        nup=2,
        ndown=1,
        ueff=1.0,
        vup=numpy.array([0.03, -0.02, 0.01, -0.04]),
        vdown=numpy.array([-0.01, 0.04, -0.03, 0.02]),
        mixing=0.4,
    )

    _assert_projector(solution["rho_up"], 2)
    _assert_projector(solution["rho_down"], 1)


@pytest.mark.unit
def test_hubbard_uhf_spin_densities_sum_to_particle_numbers():
    K = _chain_hopping()
    solution = solve_hubbard_uhf_fixed_ueff(K, nup=2, ndown=1, ueff=1.5, mixing=0.4)

    numpy.testing.assert_allclose(numpy.sum(solution["density_up"]), 2.0, atol=1e-10, rtol=0.0)
    numpy.testing.assert_allclose(numpy.sum(solution["density_down"]), 1.0, atol=1e-10, rtol=0.0)


@pytest.mark.unit
def test_hubbard_uhf_is_deterministic_with_fixed_initial_density():
    K = _chain_hopping()
    initial_density = (
        numpy.array([0.65, 0.55, 0.45, 0.35]),
        numpy.array([0.35, 0.30, 0.20, 0.15]),
    )
    sol_a = solve_hubbard_uhf_fixed_ueff(
        K, 2, 1, ueff=1.2, initial_density=initial_density, mixing=0.35
    )
    sol_b = solve_hubbard_uhf_fixed_ueff(
        K, 2, 1, ueff=1.2, initial_density=initial_density, mixing=0.35
    )

    for key in ("phi_up", "phi_down", "density_up", "density_down", "rho_up", "rho_down"):
        numpy.testing.assert_allclose(sol_a[key], sol_b[key], atol=1e-12, rtol=1e-12)
    assert sol_a["converged"] == sol_b["converged"]
    assert sol_a["niter"] == sol_b["niter"]


@pytest.mark.unit
def test_hubbard_uhf_small_chain_converges_with_and_without_pinning():
    K = _chain_hopping()
    no_pinning = solve_hubbard_uhf_fixed_ueff(K, 2, 1, ueff=1.0, mixing=0.4)
    with_pinning = solve_hubbard_uhf_fixed_ueff(
        K,
        2,
        1,
        ueff=1.0,
        vup=numpy.array([0.02, -0.03, 0.01, 0.00]),
        vdown=numpy.array([-0.01, 0.01, -0.02, 0.03]),
        mixing=0.4,
    )

    assert no_pinning["converged"]
    assert with_pinning["converged"]
    assert no_pinning["residual_history"][-1] < 1e-10
    assert with_pinning["residual_history"][-1] < 1e-10


@pytest.mark.unit
def test_build_slater_trial_from_uhf_solution_shape_and_single_det_compatibility():
    K = _chain_hopping()
    nup, ndown = 2, 1
    solution = solve_hubbard_uhf_fixed_ueff(K, nup, ndown, ueff=0.5, mixing=0.4)
    wavefunction = build_slater_trial_from_uhf_solution(solution)

    assert wavefunction.shape == (K.shape[0], nup + ndown)
    numpy.testing.assert_allclose(wavefunction[:, :nup], solution["phi_up"])
    numpy.testing.assert_allclose(wavefunction[:, nup:], solution["phi_down"])

    trial = SingleDet(wavefunction, (nup, ndown), K.shape[0])
    assert trial.psi0a.shape == (K.shape[0], nup)
    assert trial.psi0b.shape == (K.shape[0], ndown)


@pytest.mark.unit
def test_hubbard_uhf_validation_errors_for_bad_shapes():
    K = _chain_hopping()
    with pytest.raises(ValueError, match="vup"):
        solve_hubbard_uhf_fixed_ueff(K, 2, 1, 1.0, vup=numpy.zeros(3))
    with pytest.raises(ValueError, match="vup"):
        solve_hubbard_uhf_fixed_ueff(K, 2, 1, 1.0, vup=numpy.array([0.0, 0.0, 0.0, 1.0j]))
    with pytest.raises(ValueError, match="ueff"):
        solve_hubbard_uhf_fixed_ueff(K, 2, 1, 1.0 + 0.1j)
    with pytest.raises(ValueError, match="initial density_up"):
        solve_hubbard_uhf_fixed_ueff(
            K,
            2,
            1,
            1.0,
            initial_density=(numpy.ones(3), numpy.array([0.25, 0.25, 0.25, 0.25])),
        )
