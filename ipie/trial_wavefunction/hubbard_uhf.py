"""Unrestricted mean-field utilities for Hubbard trial determinants."""

import warnings

import numpy


def _validate_square_matrix(K):
    K = numpy.asarray(K)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"K must be a square matrix, got shape {K.shape}.")
    if not numpy.allclose(K, K.conj().T, atol=1e-12, rtol=1e-12):
        raise ValueError("K must be Hermitian.")
    return K


def _validate_particle_number(name, value, nsites):
    if int(value) != value:
        raise ValueError(f"{name} must be an integer.")
    value = int(value)
    if value < 0 or value > nsites:
        raise ValueError(f"{name} must satisfy 0 <= {name} <= n_sites.")
    return value


def _validate_field(name, field, nsites, dtype):
    if field is None:
        return numpy.zeros(nsites, dtype=dtype)
    field = numpy.asarray(field)
    if field.shape != (nsites,):
        raise ValueError(f"{name} must have shape ({nsites},), got {field.shape}.")
    if numpy.max(numpy.abs(field.imag), initial=0.0) > 1e-12:
        raise ValueError(f"{name} must be real-valued to keep h_sigma Hermitian.")
    return field.real.astype(dtype, copy=False)


def _gauge_fix_orbitals(orbitals):
    if orbitals.shape[1] == 0:
        return orbitals.copy()
    orbitals = numpy.array(orbitals, copy=True)
    max_abs = numpy.argmax(numpy.abs(orbitals), axis=0)
    values = orbitals[max_abs, numpy.arange(orbitals.shape[1])]
    phase = numpy.ones(orbitals.shape[1], dtype=orbitals.dtype)
    nonzero = numpy.abs(values) > 0.0
    phase[nonzero] = values[nonzero] / numpy.abs(values[nonzero])
    return orbitals / phase[numpy.newaxis, :]


def _density_from_orbitals(phi, nsites):
    if phi.shape[1] == 0:
        return numpy.zeros(nsites, dtype=numpy.float64)
    return numpy.sum(numpy.abs(phi) ** 2, axis=1).real


def _one_body_density_matrix(phi, nsites, dtype):
    if phi.shape[1] == 0:
        return numpy.zeros((nsites, nsites), dtype=dtype)
    return phi @ phi.conj().T


def _diagonalize_and_fill(hmat, nocc):
    eigs, eigv = numpy.linalg.eigh(hmat)
    eigv = _gauge_fix_orbitals(eigv)
    return eigs, eigv[:, :nocc]


def _free_particle_initial_density(K, vup, vdown, nup, ndown):
    eigs_up, phi_up = _diagonalize_and_fill(K + numpy.diag(vup), nup)
    eigs_down, phi_down = _diagonalize_and_fill(K + numpy.diag(vdown), ndown)
    return (
        _density_from_orbitals(phi_up, K.shape[0]),
        _density_from_orbitals(phi_down, K.shape[0]),
        eigs_up,
        eigs_down,
    )


def _extract_initial_density(initial_density, nsites, nup, ndown, tol):
    if isinstance(initial_density, dict):
        density_up = initial_density.get("density_up", initial_density.get("up"))
        density_down = initial_density.get("density_down", initial_density.get("down"))
    else:
        try:
            density_up, density_down = initial_density
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "initial_density must be a dict with density_up/density_down "
                "or a two-item sequence."
            ) from exc

    density_up = numpy.asarray(density_up, dtype=numpy.float64)
    density_down = numpy.asarray(density_down, dtype=numpy.float64)
    if density_up.shape != (nsites,):
        raise ValueError(f"initial density_up must have shape ({nsites},).")
    if density_down.shape != (nsites,):
        raise ValueError(f"initial density_down must have shape ({nsites},).")
    _check_density_sum("initial density_up", density_up, nup, tol)
    _check_density_sum("initial density_down", density_down, ndown, tol)
    return density_up.copy(), density_down.copy()


def _check_density_sum(name, density, target, tol):
    atol = max(10.0 * tol, 1e-10)
    total = numpy.sum(density).real
    if not numpy.isclose(total, target, atol=atol, rtol=0.0):
        raise ValueError(f"{name} sums to {total}, expected {target}.")


def _mean_field_energy(K, vup, vdown, ueff, rho_up, rho_down, density_up, density_down):
    h0_up = K + numpy.diag(vup)
    h0_down = K + numpy.diag(vdown)
    e_up = numpy.einsum("ij,ji->", h0_up, rho_up, optimize=True)
    e_down = numpy.einsum("ij,ji->", h0_down, rho_down, optimize=True)
    e_int = ueff * numpy.dot(density_up, density_down)
    return numpy.real_if_close(e_up + e_down + e_int).real


def _occupation_gap(eigs, nocc):
    if nocc == 0 or nocc == eigs.shape[0]:
        return numpy.inf
    return eigs[nocc] - eigs[nocc - 1]


def _warn_if_degenerate(gap, spin, tol):
    gap_tol = max(10.0 * tol, 1e-12)
    if numpy.isfinite(gap) and abs(gap) <= gap_tol:
        warnings.warn(
            f"Tiny {spin}-spin HOMO/LUMO gap ({gap:.6e}) in Hubbard UHF solution; "
            "open-shell degeneracy can make the determinant non-unique.",
            RuntimeWarning,
            stacklevel=3,
        )


def solve_hubbard_uhf_fixed_ueff(
    K,
    nup,
    ndown,
    ueff,
    vup=None,
    vdown=None,
    initial_density=None,
    max_iter=200,
    mixing=0.5,
    tol=1e-10,
    verbose=False,
    return_history=True,
):
    """Solve fixed-N unrestricted Hubbard mean-field equations for fixed ``U_eff``.

    The spin-resolved one-body mean-field Hamiltonians are
    ``h_up = K + diag(vup + ueff * density_down)`` and
    ``h_down = K + diag(vdown + ueff * density_up)``.  The physical Hubbard
    ``U`` used in an AFQMC Hamiltonian is not used here.
    """
    K = _validate_square_matrix(K)
    nsites = K.shape[0]
    nup = _validate_particle_number("nup", nup, nsites)
    ndown = _validate_particle_number("ndown", ndown, nsites)
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1.")
    if not (0.0 < mixing <= 1.0):
        raise ValueError("mixing must satisfy 0 < mixing <= 1.")
    if tol <= 0.0:
        raise ValueError("tol must be positive.")

    ueff_array = numpy.asarray(ueff)
    if ueff_array.shape != ():
        raise ValueError("ueff must be a scalar.")
    if numpy.max(numpy.abs(ueff_array.imag), initial=0.0) > 1e-12:
        raise ValueError("ueff must be real-valued to keep h_sigma Hermitian.")
    dtype = numpy.result_type(
        K.dtype,
        ueff_array.real.dtype,
        numpy.complex128 if numpy.iscomplexobj(K) else numpy.float64,
    )
    K = K.astype(dtype, copy=False)
    vup = _validate_field("vup", vup, nsites, dtype)
    vdown = _validate_field("vdown", vdown, nsites, dtype)
    ueff = float(numpy.real(ueff_array))

    if initial_density is None:
        density_up, density_down, _, _ = _free_particle_initial_density(K, vup, vdown, nup, ndown)
    else:
        density_up, density_down = _extract_initial_density(initial_density, nsites, nup, ndown, tol)

    residual_history = []
    energy_history = []
    converged = False
    niter = 0

    eigs_up = numpy.empty(nsites, dtype=numpy.float64)
    eigs_down = numpy.empty(nsites, dtype=numpy.float64)
    phi_up = numpy.empty((nsites, nup), dtype=dtype)
    phi_down = numpy.empty((nsites, ndown), dtype=dtype)
    density_up_new = density_up.copy()
    density_down_new = density_down.copy()

    for iteration in range(1, max_iter + 1):
        h_up = K + numpy.diag(vup + ueff * density_down)
        h_down = K + numpy.diag(vdown + ueff * density_up)
        eigs_up, phi_up = _diagonalize_and_fill(h_up, nup)
        eigs_down, phi_down = _diagonalize_and_fill(h_down, ndown)
        density_up_new = _density_from_orbitals(phi_up, nsites)
        density_down_new = _density_from_orbitals(phi_down, nsites)
        residual = numpy.sqrt(
            numpy.mean(
                (density_up_new - density_up) ** 2
                + (density_down_new - density_down) ** 2
            )
        )

        rho_up_new = _one_body_density_matrix(phi_up, nsites, dtype)
        rho_down_new = _one_body_density_matrix(phi_down, nsites, dtype)
        energy = _mean_field_energy(
            K, vup, vdown, ueff, rho_up_new, rho_down_new, density_up_new, density_down_new
        )
        residual_history.append(residual)
        energy_history.append(energy)

        if verbose:
            print(f"# UHF iteration {iteration:4d}: residual = {residual:.12e}, energy = {energy:.12e}")

        niter = iteration
        if residual < tol:
            converged = True
            break

        density_up = (1.0 - mixing) * density_up + mixing * density_up_new
        density_down = (1.0 - mixing) * density_down + mixing * density_down_new

    rho_up = _one_body_density_matrix(phi_up, nsites, dtype)
    rho_down = _one_body_density_matrix(phi_down, nsites, dtype)
    _check_density_sum("density_up", density_up_new, nup, tol)
    _check_density_sum("density_down", density_down_new, ndown, tol)

    gap_up = _occupation_gap(eigs_up, nup)
    gap_down = _occupation_gap(eigs_down, ndown)
    _warn_if_degenerate(gap_up, "up", tol)
    _warn_if_degenerate(gap_down, "down", tol)

    return {
        "phi_up": numpy.ascontiguousarray(phi_up),
        "phi_down": numpy.ascontiguousarray(phi_down),
        "density_up": density_up_new,
        "density_down": density_down_new,
        "rho_up": rho_up,
        "rho_down": rho_down,
        "eigenvalues_up": eigs_up,
        "eigenvalues_down": eigs_down,
        "converged": converged,
        "niter": niter,
        "residual_history": numpy.asarray(residual_history) if return_history else None,
        "energy_history": numpy.asarray(energy_history) if return_history else None,
    }


def build_slater_trial_from_uhf_solution(solution):
    """Build a ``SingleDet``-compatible alpha-then-beta determinant matrix."""
    try:
        phi_up = numpy.asarray(solution["phi_up"])
        phi_down = numpy.asarray(solution["phi_down"])
    except KeyError as exc:
        raise ValueError("solution must contain phi_up and phi_down.") from exc

    if phi_up.ndim != 2 or phi_down.ndim != 2:
        raise ValueError("phi_up and phi_down must be rank-2 arrays.")
    if phi_up.shape[0] != phi_down.shape[0]:
        raise ValueError("phi_up and phi_down must have the same number of sites.")
    return numpy.ascontiguousarray(numpy.hstack([phi_up, phi_down]))
