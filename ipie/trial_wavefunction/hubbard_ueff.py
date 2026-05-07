"""Density-matching helpers for Hubbard ``U_eff`` trial updates."""

import warnings

import numpy
from scipy.optimize import minimize_scalar

from ipie.trial_wavefunction.hubbard_uhf import (
    build_slater_trial_from_uhf_solution,
    solve_hubbard_uhf_fixed_ueff,
)
from ipie.trial_wavefunction.single_det import SingleDet
from ipie.utils.backend import to_host


_RDM_KEYS = ("rho_qmc", "RDMResponse", "rdm_response")


def _as_numpy(array):
    return numpy.asarray(to_host(array))


def _as_1d_real_density(name, density, nsites):
    density = numpy.asarray(density)
    if density.shape != (nsites,):
        raise ValueError(f"{name} must have shape ({nsites},), got {density.shape}.")
    return numpy.asarray(density.real, dtype=numpy.float64)


def _as_particle_number(name, value, nsites):
    if int(value) != value:
        raise ValueError(f"{name} must be an integer.")
    value = int(value)
    if value < 0 or value > nsites:
        raise ValueError(f"{name} must satisfy 0 <= {name} <= n_sites.")
    return value


def _validate_bounds(ueff_bounds):
    bounds = numpy.asarray(ueff_bounds, dtype=numpy.float64)
    if bounds.shape != (2,):
        raise ValueError("ueff_bounds must be a two-item sequence.")
    if not numpy.all(numpy.isfinite(bounds)):
        raise ValueError("ueff_bounds must be finite.")
    if bounds[0] > bounds[1]:
        raise ValueError("ueff_bounds must be ordered as (lower, upper).")
    return float(bounds[0]), float(bounds[1])


def _density_trace(name, density):
    return float(numpy.sum(density).real)


def _warn_if_trace_inconsistent(name, density, target, tol=1e-8):
    trace = _density_trace(name, density)
    if not numpy.isclose(trace, target, atol=tol, rtol=0.0):
        warnings.warn(
            f"{name} trace is {trace:.12e}, expected {target}; "
            "using a fixed-trace copy only for the UHF initial density.",
            RuntimeWarning,
            stacklevel=3,
        )
    return trace


def _fixed_trace_density(density, target):
    density = numpy.asarray(density, dtype=numpy.float64)
    if target == 0:
        return numpy.zeros_like(density)
    trace = _density_trace("density", density)
    if abs(trace) < 1e-14:
        return numpy.full_like(density, target / density.size)
    return density * (target / trace)


def _extract_density_pair(initial_density, nsites):
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
    return (
        _as_1d_real_density("initial_density_up", density_up, nsites),
        _as_1d_real_density("initial_density_down", density_down, nsites),
    )


def _normalize_initial_density(initial_density, n_qmc_up, n_qmc_down, nup, ndown):
    if initial_density is not None:
        nsites = n_qmc_up.shape[0]
        initial_up, initial_down = _extract_density_pair(initial_density, nsites)
        _warn_if_trace_inconsistent("initial_density_up", initial_up, nup)
        _warn_if_trace_inconsistent("initial_density_down", initial_down, ndown)
        return (
            _fixed_trace_density(initial_up, nup),
            _fixed_trace_density(initial_down, ndown),
        )
    return (
        _fixed_trace_density(n_qmc_up, nup),
        _fixed_trace_density(n_qmc_down, ndown),
    )


def _density_delta(density_up, density_down, n_qmc_up, n_qmc_down):
    nsites = n_qmc_up.shape[0]
    return float(
        numpy.sqrt(
            numpy.sum((density_up - n_qmc_up) ** 2 + (density_down - n_qmc_down) ** 2)
            / nsites
        )
    )


def _natural_occupations(rho_qmc):
    occupations = []
    for spin in range(2):
        rho = numpy.asarray(rho_qmc[spin])
        rho_herm = 0.5 * (rho + rho.conj().T)
        occupations.append(numpy.linalg.eigvalsh(rho_herm)[::-1].real)
    return tuple(occupations)


def _shape_from_metadata(source, nbasis):
    if nbasis is not None:
        return (2, int(nbasis), int(nbasis))
    shape = None
    if isinstance(source, dict):
        shape = source.get("shape", None)
    if shape is None:
        shape = getattr(source, "shape", getattr(source, "_shape", None))
    if shape is None:
        return None
    shape = tuple(int(s) for s in shape)
    if len(shape) != 3 or shape[0] != 2 or shape[1] != shape[2]:
        raise ValueError(f"RDM shape metadata must be (2, nbasis, nbasis), got {shape}.")
    return shape


def _reshape_rdm(value, source, nbasis=None):
    value = _as_numpy(value)
    if value.ndim == 3:
        if value.shape[0] != 2 or value.shape[1] != value.shape[2]:
            raise ValueError(f"rho_qmc must have shape (2, nbasis, nbasis), got {value.shape}.")
        if nbasis is not None and value.shape[1] != nbasis:
            raise ValueError(f"rho_qmc nbasis is {value.shape[1]}, expected {nbasis}.")
        return value

    shape = _shape_from_metadata(source, nbasis)
    if shape is None:
        raise ValueError("Flattened RDM input requires nbasis or shape metadata.")
    if value.size != int(numpy.prod(shape)):
        raise ValueError(f"Flattened RDM has size {value.size}, expected {numpy.prod(shape)}.")
    return value.reshape(shape)


def extract_qmc_1rdm(afqmc_result_or_rdm, nbasis=None):
    """Extract a spin-resolved QMC 1RDM from supported in-memory inputs."""
    source = afqmc_result_or_rdm
    if isinstance(source, numpy.ndarray):
        return _reshape_rdm(source, source, nbasis=nbasis)

    if isinstance(source, dict):
        for key in _RDM_KEYS:
            if key in source:
                return _reshape_rdm(source[key], source, nbasis=nbasis)
        raise ValueError(f"Could not find one of {_RDM_KEYS} in QMC RDM dictionary.")

    for key in _RDM_KEYS:
        try:
            value = source[key]
        except (KeyError, RuntimeError, TypeError):
            value = getattr(source, key, None)
        if value is not None:
            return _reshape_rdm(value, source, nbasis=nbasis)

    raise ValueError(
        "Could not extract rho_qmc. Expected an ndarray, a dict with rho_qmc/RDMResponse, "
        "or an estimator-like object exposing RDMResponse."
    )


def _hubbard_particle_numbers(previous_trial):
    if not hasattr(previous_trial, "nalpha") or not hasattr(previous_trial, "nbeta"):
        raise ValueError("previous_trial must expose nalpha and nbeta.")
    return int(previous_trial.nalpha), int(previous_trial.nbeta)


def _hubbard_kinetic_from_system(hubbard_system, explicit_K):
    if explicit_K is not None:
        return _as_numpy(explicit_K)
    if not hasattr(hubbard_system, "T"):
        raise ValueError("K must be supplied when hubbard_system does not expose T.")
    h1 = _as_numpy(hubbard_system.T)
    if h1.shape[0] != 2:
        raise ValueError(f"hubbard_system.T must have spin dimension 2, got shape {h1.shape}.")
    if not numpy.allclose(h1[0], h1[1], atol=1e-12, rtol=1e-12):
        raise ValueError(
            "hubbard_system.T has spin-dependent one-body blocks; supply K, vup, and vdown "
            "explicitly for the U_eff density match."
        )
    return h1[0]


def _default_ueff_bounds(hubbard_system, ueff_bounds):
    if ueff_bounds is not None:
        return ueff_bounds
    if not hasattr(hubbard_system, "U"):
        raise ValueError("ueff_bounds must be supplied when hubbard_system does not expose U.")
    bare_u = numpy.asarray(hubbard_system.U)
    if bare_u.shape != ():
        raise ValueError("hubbard_system.U must be scalar.")
    if abs(bare_u.imag) > 1e-12:
        raise ValueError("hubbard_system.U must be real to use as default U_eff bound.")
    return (0.0, float(bare_u.real))


def fit_ueff_to_qmc_density(
    K,
    nup,
    ndown,
    n_qmc_up,
    n_qmc_down,
    ueff_bounds,
    vup=None,
    vdown=None,
    initial_density=None,
    grid_size=21,
    optimizer="grid_then_brent",
    uhf_options=None,
    verbose=False,
):
    """Fit ``U_eff`` by matching fixed-N UHF site densities to QMC densities."""
    K = _as_numpy(K)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"K must be a square matrix, got shape {K.shape}.")
    nsites = K.shape[0]
    nup = _as_particle_number("nup", nup, nsites)
    ndown = _as_particle_number("ndown", ndown, nsites)
    n_qmc_up = _as_1d_real_density("n_qmc_up", n_qmc_up, nsites)
    n_qmc_down = _as_1d_real_density("n_qmc_down", n_qmc_down, nsites)
    lower, upper = _validate_bounds(ueff_bounds)

    if int(grid_size) != grid_size or int(grid_size) < 1:
        raise ValueError("grid_size must be a positive integer.")
    grid_size = int(grid_size)
    if lower < upper and grid_size < 2:
        raise ValueError("grid_size must be at least 2 when ueff_bounds has nonzero width.")
    if optimizer not in ("grid", "grid_then_brent", None):
        raise ValueError('optimizer must be "grid", "grid_then_brent", or None.')

    qmc_trace_up = _warn_if_trace_inconsistent("n_qmc_up", n_qmc_up, nup)
    qmc_trace_down = _warn_if_trace_inconsistent("n_qmc_down", n_qmc_down, ndown)
    solver_initial_density = _normalize_initial_density(
        initial_density, n_qmc_up, n_qmc_down, nup, ndown
    )

    uhf_options = {} if uhf_options is None else dict(uhf_options)
    uhf_options.setdefault("return_history", True)
    uhf_options.setdefault("verbose", verbose)
    history = []
    cache = {}

    def evaluate(ueff):
        ueff = float(ueff)
        if ueff in cache:
            return cache[ueff]["delta"]
        solution = solve_hubbard_uhf_fixed_ueff(
            K,
            nup,
            ndown,
            ueff,
            vup=vup,
            vdown=vdown,
            initial_density=solver_initial_density,
            **uhf_options,
        )
        delta = _density_delta(
            solution["density_up"], solution["density_down"], n_qmc_up, n_qmc_down
        )
        record = {
            "ueff": ueff,
            "delta": delta,
            "converged": bool(solution["converged"]),
            "niter": int(solution["niter"]),
            "trace_up": _density_trace("density_up", solution["density_up"]),
            "trace_down": _density_trace("density_down", solution["density_down"]),
        }
        history.append(record)
        cache[ueff] = {"delta": delta, "solution": solution, "record": record}
        if verbose:
            print(
                "# U_eff density match: "
                f"U_eff = {ueff:.12e}, delta = {delta:.12e}, "
                f"converged = {solution['converged']}"
            )
        return delta

    grid = numpy.linspace(lower, upper, grid_size) if lower < upper else numpy.array([lower])
    for ueff in grid:
        evaluate(ueff)

    best_record = min(history, key=lambda item: item["delta"])
    best_ueff = best_record["ueff"]
    best_delta = best_record["delta"]

    if optimizer == "grid_then_brent" and lower < upper:
        best_index = int(numpy.argmin([cache[float(ueff)]["delta"] for ueff in grid]))
        if best_index == 0:
            bracket = (float(grid[0]), float(grid[1]))
        elif best_index == grid.size - 1:
            bracket = (float(grid[-2]), float(grid[-1]))
        else:
            bracket = (float(grid[best_index - 1]), float(grid[best_index + 1]))
        result = minimize_scalar(evaluate, bounds=bracket, method="bounded")
        if result.success:
            evaluate(result.x)
            best_record = min(history, key=lambda item: item["delta"])
            best_ueff = best_record["ueff"]
            best_delta = best_record["delta"]
        else:
            warnings.warn(
                f"Bounded U_eff refinement did not converge: {result.message}",
                RuntimeWarning,
                stacklevel=2,
            )

    best_solution = cache[best_ueff]["solution"]
    return {
        "best_ueff": best_ueff,
        "best_solution": best_solution,
        "best_delta": best_delta,
        "history": history,
        "qmc_trace_up": qmc_trace_up,
        "qmc_trace_down": qmc_trace_down,
        "ueff_bounds": (lower, upper),
    }


def update_trial_by_ueff_density_match(
    afqmc_result_or_rdm,
    hubbard_system,
    previous_trial,
    ueff_bounds=None,
    K=None,
    vup=None,
    vdown=None,
    grid_size=21,
    optimizer="grid_then_brent",
    uhf_options=None,
    verbose=False,
):
    """Build a new Hubbard ``SingleDet`` trial by fitting ``U_eff`` to QMC densities."""
    if not hasattr(hubbard_system, "nbasis"):
        raise ValueError("hubbard_system must expose nbasis.")
    nbasis = int(hubbard_system.nbasis)
    rho_qmc = extract_qmc_1rdm(afqmc_result_or_rdm, nbasis=nbasis)
    n_qmc_up = numpy.asarray(numpy.diag(rho_qmc[0]).real, dtype=numpy.float64)
    n_qmc_down = numpy.asarray(numpy.diag(rho_qmc[1]).real, dtype=numpy.float64)

    nup, ndown = _hubbard_particle_numbers(previous_trial)
    K = _hubbard_kinetic_from_system(hubbard_system, K)
    ueff_bounds = _default_ueff_bounds(hubbard_system, ueff_bounds)
    fit = fit_ueff_to_qmc_density(
        K,
        nup,
        ndown,
        n_qmc_up,
        n_qmc_down,
        ueff_bounds,
        vup=vup,
        vdown=vdown,
        grid_size=grid_size,
        optimizer=optimizer,
        uhf_options=uhf_options,
        verbose=verbose,
    )

    solution = fit["best_solution"]
    wavefunction = build_slater_trial_from_uhf_solution(solution)
    trial = SingleDet(
        wavefunction, (nup, ndown), nbasis, verbose=getattr(previous_trial, "verbose", False)
    )
    trial.build()
    trial.half_rotate(hubbard_system)

    n_ip_up = numpy.asarray(solution["density_up"], dtype=numpy.float64)
    n_ip_down = numpy.asarray(solution["density_down"], dtype=numpy.float64)
    natural_up, natural_down = _natural_occupations(rho_qmc)
    diagnostics = {
        "best_ueff": fit["best_ueff"],
        "delta": fit["best_delta"],
        "history": fit["history"],
        "qmc_trace_up": fit["qmc_trace_up"],
        "qmc_trace_down": fit["qmc_trace_down"],
        "ip_trace_up": _density_trace("n_ip_up", n_ip_up),
        "ip_trace_down": _density_trace("n_ip_down", n_ip_down),
        "density_residual_up": n_ip_up - n_qmc_up,
        "density_residual_down": n_ip_down - n_qmc_down,
        "n_ip_up": n_ip_up,
        "n_ip_down": n_ip_down,
        "n_qmc_up": n_qmc_up,
        "n_qmc_down": n_qmc_down,
        "natural_occupations_up": natural_up,
        "natural_occupations_down": natural_down,
        "best_solution": solution,
    }
    return trial, diagnostics
