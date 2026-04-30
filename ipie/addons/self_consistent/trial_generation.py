# Copyright 2026 The ipie Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Tuple

import numpy
import scipy.linalg

from ipie.config import MPI
from ipie.systems.generic import Generic
from ipie.trial_wavefunction.single_det import SingleDet
from ipie.utils.backend import to_host


_GREEN_CONVENTIONS = {
    "ipie_default",
    "direct",
    "transpose",
    "one_minus_direct",
    "one_minus_transpose",
}


def green_to_density(G, convention: str):
    """Convert a walker Green's function to a one-body density matrix."""
    if convention not in _GREEN_CONVENTIONS:
        raise ValueError(
            f"Unknown Green's-function convention '{convention}'. "
            f"Expected one of {sorted(_GREEN_CONVENTIONS)}."
        )

    G = numpy.asarray(to_host(G))
    if convention in ("ipie_default", "direct"):
        return G.copy()
    if convention == "transpose":
        return G.T.copy()

    eye = numpy.eye(G.shape[-1], dtype=G.dtype)
    if convention == "one_minus_direct":
        return eye - G
    return eye - G.T


def compute_mixed_1rdm_from_ipie_walkers(
    trial,
    walkers,
    num_elec: Tuple[int, int],
    green_convention: str = "ipie_default",
    mpi_handler=None,
):
    """Compute the zero-temperature mixed 1-RDM from existing ipie walker batches."""
    _validate_zero_temperature_single_det_inputs(trial, walkers, num_elec)

    trial.calc_greens_function(walkers, build_full=True)
    weights = numpy.asarray(to_host(walkers.weight))
    if weights.ndim != 1:
        raise ValueError("walkers.weight must be a one-dimensional walker-weight array.")
    if numpy.allclose(numpy.sum(weights), 0.0):
        raise ValueError("Cannot build a mixed 1-RDM from walkers with zero total weight.")

    _, nbeta = num_elec
    Ga = numpy.asarray(to_host(walkers.Ga))
    if nbeta > 0:
        Gb = numpy.asarray(to_host(walkers.Gb))
    else:
        Gb = numpy.zeros((weights.shape[0], Ga.shape[1], Ga.shape[2]), dtype=Ga.dtype)
    if Ga.shape[0] != weights.shape[0] or Gb.shape[0] != weights.shape[0]:
        raise ValueError("Ga/Gb walker dimensions must match walkers.weight.")

    rho_a_num = numpy.zeros(Ga.shape[1:], dtype=numpy.result_type(Ga, weights, numpy.complex128))
    rho_b_num = numpy.zeros(Gb.shape[1:], dtype=numpy.result_type(Gb, weights, numpy.complex128))
    for iw, weight in enumerate(weights):
        rho_a_num += weight * green_to_density(Ga[iw], green_convention)
        rho_b_num += weight * green_to_density(Gb[iw], green_convention)
    denom = numpy.sum(weights)

    if mpi_handler is not None:
        comm = mpi_handler.comm
        rho_a_num = comm.allreduce(rho_a_num, op=MPI.SUM)
        rho_b_num = comm.allreduce(rho_b_num, op=MPI.SUM)
        denom = comm.allreduce(denom, op=MPI.SUM)

    return rho_a_num / denom, rho_b_num / denom


def natural_orbitals_from_rho(rho, nocc: int, hermitize: bool = True):
    """Return occupied natural orbitals and all occupations sorted high to low."""
    rho = numpy.asarray(rho)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("rho must be a square matrix.")
    if not 0 <= nocc <= rho.shape[0]:
        raise ValueError("nocc must be between 0 and the number of basis functions.")

    if hermitize:
        rho_diag = 0.5 * (rho + rho.conj().T)
        occupations, orbitals = scipy.linalg.eigh(rho_diag, check_finite=False)
    else:
        occupations, orbitals = scipy.linalg.eig(rho, check_finite=False)

    order = numpy.argsort(occupations.real)[::-1]
    occupations = occupations[order]
    orbitals = orbitals[:, order]
    selected = orbitals[:, :nocc]
    if nocc > 0:
        selected, _ = scipy.linalg.qr(selected, mode="economic", check_finite=False)
        selected = _fix_orbital_phases(selected)
    else:
        selected = numpy.empty((rho.shape[0], 0), dtype=orbitals.dtype)

    return selected, occupations


def build_ipie_single_det_trial(
    phi_a,
    phi_b,
    old_trial,
    hamiltonian,
    num_elec: Tuple[int, int],
    mpi_handler=None,
    verbose: bool = False,
):
    """Build a new zero-temperature UHF SingleDet trial from spin orbitals."""
    handler = mpi_handler if mpi_handler is not None else getattr(old_trial, "handler", None)
    kwargs = {"verbose": verbose}
    if handler is not None:
        kwargs["handler"] = handler

    nbasis = _get_nbasis(hamiltonian, old_trial)
    psi = numpy.concatenate([phi_a, phi_b], axis=1)
    new_trial = SingleDet(psi, num_elec, nbasis, **kwargs)

    comm = getattr(mpi_handler, "scomm", None)
    if comm is None and handler is not None:
        comm = getattr(handler, "scomm", None)
    if comm is None:
        new_trial.half_rotate(hamiltonian)
    else:
        new_trial.half_rotate(hamiltonian, comm=comm)
    new_trial.calculate_energy(Generic(nelec=num_elec), hamiltonian)
    return new_trial


def generate_self_consistent_trial(
    hamiltonian,
    initial_trial,
    run_afqmc_once: Callable,
    afqmc_options,
    num_elec: Tuple[int, int],
    max_iter: int = 10,
    rho_tol: float = 1e-3,
    subspace_tol: float = 1e-3,
    hermitize_rho: bool = True,
    green_convention: str = "ipie_default",
    mpi_handler=None,
    verbose: bool = True,
):
    """Generate a zero-temperature self-consistent natural-orbital SingleDet trial."""
    if not isinstance(initial_trial, SingleDet):
        raise TypeError("initial_trial must be a zero-temperature UHF SingleDet trial.")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1.")
    if tuple(num_elec) != tuple(initial_trial.nelec):
        raise ValueError("num_elec must match initial_trial.nelec.")
    if green_convention not in _GREEN_CONVENTIONS:
        raise ValueError(
            f"Unknown Green's-function convention '{green_convention}'. "
            f"Expected one of {sorted(_GREEN_CONVENTIONS)}."
        )

    nalpha, nbeta = num_elec
    nbasis = _get_nbasis(hamiltonian, initial_trial)
    trial = initial_trial
    history = []
    rho_a_old = None
    rho_b_old = None

    for iteration in range(max_iter):
        result = run_afqmc_once(hamiltonian, trial, afqmc_options)
        if not hasattr(result, "walkers"):
            raise AttributeError("run_afqmc_once result must provide a walkers attribute.")

        rho_a, rho_b = compute_mixed_1rdm_from_ipie_walkers(
            trial,
            result.walkers,
            num_elec,
            green_convention=green_convention,
            mpi_handler=mpi_handler,
        )
        phi_a_new, occ_a = natural_orbitals_from_rho(rho_a, nalpha, hermitize=hermitize_rho)
        phi_b_new, occ_b = natural_orbitals_from_rho(rho_b, nbeta, hermitize=hermitize_rho)

        subspace_change = max(
            _subspace_change(trial.psi0a, phi_a_new, nbasis),
            _subspace_change(trial.psi0b, phi_b_new, nbasis),
        )
        if rho_a_old is None:
            rho_change = numpy.inf
        else:
            rho_change = max(_relative_change(rho_a, rho_a_old), _relative_change(rho_b, rho_b_old))

        history_entry = {
            "iteration": iteration,
            "energy_mean": getattr(result, "energy_mean", None),
            "rho_change": rho_change,
            "subspace_change": subspace_change,
            "trace_rho_a": numpy.trace(rho_a),
            "trace_rho_b": numpy.trace(rho_b),
            "idempotency_a": numpy.linalg.norm(rho_a @ rho_a - rho_a),
            "idempotency_b": numpy.linalg.norm(rho_b @ rho_b - rho_b),
            "occupations_a": occ_a,
            "occupations_b": occ_b,
        }
        history.append(history_entry)

        if _should_print(verbose, mpi_handler):
            print(
                "# sc-no iter "
                f"{iteration}: rho_change={rho_change:.6e} "
                f"subspace_change={subspace_change:.6e} "
                f"Tr(rho_a)={numpy.trace(rho_a):.8e} "
                f"Tr(rho_b)={numpy.trace(rho_b):.8e}"
            )

        new_trial = build_ipie_single_det_trial(
            phi_a_new,
            phi_b_new,
            trial,
            hamiltonian,
            num_elec,
            mpi_handler=mpi_handler,
            verbose=verbose,
        )

        converged = rho_change < rho_tol and subspace_change < subspace_tol
        trial = new_trial
        rho_a_old = rho_a
        rho_b_old = rho_b
        if converged:
            break

    return trial, history


def _validate_zero_temperature_single_det_inputs(trial, walkers, num_elec):
    if not isinstance(trial, SingleDet):
        raise TypeError("Only zero-temperature UHF SingleDet trials are supported.")
    required = ("Ga", "weight")
    missing = [name for name in required if not hasattr(walkers, name)]
    if missing:
        raise TypeError(f"Zero-temperature UHF walkers must provide {', '.join(missing)}.")
    if num_elec[1] > 0 and not hasattr(walkers, "Gb"):
        raise TypeError("Zero-temperature UHF walkers with beta electrons must provide Gb.")
    if getattr(walkers, "stack", None) is not None:
        raise TypeError("Thermal walkers are not supported by this zero-temperature utility.")
    if tuple(num_elec) != tuple(trial.nelec):
        raise ValueError("num_elec must match trial.nelec.")


def _get_nbasis(hamiltonian, trial):
    nbasis = getattr(hamiltonian, "nbasis", getattr(trial, "nbasis", None))
    if nbasis is None:
        raise ValueError("Could not determine nbasis from hamiltonian or trial.")
    return nbasis


def _fix_orbital_phases(orbitals):
    orbitals = orbitals.copy()
    for iorb in range(orbitals.shape[1]):
        col = orbitals[:, iorb]
        pivot = numpy.argmax(numpy.abs(col))
        if numpy.abs(col[pivot]) > 0.0:
            if numpy.isrealobj(orbitals):
                orbitals[:, iorb] *= numpy.sign(col[pivot].real)
            else:
                orbitals[:, iorb] *= numpy.exp(-1.0j * numpy.angle(col[pivot]))
            if orbitals[pivot, iorb].real < 0.0:
                orbitals[:, iorb] *= -1.0
    if numpy.allclose(orbitals.imag, 0.0):
        orbitals = orbitals.real
    return numpy.ascontiguousarray(orbitals)


def _subspace_change(phi_old, phi_new, nbasis):
    if phi_old.shape[1] == 0 and phi_new.shape[1] == 0:
        return 0.0
    p_old = phi_old @ phi_old.conj().T
    p_new = phi_new @ phi_new.conj().T
    return numpy.linalg.norm(p_new - p_old) / numpy.sqrt(nbasis)


def _relative_change(new, old):
    return numpy.linalg.norm(new - old) / max(numpy.linalg.norm(old), 1.0e-12)


def _should_print(verbose, mpi_handler):
    if not verbose:
        return False
    if mpi_handler is None:
        return True
    return getattr(mpi_handler, "rank", 0) == 0
