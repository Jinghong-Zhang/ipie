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

import copy
from typing import Callable, Tuple

import numpy
import scipy.linalg

from ipie.config import MPI
from ipie.estimators.estimator_base import EstimatorBase
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
    rho_a_num, rho_b_num, denom = mixed_1rdm_numerators_from_ipie_walkers(
        trial,
        walkers,
        num_elec,
        green_convention=green_convention,
    )

    if mpi_handler is not None:
        comm = mpi_handler.comm
        rho_a_num = comm.allreduce(rho_a_num, op=MPI.SUM)
        rho_b_num = comm.allreduce(rho_b_num, op=MPI.SUM)
        denom = comm.allreduce(denom, op=MPI.SUM)

    if numpy.allclose(denom, 0.0):
        raise ValueError("Cannot build a mixed 1-RDM from walkers with zero total weight.")

    return rho_a_num / denom, rho_b_num / denom


def mixed_1rdm_numerators_from_ipie_walkers(
    trial,
    walkers,
    num_elec: Tuple[int, int],
    green_convention: str = "ipie_default",
):
    """Return unnormalised mixed 1-RDM numerators and total walker weight."""
    _validate_zero_temperature_single_det_inputs(trial, walkers, num_elec)

    trial.calc_greens_function(walkers, build_full=True)
    weights = numpy.asarray(to_host(walkers.weight))
    if weights.ndim != 1:
        raise ValueError("walkers.weight must be a one-dimensional walker-weight array.")
    if not numpy.all(numpy.isfinite(weights.real)) or not numpy.all(numpy.isfinite(weights.imag)):
        raise ValueError("Cannot build a mixed 1-RDM from non-finite walker weights.")

    _, nbeta = num_elec
    Ga = numpy.asarray(to_host(walkers.Ga))
    if nbeta > 0:
        Gb = numpy.asarray(to_host(walkers.Gb))
    else:
        Gb = numpy.zeros((weights.shape[0], Ga.shape[1], Ga.shape[2]), dtype=Ga.dtype)
    if Ga.shape[0] != weights.shape[0] or Gb.shape[0] != weights.shape[0]:
        raise ValueError("Ga/Gb walker dimensions must match walkers.weight.")
    if not numpy.all(numpy.isfinite(Ga.real)) or not numpy.all(numpy.isfinite(Ga.imag)):
        raise ValueError("Cannot build a mixed 1-RDM from non-finite alpha Green's functions.")
    if not numpy.all(numpy.isfinite(Gb.real)) or not numpy.all(numpy.isfinite(Gb.imag)):
        raise ValueError("Cannot build a mixed 1-RDM from non-finite beta Green's functions.")

    rho_a_num = numpy.zeros(Ga.shape[1:], dtype=numpy.result_type(Ga, weights, numpy.complex128))
    rho_b_num = numpy.zeros(Gb.shape[1:], dtype=numpy.result_type(Gb, weights, numpy.complex128))
    for iw, weight in enumerate(weights):
        rho_a_num += weight * green_to_density(Ga[iw], green_convention)
        rho_b_num += weight * green_to_density(Gb[iw], green_convention)
    denom = numpy.sum(weights)
    return rho_a_num, rho_b_num, denom


class MixedOneRDMAccumulator:
    """Accumulate a block/time average of the mixed zero-temperature 1-RDM."""

    def __init__(
        self,
        num_elec: Tuple[int, int],
        green_convention: str = "ipie_default",
        mpi_handler=None,
        collect_sample_stats: bool = False,
    ):
        if green_convention not in _GREEN_CONVENTIONS:
            raise ValueError(
                f"Unknown Green's-function convention '{green_convention}'. "
                f"Expected one of {sorted(_GREEN_CONVENTIONS)}."
            )
        self.num_elec = tuple(num_elec)
        self.green_convention = green_convention
        self.mpi_handler = mpi_handler
        self.collect_sample_stats = collect_sample_stats
        self.rho_a_num = None
        self.rho_b_num = None
        self.denom = 0.0j
        self.num_samples = 0
        self.rho_a_sample_sum = None
        self.rho_b_sample_sum = None
        self.rho_a_sample_abs2_sum = None
        self.rho_b_sample_abs2_sum = None

    def update(self, trial, walkers):
        rho_a_num, rho_b_num, denom = mixed_1rdm_numerators_from_ipie_walkers(
            trial,
            walkers,
            self.num_elec,
            green_convention=self.green_convention,
        )
        if self.rho_a_num is None:
            self.rho_a_num = numpy.zeros_like(rho_a_num, dtype=numpy.result_type(rho_a_num, denom))
            self.rho_b_num = numpy.zeros_like(rho_b_num, dtype=numpy.result_type(rho_b_num, denom))
        self.rho_a_num += rho_a_num
        self.rho_b_num += rho_b_num
        self.denom += denom
        self.num_samples += 1
        if self.collect_sample_stats:
            self._update_sample_stats(rho_a_num, rho_b_num, denom)

    def finalize(self):
        if self.num_samples == 0:
            raise ValueError("Cannot build a mixed 1-RDM before accumulating any samples.")
        rho_a_num = self.rho_a_num
        rho_b_num = self.rho_b_num
        denom = self.denom
        if self.mpi_handler is not None:
            comm = self.mpi_handler.comm
            rho_a_num = comm.allreduce(rho_a_num, op=MPI.SUM)
            rho_b_num = comm.allreduce(rho_b_num, op=MPI.SUM)
            denom = comm.allreduce(denom, op=MPI.SUM)
        if numpy.allclose(denom, 0.0):
            raise ValueError("Cannot build a mixed 1-RDM from accumulated zero total weight.")
        return rho_a_num / denom, rho_b_num / denom

    def sample_standard_error(self):
        """Return uncorrelated per-entry SEM estimates from block-normalised samples."""
        if not self.collect_sample_stats or self.num_samples < 2:
            return None, None
        mean_a = self.rho_a_sample_sum / self.num_samples
        mean_b = self.rho_b_sample_sum / self.num_samples
        var_a = (
            self.rho_a_sample_abs2_sum - self.num_samples * numpy.abs(mean_a) ** 2
        ) / (self.num_samples - 1)
        var_b = (
            self.rho_b_sample_abs2_sum - self.num_samples * numpy.abs(mean_b) ** 2
        ) / (self.num_samples - 1)
        var_a = numpy.maximum(var_a.real, 0.0)
        var_b = numpy.maximum(var_b.real, 0.0)
        return numpy.sqrt(var_a / self.num_samples), numpy.sqrt(var_b / self.num_samples)

    def sample_standard_deviation(self):
        """Return per-entry standard deviations from block-normalised samples."""
        if not self.collect_sample_stats or self.num_samples < 2:
            return None, None
        mean_a = self.rho_a_sample_sum / self.num_samples
        mean_b = self.rho_b_sample_sum / self.num_samples
        var_a = (
            self.rho_a_sample_abs2_sum - self.num_samples * numpy.abs(mean_a) ** 2
        ) / (self.num_samples - 1)
        var_b = (
            self.rho_b_sample_abs2_sum - self.num_samples * numpy.abs(mean_b) ** 2
        ) / (self.num_samples - 1)
        var_a = numpy.maximum(var_a.real, 0.0)
        var_b = numpy.maximum(var_b.real, 0.0)
        return numpy.sqrt(var_a), numpy.sqrt(var_b)

    def _update_sample_stats(self, rho_a_num, rho_b_num, denom):
        if self.mpi_handler is not None:
            comm = self.mpi_handler.comm
            rho_a_num = comm.allreduce(rho_a_num, op=MPI.SUM)
            rho_b_num = comm.allreduce(rho_b_num, op=MPI.SUM)
            denom = comm.allreduce(denom, op=MPI.SUM)
        if numpy.allclose(denom, 0.0):
            return
        rho_a_sample = rho_a_num / denom
        rho_b_sample = rho_b_num / denom
        if self.rho_a_sample_sum is None:
            self.rho_a_sample_sum = numpy.zeros_like(rho_a_sample)
            self.rho_b_sample_sum = numpy.zeros_like(rho_b_sample)
            self.rho_a_sample_abs2_sum = numpy.zeros_like(rho_a_sample.real)
            self.rho_b_sample_abs2_sum = numpy.zeros_like(rho_b_sample.real)
        self.rho_a_sample_sum += rho_a_sample
        self.rho_b_sample_sum += rho_b_sample
        self.rho_a_sample_abs2_sum += numpy.abs(rho_a_sample) ** 2
        self.rho_b_sample_abs2_sum += numpy.abs(rho_b_sample) ** 2


class MixedOneRDMEstimator(EstimatorBase):
    """Mixed zero-temperature 1-RDM estimator written to the normal ipie stream.

    The estimator stores block numerators and denominators, not already-averaged
    matrices. Downstream analysis should form a block time series from
    RhoNumer/RhoDenom, then reblock that series just as for ETotal.
    """

    def __init__(
        self,
        num_elec: Tuple[int, int],
        nbasis: int,
        green_convention: str = "ipie_default",
    ):
        super().__init__()
        if green_convention not in _GREEN_CONVENTIONS:
            raise ValueError(
                f"Unknown Green's-function convention '{green_convention}'. "
                f"Expected one of {sorted(_GREEN_CONVENTIONS)}."
            )
        self.num_elec = tuple(num_elec)
        self.nbasis = int(nbasis)
        self.green_convention = green_convention
        self.scalar_estimator = False
        matrix_size = self.nbasis * self.nbasis
        self._data = {
            "RhoANumer": numpy.zeros(matrix_size, dtype=numpy.complex128),
            "RhoBNumer": numpy.zeros(matrix_size, dtype=numpy.complex128),
            "RhoDenom": numpy.zeros(1, dtype=numpy.complex128),
        }
        self._shape = (2 * matrix_size + 1,)
        self.print_to_stdout = False

    @property
    def data(self):
        return numpy.concatenate(
            [numpy.asarray(value).reshape(-1) for value in self._data.values()]
        )

    def compute_estimator(self, system=None, walkers=None, hamiltonian=None, trial=None):
        rho_a_num, rho_b_num, denom = mixed_1rdm_numerators_from_ipie_walkers(
            trial,
            walkers,
            self.num_elec,
            green_convention=self.green_convention,
        )
        expected_shape = (self.nbasis, self.nbasis)
        if rho_a_num.shape != expected_shape or rho_b_num.shape != expected_shape:
            raise ValueError(
                "Mixed 1-RDM estimator shape mismatch: "
                f"expected {expected_shape}, got {rho_a_num.shape} and {rho_b_num.shape}."
            )
        self._data["RhoANumer"][:] = rho_a_num.reshape(-1)
        self._data["RhoBNumer"][:] = rho_b_num.reshape(-1)
        self._data["RhoDenom"][0] = denom
        return self.data


class MixedOneRDMElementEstimator(EstimatorBase):
    """Scalar mixed 1-RDM element estimator for block-by-block stdout diagnostics."""

    def __init__(
        self,
        num_elec: Tuple[int, int],
        element: Tuple[int, int] = (0, 0),
        green_convention: str = "ipie_default",
    ):
        super().__init__()
        if green_convention not in _GREEN_CONVENTIONS:
            raise ValueError(
                f"Unknown Green's-function convention '{green_convention}'. "
                f"Expected one of {sorted(_GREEN_CONVENTIONS)}."
            )
        self.num_elec = tuple(num_elec)
        self.element = tuple(element)
        if len(self.element) != 2:
            raise ValueError("1-RDM element must be a pair of indices.")
        self.green_convention = green_convention
        self.scalar_estimator = True
        i, j = self.element
        self._data = {
            f"RhoA{i}{j}Numer": 0.0j,
            f"RhoB{i}{j}Numer": 0.0j,
            "RhoDenom": 0.0j,
            f"RhoA{i}{j}": 0.0j,
            f"RhoB{i}{j}": 0.0j,
        }
        self._shape = (len(self.names),)
        self._data_index = {key: idx for idx, key in enumerate(list(self._data.keys()))}
        self.print_to_stdout = True

    def compute_estimator(self, system=None, walkers=None, hamiltonian=None, trial=None):
        rho_a_num, rho_b_num, denom = mixed_1rdm_numerators_from_ipie_walkers(
            trial,
            walkers,
            self.num_elec,
            green_convention=self.green_convention,
        )
        i, j = self.element
        if not (0 <= i < rho_a_num.shape[0] and 0 <= j < rho_a_num.shape[1]):
            raise ValueError(
                f"1-RDM element {self.element} is outside matrix shape {rho_a_num.shape}."
            )
        self._data[f"RhoA{i}{j}Numer"] = rho_a_num[i, j]
        self._data[f"RhoB{i}{j}Numer"] = rho_b_num[i, j]
        self._data["RhoDenom"] = denom
        self._data[f"RhoA{i}{j}"] = 0.0j
        self._data[f"RhoB{i}{j}"] = 0.0j
        return self.data

    def post_reduce_hook(self, data):
        i, j = self.element
        denom = data[self._data_index["RhoDenom"]]
        data[self._data_index[f"RhoA{i}{j}"]] = (
            data[self._data_index[f"RhoA{i}{j}Numer"]] / denom
        )
        data[self._data_index[f"RhoB{i}{j}"]] = (
            data[self._data_index[f"RhoB{i}{j}Numer"]] / denom
        )


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
    phi_a = _as_host_array(phi_a)
    phi_b = _as_host_array(phi_b)
    psi = numpy.concatenate([phi_a, phi_b], axis=1)
    new_trial = SingleDet(psi, num_elec, nbasis, **kwargs)
    hamiltonian_host = _host_hamiltonian_copy(hamiltonian)

    comm = getattr(mpi_handler, "scomm", None)
    if comm is None and handler is not None:
        comm = getattr(handler, "scomm", None)
    if comm is None:
        new_trial.half_rotate(hamiltonian_host)
    else:
        new_trial.half_rotate(hamiltonian_host, comm=comm)
    new_trial.calculate_energy(Generic(nelec=num_elec), hamiltonian_host)
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
    diagnostic_callback: Callable | None = None,
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
        result_rho_a = getattr(result, "rho_a", None)
        result_rho_b = getattr(result, "rho_b", None)
        if result_rho_a is not None and result_rho_b is not None:
            rho_a = numpy.asarray(to_host(result_rho_a))
            rho_b = numpy.asarray(to_host(result_rho_b))
        else:
            if not hasattr(result, "walkers"):
                raise AttributeError(
                    "run_afqmc_once result must provide either rho_a/rho_b or a walkers attribute."
                )
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
            "energy_sem": getattr(result, "energy_sem", None),
            "energy_block10_sem": getattr(result, "energy_block10_sem", None),
            "energy_num_blocks": getattr(result, "energy_num_blocks", None),
            "rho_change": rho_change,
            "subspace_change": subspace_change,
            "natural_orbital_diagonalization": "hermitian" if hermitize_rho else "raw",
            "trace_rho_a": numpy.trace(rho_a),
            "trace_rho_b": numpy.trace(rho_b),
            "antihermiticity_a": numpy.linalg.norm(rho_a - rho_a.conj().T),
            "antihermiticity_b": numpy.linalg.norm(rho_b - rho_b.conj().T),
            "idempotency_a": numpy.linalg.norm(rho_a @ rho_a - rho_a),
            "idempotency_b": numpy.linalg.norm(rho_b @ rho_b - rho_b),
            "raw_eigenvalues_a": _sorted_eigvals(rho_a),
            "raw_eigenvalues_b": _sorted_eigvals(rho_b),
            "hermitian_eigenvalues_a": _sorted_hermitian_eigvals(rho_a),
            "hermitian_eigenvalues_b": _sorted_hermitian_eigvals(rho_b),
            "occupations_a": occ_a,
            "occupations_b": occ_b,
            "rho_num_samples": getattr(result, "rho_num_samples", None),
            "rho_a_sem_abs_max": _array_abs_max_or_none(getattr(result, "rho_a_sem", None)),
            "rho_b_sem_abs_max": _array_abs_max_or_none(getattr(result, "rho_b_sem", None)),
            "rho_a_sem_fro": _array_norm_or_none(getattr(result, "rho_a_sem", None)),
            "rho_b_sem_fro": _array_norm_or_none(getattr(result, "rho_b_sem", None)),
            "rho_mean_kind": getattr(result, "rho_mean_kind", None),
            "rho_block_weight_diff_max_a": getattr(
                result, "rho_block_weight_diff_max_a", None
            ),
            "rho_block_weight_diff_max_b": getattr(
                result, "rho_block_weight_diff_max_b", None
            ),
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

        if diagnostic_callback is not None:
            diagnostic_callback(
                iteration=iteration,
                trial=trial,
                new_trial=new_trial,
                result=result,
                rho_a=rho_a,
                rho_b=rho_b,
                occ_a=occ_a,
                occ_b=occ_b,
                history_entry=history_entry,
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


def _as_host_array(value):
    if isinstance(value, numpy.ndarray):
        return numpy.asarray(value)
    if hasattr(value, "__cuda_array_interface__"):
        return numpy.asarray(to_host(value))
    return numpy.asarray(value)


def _host_value(value):
    if isinstance(value, numpy.ndarray):
        return numpy.asarray(value)
    if hasattr(value, "__cuda_array_interface__"):
        return numpy.asarray(to_host(value))
    if isinstance(value, list):
        return [_host_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_host_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _host_value(item) for key, item in value.items()}
    return value


def _host_hamiltonian_copy(hamiltonian):
    hamiltonian_host = copy.copy(hamiltonian)
    for key, value in hamiltonian.__dict__.items():
        hamiltonian_host.__dict__[key] = _host_value(value)
    return hamiltonian_host


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
    phi_old = numpy.asarray(to_host(phi_old))
    phi_new = numpy.asarray(to_host(phi_new))
    if phi_old.shape[1] == 0 and phi_new.shape[1] == 0:
        return 0.0
    p_old = phi_old @ phi_old.conj().T
    p_new = phi_new @ phi_new.conj().T
    return numpy.linalg.norm(p_new - p_old) / numpy.sqrt(nbasis)


def _relative_change(new, old):
    new = numpy.asarray(to_host(new))
    old = numpy.asarray(to_host(old))
    return numpy.linalg.norm(new - old) / max(numpy.linalg.norm(old), 1.0e-12)


def _array_abs_max_or_none(value):
    if value is None:
        return None
    arr = numpy.asarray(to_host(value))
    if arr.size == 0:
        return 0.0
    return numpy.max(numpy.abs(arr))


def _array_norm_or_none(value):
    if value is None:
        return None
    return numpy.linalg.norm(numpy.asarray(to_host(value)))


def _sorted_eigvals(value):
    vals = scipy.linalg.eigvals(numpy.asarray(to_host(value)), check_finite=False)
    order = numpy.argsort(vals.real)[::-1]
    return vals[order]


def _sorted_hermitian_eigvals(value):
    rho = numpy.asarray(to_host(value))
    vals = scipy.linalg.eigvalsh(0.5 * (rho + rho.conj().T), check_finite=False)
    return vals[numpy.argsort(vals)[::-1]]


def _should_print(verbose, mpi_handler):
    if not verbose:
        return False
    if mpi_handler is None:
        return True
    return getattr(mpi_handler, "rank", 0) == 0
