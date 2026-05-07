"""JAX reverse-AD response estimators for the single-site Hubbard propagator."""

import numpy

from ipie.estimators.estimator_base import EstimatorBase
from ipie.hamiltonians.hubbard import Hubbard
from ipie.propagation.hirsch_base import construct_hirsch_auxiliaries
from ipie.utils.backend import arraylib as xp


class HubbardJAX1RDMResponseEstimator(EstimatorBase):
    """Spin-resolved 1RDM response estimator for the JAX Hubbard propagator.

    The estimator differentiates a fixed-field response block with respect to a
    real spin-resolved perturbation added to ``hamiltonian.T``.  Setting
    ``trial_response="jax_uhf"`` also differentiates through a fixed-iteration
    JAX port of the legacy Hubbard UHF mean-field equations.
    """

    def __init__(
        self,
        ham=None,
        hamiltonian=None,
        trial=None,
        num_steps=None,
        timestep=None,
        spin_decomp=True,
        trial_response=False,
        include_weight_gradient=True,
        uhf_mixing=0.5,
        uhf_n_scf=50,
        uhf_ueff=None,
        random_fields=None,
        walker_batch_size=None,
        filename=None,
    ):
        super().__init__()
        hamiltonian = hamiltonian if hamiltonian is not None else ham
        if trial_response is True:
            trial_response = "jax_uhf"
        if hamiltonian is None or not isinstance(hamiltonian, Hubbard):
            raise ValueError("HubbardJAX1RDMResponseEstimator requires a Hubbard hamiltonian.")
        if num_steps is None:
            raise ValueError("num_steps is required for HubbardJAX1RDMResponseEstimator.")
        if timestep is None:
            raise ValueError("timestep is required for HubbardJAX1RDMResponseEstimator.")
        if trial_response not in (False, "jax_uhf"):
            raise ValueError('trial_response must be False or "jax_uhf".')

        self.nbasis = hamiltonian.nbasis
        self.num_steps = int(num_steps)
        self.timestep = float(timestep)
        self.spin_decomp = spin_decomp
        self.trial_response = trial_response
        self.include_weight_gradient = include_weight_gradient
        self.uhf_mixing = float(uhf_mixing)
        self.uhf_n_scf = int(uhf_n_scf)
        self.uhf_ueff = float(hamiltonian.U.real if uhf_ueff is None else uhf_ueff)
        self.random_fields = random_fields
        self.walker_batch_size = None if walker_batch_size is None else int(walker_batch_size)
        if self.walker_batch_size is not None and self.walker_batch_size < 1:
            raise ValueError("walker_batch_size must be positive.")

        response_size = 2 * self.nbasis * self.nbasis
        self._data = {
            "ENumer": numpy.zeros(1, dtype=numpy.complex128),
            "EDenom": numpy.zeros(1, dtype=numpy.complex128),
            "RDMResponse": numpy.zeros(response_size, dtype=numpy.complex128),
            "WeightResponse": numpy.zeros(response_size, dtype=numpy.complex128),
            "dENumer": numpy.zeros(response_size, dtype=numpy.complex128),
            "dEDenom": numpy.zeros(response_size, dtype=numpy.complex128),
        }
        if not self.include_weight_gradient:
            self._data.pop("WeightResponse")

        self._shape = (2, self.nbasis, self.nbasis)
        self.scalar_estimator = False
        self.print_to_stdout = False
        self.ascii_filename = None

        self._data_index = {}
        offset = 0
        for key, value in self._data.items():
            self._data_index[key] = offset
            offset += int(numpy.prod(value.shape))

    @property
    def data(self):
        return numpy.concatenate([numpy.asarray(value).ravel() for value in self._data.values()])

    def get_index(self, name):
        index = self._data_index.get(name, None)
        if index is None:
            raise RuntimeError(f"Unknown estimator {name}")
        return index

    def _slice(self, name):
        start = self.get_index(name)
        size = int(numpy.prod(self._data[name].shape))
        return slice(start, start + size)

    def _draw_random_fields(self, walkers):
        shape = (self.num_steps, self.nbasis, walkers.nwalkers)
        if self.random_fields is not None:
            fields = self.random_fields
            if fields.shape != shape:
                raise ValueError(f"random_fields must have shape {shape}, got {fields.shape}.")
            return fields
        if hasattr(xp, "RawKernel"):
            return xp.random.random(shape, dtype=xp.float64)
        return numpy.random.random(shape)

    def _dummy_phib(self, walkers):
        return xp.zeros((walkers.nwalkers, walkers.nbasis, 0), dtype=xp.complex128)

    def _walker_view(self, walkers, start, stop):
        class WalkerView:
            pass

        view = WalkerView()
        view.nwalkers = stop - start
        view.nup = walkers.nup
        view.ndown = walkers.ndown
        view.nbasis = walkers.nbasis
        view.rhf = walkers.rhf
        view.phia = walkers.phia[start:stop]
        view.phib = None if walkers.phib is None else walkers.phib[start:stop]
        view.weight = walkers.weight[start:stop]
        view.log_shift = walkers.log_shift[start:stop]
        return view

    def _compute_raw_response_chunk(self, walkers, hamiltonian, trial, random_fields):
        from ipie.propagation.hubbard_jax import (
            as_jax_array,
            block_until_ready,
            hubbard_response_raw,
        )

        if not isinstance(hamiltonian, Hubbard):
            raise ValueError("HubbardJAX1RDMResponseEstimator only supports Hubbard.")

        _, _, aux_wfac, delta = construct_hirsch_auxiliaries(
            hamiltonian, self.timestep, spin_decomp=self.spin_decomp
        )
        phib = walkers.phib if walkers.phib is not None else self._dummy_phib(walkers)
        psi0b = trial.psi0b
        if psi0b.shape[1] == 0:
            psi0b = numpy.zeros((hamiltonian.nbasis, 0), dtype=numpy.complex128)

        coupling = numpy.zeros((2, hamiltonian.nbasis, hamiltonian.nbasis), dtype=numpy.float64)
        result = hubbard_response_raw(
            as_jax_array(coupling, dtype="float64", force_host=True),
            as_jax_array(walkers.phia, dtype="complex128", force_host=True),
            as_jax_array(phib, dtype="complex128", force_host=True),
            as_jax_array(walkers.weight, dtype="float64", force_host=True),
            as_jax_array(walkers.log_shift, dtype="float64", force_host=True),
            as_jax_array(trial.psi0a, dtype="complex128", force_host=True),
            as_jax_array(psi0b, dtype="complex128", force_host=True),
            as_jax_array(hamiltonian.T, dtype="complex128", force_host=True),
            as_jax_array(hamiltonian.U, dtype="complex128", force_host=True),
            as_jax_array(hamiltonian.ecore, dtype="complex128", force_host=True),
            as_jax_array(delta, dtype="complex128", force_host=True),
            as_jax_array(aux_wfac, dtype="complex128", force_host=True),
            as_jax_array(random_fields, dtype="float64", force_host=True),
            self.timestep,
            walkers.nup,
            walkers.ndown,
            rhf=bool(walkers.rhf),
            trial_response=self.trial_response,
            uhf_n_scf=self.uhf_n_scf,
            uhf_mixing=self.uhf_mixing,
            uhf_ueff=self.uhf_ueff,
        )
        result = block_until_ready(result)
        enumer, edenom, d_enumer, d_edenom = result
        return {
            "ENumer": numpy.asarray(enumer, dtype=numpy.complex128).reshape(1),
            "EDenom": numpy.asarray(edenom, dtype=numpy.complex128).reshape(1),
            "dENumer": numpy.asarray(d_enumer, dtype=numpy.complex128).ravel(),
            "dEDenom": numpy.asarray(d_edenom, dtype=numpy.complex128).ravel(),
        }

    def _compute_raw_response(self, walkers, hamiltonian, trial, random_fields=None):
        if random_fields is None:
            random_fields = self._draw_random_fields(walkers)
        if self.walker_batch_size is None or self.walker_batch_size >= walkers.nwalkers:
            return self._compute_raw_response_chunk(walkers, hamiltonian, trial, random_fields)

        total = None
        start = 0
        while start < walkers.nwalkers:
            stop = min(walkers.nwalkers, start + self.walker_batch_size)
            walker_view = self._walker_view(walkers, start, stop)
            fields_view = xp.ascontiguousarray(random_fields[:, :, start:stop])
            raw = self._compute_raw_response_chunk(
                walker_view, hamiltonian, trial, random_fields=fields_view
            )
            if total is None:
                total = {key: value.copy() for key, value in raw.items()}
            else:
                for key, value in raw.items():
                    total[key] += value
            start = stop
        return total

    def _ratio_gradient(self, enumer, edenom, d_enumer, d_edenom):
        if abs(edenom) == 0.0:
            return numpy.zeros_like(d_enumer)
        energy = enumer / edenom
        return d_enumer / edenom - energy * d_edenom / edenom

    def compute_estimator(self, system=None, walkers=None, hamiltonian=None, trial=None):
        if walkers is None:
            raise ValueError("Walkers cannot be None in HubbardJAX1RDMResponseEstimator.")
        raw = self._compute_raw_response(walkers, hamiltonian, trial)
        enumer = raw["ENumer"][0]
        edenom = raw["EDenom"][0]
        d_enumer = raw["dENumer"]
        d_edenom = raw["dEDenom"]

        self._data["ENumer"][:] = raw["ENumer"]
        self._data["EDenom"][:] = raw["EDenom"]
        self._data["RDMResponse"][:] = self._ratio_gradient(enumer, edenom, d_enumer, d_edenom)
        self._data["dENumer"][:] = d_enumer
        self._data["dEDenom"][:] = d_edenom
        if self.include_weight_gradient:
            self._data["WeightResponse"][:] = d_edenom
        return self.data

    def post_reduce_hook(self, data):
        enumer = data[self._slice("ENumer")][0]
        edenom = data[self._slice("EDenom")][0]
        d_enumer = data[self._slice("dENumer")]
        d_edenom = data[self._slice("dEDenom")]
        data[self._slice("RDMResponse")] = self._ratio_gradient(
            enumer, edenom, d_enumer, d_edenom
        )
        if self.include_weight_gradient:
            data[self._slice("WeightResponse")] = d_edenom
