"""AD-block Hubbard response driver helpers for the JAX single-site path."""

from __future__ import annotations

from dataclasses import dataclass

import numpy

from ipie.hamiltonians.hubbard import Hubbard
from ipie.propagation.hirsch_base import construct_hirsch_auxiliaries
from ipie.utils.backend import arraylib as xp


def _random_uniform(shape):
    if hasattr(xp, "RawKernel"):
        return xp.random.random(shape, dtype=xp.float64)
    return numpy.random.random(shape)


def _dummy_phib(walkers):
    return xp.zeros((walkers.nwalkers, walkers.nbasis, 0), dtype=xp.complex128)


def _dummy_invb(walkers):
    return xp.zeros((walkers.nwalkers, 0, 0), dtype=xp.complex128)


def _walker_view(walkers, start, stop):
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
    view.inv_ovlp_a = walkers.inv_ovlp_a[start:stop]
    view.inv_ovlp_b = None if walkers.inv_ovlp_b is None else walkers.inv_ovlp_b[start:stop]
    view.weight = walkers.weight[start:stop]
    view.ovlp = walkers.ovlp[start:stop]
    view.log_shift = walkers.log_shift[start:stop]
    return view


def _zero_raw(nbasis):
    response_size = 2 * nbasis * nbasis
    return {
        "ENumer": numpy.zeros(1, dtype=numpy.complex128),
        "EDenom": numpy.zeros(1, dtype=numpy.complex128),
        "dENumer": numpy.zeros(response_size, dtype=numpy.complex128),
        "dEDenom": numpy.zeros(response_size, dtype=numpy.complex128),
    }


def _expand_diagonal_response(values, nbasis):
    diag = numpy.asarray(values, dtype=numpy.complex128).reshape(2, nbasis)
    out = numpy.zeros((2, nbasis, nbasis), dtype=numpy.complex128)
    idx = numpy.arange(nbasis)
    out[:, idx, idx] = diag
    return out.ravel()


def _ratio_gradient(enumer, edenom, d_enumer, d_edenom):
    if abs(edenom) == 0.0:
        return numpy.zeros_like(d_enumer)
    energy = enumer / edenom
    return d_enumer / edenom - energy * d_edenom / edenom


@dataclass
class HubbardJAXADBlockResult:
    """Raw AD-block accumulators and ratio response for one block."""

    ENumer: numpy.ndarray
    EDenom: numpy.ndarray
    dENumer: numpy.ndarray
    dEDenom: numpy.ndarray
    RDMResponse: numpy.ndarray
    WeightResponse: numpy.ndarray

    @property
    def energy(self):
        if abs(self.EDenom[0]) == 0.0:
            return numpy.nan
        return self.ENumer[0] / self.EDenom[0]

    def as_dict(self):
        return {
            "ENumer": self.ENumer,
            "EDenom": self.EDenom,
            "dENumer": self.dENumer,
            "dEDenom": self.dEDenom,
            "RDMResponse": self.RDMResponse,
            "WeightResponse": self.WeightResponse,
        }


class HubbardJAXADBlockRunner:
    """Run long Hubbard AD blocks and update the live walker population.

    By default the block is evaluated chunk-by-chunk over walkers and each
    chunk performs local stochastic reconfiguration inside the JAX scan.  With
    ``global_pop_control=True`` the block must run unchunked and uses mpi4jax
    collectives inside the differentiated scan so copied-walker gradients can
    flow across ranks.
    """

    def __init__(
        self,
        hamiltonian,
        trial,
        timestep,
        ad_block_size,
        measure_freq=25,
        stabilize_freq=5,
        pop_control_freq=5,
        spin_decomp=True,
        trial_response=False,
        uhf_mixing=0.5,
        uhf_n_scf=50,
        uhf_ueff=None,
        walker_batch_size=None,
        local_pop_control=True,
        global_pop_control=False,
        mpi_handler=None,
        checkpoint_steps=True,
        force_host=False,
        response_mode="full",
    ):
        if not isinstance(hamiltonian, Hubbard):
            raise ValueError("HubbardJAXADBlockRunner requires a Hubbard hamiltonian.")
        if trial_response is True:
            trial_response = "jax_uhf"
        if trial_response not in (False, "jax_uhf"):
            raise ValueError('trial_response must be False or "jax_uhf".')
        for name, value in (
            ("ad_block_size", ad_block_size),
            ("measure_freq", measure_freq),
            ("stabilize_freq", stabilize_freq),
            ("pop_control_freq", pop_control_freq),
        ):
            if int(value) < 1:
                raise ValueError(f"{name} must be positive.")
        if int(ad_block_size) % int(measure_freq) != 0:
            raise ValueError("ad_block_size must be divisible by measure_freq.")

        self.hamiltonian = hamiltonian
        self.trial = trial
        self.timestep = float(timestep)
        self.ad_block_size = int(ad_block_size)
        self.measure_freq = int(measure_freq)
        self.stabilize_freq = int(stabilize_freq)
        self.pop_control_freq = int(pop_control_freq)
        self.spin_decomp = spin_decomp
        self.trial_response = trial_response
        self.uhf_mixing = float(uhf_mixing)
        self.uhf_n_scf = int(uhf_n_scf)
        self.uhf_ueff = float(hamiltonian.U.real if uhf_ueff is None else uhf_ueff)
        self.walker_batch_size = None if walker_batch_size is None else int(walker_batch_size)
        self.local_pop_control = bool(local_pop_control)
        self.global_pop_control = bool(global_pop_control)
        self.mpi_handler = mpi_handler
        self.checkpoint_steps = bool(checkpoint_steps)
        self.force_host = bool(force_host)
        self.response_mode = response_mode
        if self.walker_batch_size is not None and self.walker_batch_size < 1:
            raise ValueError("walker_batch_size must be positive.")
        if self.response_mode not in ("full", "diagonal"):
            raise ValueError('response_mode must be "full" or "diagonal".')
        if self.global_pop_control and self.walker_batch_size is not None:
            raise ValueError("global_pop_control=True requires walker_batch_size=None.")

        _, _, self.aux_wfac, self.delta = construct_hirsch_auxiliaries(
            hamiltonian, self.timestep, spin_decomp=self.spin_decomp
        )
        self.response_shape = (2, hamiltonian.nbasis, hamiltonian.nbasis)
        self.coupling_shape = (
            self.response_shape if self.response_mode == "full" else (2, hamiltonian.nbasis)
        )
        self._jax_static_inputs = None

    def _static_inputs(self):
        from ipie.propagation.hubbard_jax import as_jax_array

        if self._jax_static_inputs is None:
            hamiltonian = self.hamiltonian
            trial = self.trial
            psi0b = trial.psi0b
            if psi0b.shape[1] == 0:
                psi0b = numpy.zeros((hamiltonian.nbasis, 0), dtype=numpy.complex128)
            coupling = numpy.zeros(self.coupling_shape, dtype=numpy.float64)
            self._jax_static_inputs = {
                "coupling": as_jax_array(coupling, dtype="float64", force_host=True),
                "psi0a": as_jax_array(trial.psi0a, dtype="complex128", force_host=self.force_host),
                "psi0b": as_jax_array(psi0b, dtype="complex128", force_host=self.force_host),
                "h1e": as_jax_array(hamiltonian.T, dtype="complex128", force_host=self.force_host),
                "U": as_jax_array(hamiltonian.U, dtype="complex128", force_host=True),
                "ecore": as_jax_array(hamiltonian.ecore, dtype="complex128", force_host=True),
                "delta": as_jax_array(self.delta, dtype="complex128", force_host=self.force_host),
                "aux_wfac": as_jax_array(
                    self.aux_wfac, dtype="complex128", force_host=self.force_host
                ),
            }
        return self._jax_static_inputs

    def _compute_chunk(
        self,
        walkers,
        random_fields,
        pop_control_randoms,
        mpi_comm=None,
        mpi_rank=0,
        mpi_size=1,
    ):
        from ipie.propagation.hubbard_jax import (
            as_jax_array,
            block_until_ready,
            hubbard_ad_block_raw,
        )

        phib = walkers.phib if walkers.phib is not None else _dummy_phib(walkers)
        static = self._static_inputs()
        result = hubbard_ad_block_raw(
            static["coupling"],
            as_jax_array(walkers.phia, dtype="complex128", force_host=self.force_host),
            as_jax_array(phib, dtype="complex128", force_host=self.force_host),
            as_jax_array(walkers.weight, dtype="float64", force_host=self.force_host),
            as_jax_array(walkers.log_shift, dtype="float64", force_host=self.force_host),
            static["psi0a"],
            static["psi0b"],
            static["h1e"],
            static["U"],
            static["ecore"],
            static["delta"],
            static["aux_wfac"],
            as_jax_array(random_fields, dtype="float64", force_host=self.force_host),
            as_jax_array(pop_control_randoms, dtype="float64", force_host=self.force_host),
            self.timestep,
            walkers.nup,
            walkers.ndown,
            rhf=bool(walkers.rhf),
            trial_response=self.trial_response,
            uhf_n_scf=self.uhf_n_scf,
            uhf_mixing=self.uhf_mixing,
            uhf_ueff=self.uhf_ueff,
            stabilize_freq=self.stabilize_freq,
            pop_control_freq=self.pop_control_freq,
            measure_freq=self.measure_freq,
            local_pop_control=self.local_pop_control,
            global_pop_control=self.global_pop_control,
            mpi_comm=mpi_comm,
            mpi_rank=mpi_rank,
            mpi_size=mpi_size,
            checkpoint_steps=self.checkpoint_steps,
            coupling_mode=self.response_mode,
        )
        result = block_until_ready(result)
        enumer, edenom, d_enumer, d_edenom, phia, phib, inva, invb, weight, ovlp = result
        if self.response_mode == "diagonal":
            d_enumer = _expand_diagonal_response(d_enumer, self.hamiltonian.nbasis)
            d_edenom = _expand_diagonal_response(d_edenom, self.hamiltonian.nbasis)
        raw = {
            "ENumer": numpy.asarray(enumer, dtype=numpy.complex128).reshape(1),
            "EDenom": numpy.asarray(edenom, dtype=numpy.complex128).reshape(1),
            "dENumer": numpy.asarray(d_enumer, dtype=numpy.complex128).ravel(),
            "dEDenom": numpy.asarray(d_edenom, dtype=numpy.complex128).ravel(),
        }
        final = {
            "phia": phia,
            "phib": phib,
            "inv_ovlp_a": inva,
            "inv_ovlp_b": invb,
            "weight": weight,
            "ovlp": ovlp,
        }
        return raw, final

    def _mpi_inputs(self, walkers):
        if not self.global_pop_control:
            return None, 0, 1
        mpi_handler = self.mpi_handler or getattr(walkers, "mpi_handler", None)
        if mpi_handler is None:
            raise ValueError("global_pop_control=True requires mpi_handler or walkers.mpi_handler.")
        from ipie.propagation.hubbard_jax import hashable_mpi_comm

        comm = mpi_handler.comm
        return hashable_mpi_comm(comm), comm.rank, comm.size

    def _sync_pop_control_randoms(self, pop_control_randoms, walkers):
        if not self.global_pop_control:
            return pop_control_randoms
        mpi_handler = self.mpi_handler or getattr(walkers, "mpi_handler", None)
        if mpi_handler is None:
            raise ValueError("global_pop_control=True requires mpi_handler or walkers.mpi_handler.")
        comm = mpi_handler.comm
        host_randoms = numpy.asarray(
            (
                pop_control_randoms.get()
                if hasattr(pop_control_randoms, "get")
                else pop_control_randoms
            ),
            dtype=numpy.float64,
        ).copy()
        comm.Bcast(host_randoms, root=0)
        return xp.asarray(host_randoms)

    def _store_chunk(self, walkers, start, stop, final):
        walkers.phia[start:stop] = xp.asarray(final["phia"]).copy()
        if walkers.phib is not None:
            walkers.phib[start:stop] = xp.asarray(final["phib"]).copy()
        walkers.inv_ovlp_a[start:stop] = xp.asarray(final["inv_ovlp_a"]).copy()
        if walkers.inv_ovlp_b is not None:
            walkers.inv_ovlp_b[start:stop] = xp.asarray(final["inv_ovlp_b"]).copy()
        walkers.weight[start:stop] = xp.asarray(final["weight"]).copy()
        walkers.ovlp[start:stop] = xp.asarray(final["ovlp"]).copy()

    def run_block(self, walkers):
        if walkers.inv_ovlp_a is None:
            walkers.inverse_overlap(self.trial)
        total = _zero_raw(self.hamiltonian.nbasis)
        mpi_comm, mpi_rank, mpi_size = self._mpi_inputs(walkers)
        chunk_size = self.walker_batch_size or walkers.nwalkers
        start = 0
        while start < walkers.nwalkers:
            stop = min(walkers.nwalkers, start + chunk_size)
            view = _walker_view(walkers, start, stop)
            random_fields = _random_uniform(
                (self.ad_block_size, self.hamiltonian.nbasis, view.nwalkers)
            )
            pop_control_randoms = _random_uniform((self.ad_block_size,))
            random_fields = xp.ascontiguousarray(random_fields)
            pop_control_randoms = xp.ascontiguousarray(
                self._sync_pop_control_randoms(pop_control_randoms, walkers)
            )
            raw, final = self._compute_chunk(
                view,
                random_fields,
                pop_control_randoms,
                mpi_comm=mpi_comm,
                mpi_rank=mpi_rank,
                mpi_size=mpi_size,
            )
            for key in total:
                total[key] += raw[key]
            self._store_chunk(walkers, start, stop, final)
            start = stop

        walkers.unscaled_weight = walkers.weight.copy()
        response = _ratio_gradient(
            total["ENumer"][0], total["EDenom"][0], total["dENumer"], total["dEDenom"]
        )
        return HubbardJAXADBlockResult(
            ENumer=total["ENumer"],
            EDenom=total["EDenom"],
            dENumer=total["dENumer"],
            dEDenom=total["dEDenom"],
            RDMResponse=response,
            WeightResponse=total["dEDenom"].copy(),
        )
