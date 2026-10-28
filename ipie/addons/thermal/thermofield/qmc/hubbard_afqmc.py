# Copyright 2022 The ipie Developers. All Rights Reserved.
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
#
# Authors: Jinghong Zhang <jinghongzhang@fas.harvard.edu>
#
"""Driver to perform thermofield constrained-path AFQMC for Hubbard models"""

import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy

from ipie.addons.thermal.qmc.options import ThermalQMCParams
from ipie.addons.thermal.thermofield.estimators.cyclic_average import cyclic_average_hubbard
from ipie.addons.thermal.thermofield.estimators.handler import ThermofieldEstimatorHandler
from ipie.addons.thermal.thermofield.estimators.hubbard import (
    delete_group_sums,
    HubbardSpinObservables,
    HubbardTraceEstimator,
)
from ipie.addons.thermal.thermofield.utils.linalg import principal_log_phase
from ipie.addons.thermal.thermofield.walkers.pop_controller import ThermofieldPopController
from ipie.config import config, MPI
from ipie.qmc.afqmc import AFQMCBase
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import to_host
from ipie.walkers.base_walkers import WalkerAccumulator


def propagate_population_cp(
    afqmc, walkers, trial, prop, nslices, pcontrol, weight_cap=0.0, on_slice=None
):
    """Walk the population over [0, beta] from the identity with constrained-path propagation.

    Parameters
    ----------
    afqmc : :class:`ThermofieldHubbardAFQMC`
        Driver; provides the communicator, stabilization and population
        control frequencies, and receives `population_log_normalization`, the
        log of the weight removed by population control.
    walkers : :class:`ThermofieldWalkers`
        Walkers, reset at the start.
    trial : :class:`ThermofieldCPTrial`
        Guide.
    prop : :class:`ThermofieldHubbardBase`
        Propagator.
    nslices : int
        Number of slices.
    pcontrol : :class:`ThermofieldPopController`
        Population controller.
    weight_cap : float
        Walker weights are clipped at this multiple of the total weight; 0
        uses 1e6. Default 0.0.
    on_slice : callable
        Called with t + 1 after slice t, after stabilization and before
        population control. Default None.
    """
    if not isinstance(nslices, (int, numpy.integer)) or nslices < 1:
        raise ValueError("nslices must be a positive integer.")
    if not numpy.isfinite(weight_cap) or weight_cap < 0:
        raise ValueError("weight_cap must be finite and non-negative.")
    comm = afqmc.mpi_handler.comm
    afqmc.population_log_normalization = 0.0
    walkers.reset(trial, compute_cache=False)
    if prop.record_fields:
        # Population control copies the field histories with the walkers.
        if not hasattr(walkers, "field_hist") or walkers.field_hist.shape[1] != nslices:
            walkers.add_walker_array(
                "field_hist", numpy.zeros((walkers.nwalkers, nslices, prop.nfields))
            )
        else:
            walkers.field_hist[:] = 0.0
    prop.init_walker_cache(walkers, trial)
    if prop.batched and prop.cache_qinv:
        prop.seed_qinv_cache(walkers)
    for t in range(nslices):
        if prop.discrete:
            # This path can fail on one process; all fail together.
            failure = None
            try:
                prop.propagate_walkers(walkers, trial, t)
            except (ValueError, ArithmeticError, numpy.linalg.LinAlgError) as exc:
                failure = f"{type(exc).__name__}: {exc}"
            _synchronize_error(comm, failure, "Discrete-field propagation failed")
        else:
            prop.propagate_walkers(walkers, trial, t)
        if t > 0:
            invalid_count = int((~numpy.isfinite(walkers.weight)).sum())
            prop.invalid_weight_count += invalid_count
            numpy.nan_to_num(walkers.weight, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            wbound = pcontrol.total_weight * (weight_cap if weight_cap > 0 else 1.0e6)
            population_hits = int((numpy.abs(walkers.weight) > wbound).sum())
            prop.population_weight_bound_count += population_hits
            numpy.clip(walkers.weight, a_min=-wbound, a_max=wbound, out=walkers.weight)
        if t > 0 and (t + 1) % afqmc.params.num_stblz == 0:
            walkers.stabilize()
            if prop.batched and prop.cache_qinv:
                prop.refresh_qinv_after_stabilize(walkers)
            if not prop.batched:
                # Stabilization does not change the walkers, so the batched
                # path keeps the cached overlaps and Green's functions; the
                # per-walker path recomputes them as it always did.
                prop.refresh_cache_after_stabilize(walkers, trial, t)
        if on_slice is not None:
            on_slice(t + 1)
        if t > 0 and t % afqmc.params.pop_control_freq == 0:
            # All ranks must fail together before the controller divides by
            # its total weight. This does not change any surviving weights.
            local_health = numpy.array(
                [
                    numpy.sum(numpy.abs(walkers.weight)),
                    float(not numpy.isfinite(walkers.weight).all()),
                ]
            )
            global_health = numpy.zeros_like(local_health)
            comm.Allreduce(local_health, global_health)
            if (
                global_health[1] > 0
                or not numpy.isfinite(global_health[0])
                or global_health[0] <= 0
            ):
                raise RuntimeError(
                    "Hubbard walker population is all dead or non-finite before population control."
                )
            # Every block restarts from the identity. Keep the scale removed
            # by population control, so that trace sums of independent blocks
            # combine without equalizing their partition functions. A single
            # walker is not rescaled.
            if pcontrol.ntot_walkers > 1:
                afqmc.population_log_normalization += numpy.log(global_health[0]) - numpy.log(
                    pcontrol.target_weight
                )
            pcontrol.pop_control(walkers, comm)
            if prop.batched and prop.cache_qinv:
                # Population control replaces Q factors; recompute the cache.
                prop.seed_qinv_cache(walkers)


def _json_default(value):
    if isinstance(value, numpy.ndarray):
        return value.tolist()
    if isinstance(value, numpy.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value)}")


def _synchronize_error(comm, error, context):
    """Raise on every process if any process reports an error."""
    failures = comm.allgather(error)
    if any(item is not None for item in failures):
        details = "; ".join(f"rank {rank}: {item}" for rank, item in enumerate(failures) if item)
        raise RuntimeError(f"{context}: {details}")


class ThermofieldHubbardAFQMC(AFQMCBase):
    """Thermofield constrained-path AFQMC driver for Hubbard models.

    Every block walks the population over [0, beta] from the identity and
    measures the closed trace at the endpoint, with the signed correction
    Q = det(I + Delta) / S for the guide, optionally averaged over the cyclic
    cuts of the recorded paths, and the mixed estimator of the guide. Block
    sums are kept separately (ratio of means) with a log scale; blocks with a
    numerical weight guard event are refused. On the GPU (IPIE_USE_GPU=1) the
    walker matrices stay on the device and the measurements run on the host;
    this needs discrete fields.

    Parameters
    ----------
    hamiltonian :
        Hamiltonian.
    hop : :class:`numpy.ndarray`
        Hopping matrix used by the default observables.
    trial : :class:`ThermofieldCPTrial`
        Guide.
    walkers : :class:`ThermofieldWalkers`
        Walkers; their number per MPI process sets the population.
    propagator : :class:`ThermofieldHubbardBase`
        Constrained-path propagator.
    mpi_handler : :class:`ipie.utils.mpi.MPIHandler`
        MPI handler.
    lx, ly : int
        Lattice dimensions (lx = nbasis and ly = 1 for other observables).
    nblocks : int
        Number of blocks. Default 100.
    stabilize_freq : int
        Stabilization frequency in slices. Default 5.
    pop_control_freq : int
        Population control frequency in slices. Default 10.
    wrap_stride : int
        Stride of the cyclic cuts; None measures the endpoint only. Default 10.
    wrap_memory : float
        Bytes for the cyclic-averaging snapshots. Default 1.5e8.
    seed : int
        Random number seed (`set_rng_seed`). Default 7.
    metadata : dict
        Annotations saved with the results. Default None.
    reconstruction_tolerance : float
        Largest allowed deviation of the cyclic reconstruction. Default 1e-7.
    preserve_population_normalization : bool
        Keep the scale removed by population control in the block log scale,
        so that blocks combine as one ratio of means. Default False.
    measure_one_rdm : bool
        Accumulate the one-body density matrix. Default False.
    shell_distances_squared : tuple
        Correlation shells of the default observables. Default (1, 2, 4, 5).
    wrap_stabilize_freq : int
        Restratification interval of the cyclic averaging. Default 20.
    measure_mixed : bool
        Measure the mixed estimator with the guide as bra; False for a
        spin-mixing guide. Default True.
    observables : :class:`TraceRatioObservables`
        Observables; default :class:`HubbardSpinObservables` of the lattice.
    snapshots : tuple
        (slice, eta) pairs: after that slice, measure the mixed estimator with
        the trial at time eta, reweighting every walker by F = S_eta / S_guide,
        the ratio of its overlaps with that trial and with the guide. Pair k is
        stored as snap<k>_numer, snap<k>_denom, ... in the block results.
        Default ().
    verbose : int
        Print information. Default 0.
    """

    def __init__(
        self,
        hamiltonian,
        hop,
        trial,
        walkers,
        propagator,
        mpi_handler,
        lx,
        ly,
        nblocks=100,
        stabilize_freq=5,
        pop_control_freq=10,
        wrap_stride=10,
        wrap_memory=1.5e8,
        seed=7,
        metadata=None,
        reconstruction_tolerance=1e-7,
        preserve_population_normalization=False,
        measure_one_rdm=False,
        shell_distances_squared=(1, 2, 4, 5),
        wrap_stabilize_freq=20,
        measure_mixed=True,
        observables=None,
        snapshots=(),
        verbose=0,
    ):
        for name, value in (
            ("nblocks", nblocks),
            ("stabilize_freq", stabilize_freq),
            ("pop_control_freq", pop_control_freq),
            ("wrap_stabilize_freq", wrap_stabilize_freq),
        ):
            if int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        nslices = int(round(propagator.beta / propagator.dt))
        if nslices < 1 or not numpy.isclose(
            nslices * propagator.dt, propagator.beta, rtol=0, atol=1e-12
        ):
            raise ValueError("beta / timestep must be an integer; use dt=beta/ceil(beta/dt_max)")
        if wrap_stride is not None and (int(wrap_stride) != wrap_stride or wrap_stride < 1):
            raise ValueError("wrap_stride must be a positive integer or None")
        if not numpy.isfinite(wrap_memory) or wrap_memory <= 0:
            raise ValueError("wrap_memory must be positive")
        if not numpy.isfinite(reconstruction_tolerance) or reconstruction_tolerance <= 0:
            raise ValueError("reconstruction_tolerance must be finite and positive")
        if walkers.nbasis != lx * ly or trial.nbasis != walkers.nbasis:
            raise ValueError("Lattice, walker, and trial dimensions disagree")
        if not numpy.isclose(trial.beta, propagator.beta):
            raise ValueError("Trial and propagator beta disagree")
        if not propagator.constraint:
            raise ValueError("This driver requires constrained-path propagation")
        comm = mpi_handler.comm
        # pylint: disable = no-value-for-parameter
        params = ThermalQMCParams(
            mu=0.0,  # The chemical potential is part of the propagator's h_kin.
            beta=propagator.beta,
            num_walkers=walkers.nwalkers,
            total_num_walkers=walkers.nwalkers * comm.size,
            num_blocks=int(nblocks),
            timestep=propagator.dt,
            num_stblz=stabilize_freq,
            pop_control_freq=pop_control_freq,
            pop_control_method="pair_branch",
            rng_seed=seed,
        )
        super().__init__(
            None, hamiltonian, trial, walkers, propagator, mpi_handler, params, verbose
        )
        # On the GPU set_rng_seed seeds the device stream only; population
        # control draws from the host stream.
        if config.get_option("use_gpu"):
            numpy.random.seed(self._parallel_rng_seed)
        self.nslices = nslices
        self.wrap_stride = wrap_stride
        self.wrap_memory = float(wrap_memory)
        self.wrap_stabilize_freq = int(wrap_stabilize_freq)
        self.reconstruction_tolerance = reconstruction_tolerance
        self.preserve_population_normalization = bool(preserve_population_normalization)
        self.measure_mixed = bool(measure_mixed)
        self.population_log_normalization = 0.0
        self.measure_one_rdm = bool(measure_one_rdm)
        self.seed = seed
        self.propagator.record_fields = wrap_stride is not None
        if observables is None:
            observables = HubbardSpinObservables(hop, propagator.U, lx, ly, shell_distances_squared)
        self.observables = observables
        self.pcontrol = ThermofieldPopController(
            walkers.nwalkers, nslices, mpi_handler, "pair_branch"
        )
        self.metadata = dict(metadata or {})
        # Numerical facts override caller annotations.
        self.metadata.update(
            lx=lx,
            ly=ly,
            beta=propagator.beta,
            timestep=propagator.dt,
            nslices=nslices,
            nwalkers_per_rank=walkers.nwalkers,
            nranks=mpi_handler.comm.size,
            nblocks=self.params.num_blocks,
            seed=seed,
            guide_mode=propagator.guide_mode,
            U=propagator.U,
            hs_type=propagator.hs_type,
            hs_gamma=propagator.gamma,
            field_proposal=propagator.field_proposal,
            energy_shift=propagator.energy_shift,
            fixed_guide_time=propagator.fixed_guide_time,
            wrap_stride=wrap_stride,
            stabilize_freq=stabilize_freq,
            wrap_stabilize_freq=self.wrap_stabilize_freq,
            pop_control_freq=pop_control_freq,
            weight_cap=0.0,
            preserve_population_normalization=self.preserve_population_normalization,
            estimator="signed trace; ratio of means; connected after ensemble averaging",
            mixed_estimator="guide bra: Tr[Lambda H B]/Tr[Lambda B], Lambda=exp(-fixed_guide_time k_T), plain weights",
            measure_one_rdm=self.measure_one_rdm,
            measure_mixed=self.measure_mixed,
            one_rdm_convention="G[s,i,j]=<c_j^dag c_i>; signed trace ROM",
            rng_seeds=comm.allgather(int(self._parallel_rng_seed)),
        )
        self.snapshots = [(int(s), float(eta)) for s, eta in snapshots]
        for s, _ in self.snapshots:
            if not 1 <= s <= nslices:
                raise ValueError("Snapshot slice outside the walk.")
        if self.snapshots:
            self.metadata.update(
                snapshots=self.snapshots,
                snapshot_estimator="sum w F O[G_eta] / sum w F, F = S_eta / S_guide",
            )
        self._snaps = {}
        self.results = []

    # Compatibility with earlier versions.
    @property
    def nblocks(self):
        return self.params.num_blocks

    def _measure_snapshots(self, slice_index):
        """Measure the snapshots taken after slice `slice_index`."""
        for k, (s, eta) in enumerate(self.snapshots):
            if s == slice_index:
                self._snaps[k] = self._measure_snapshot(eta)

    def _measure_snapshot(self, eta):
        """Mixed measurement with the trial at time `eta`, reweighted by S_eta / S_guide."""
        comm, walkers = self.mpi_handler.comm, self.walkers
        alive = walkers.weight > 0
        error = None
        try:
            log_s, greens = self.trial.calc_log_overlap_and_greens_function_scaled_batched(
                walkers.Qmat, walkers.log_d, walkers.Tmat, eta
            )
            log_s, greens = to_host(log_s), to_host(greens)
        except (ValueError, ArithmeticError, numpy.linalg.LinAlgError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        _synchronize_error(comm, error, "Snapshot contraction failed")
        bad = alive & (
            ~numpy.isfinite(log_s) | ~numpy.isfinite(greens.reshape(len(greens), -1)).all(axis=1)
        )
        if comm.allreduce(int(bad.sum())):
            raise FloatingPointError("Nonfinite snapshot contraction on a live walker")
        log_f = log_s - walkers.log_ovlp
        shift = float(numpy.max(log_f.real[alive])) if alive.any() else -numpy.inf
        shift = comm.allreduce(shift, op=MPI.MAX)
        wf = numpy.zeros(walkers.nwalkers, dtype=numpy.complex128)
        wf[alive] = walkers.weight[alive] * numpy.exp(log_f[alive] - shift)
        obs = numpy.zeros((walkers.nwalkers, len(self.observables.names)), dtype=complex)
        error = None
        try:
            for iw in numpy.flatnonzero(alive):
                obs[iw] = self.observables.measure(greens[iw])
        except (ValueError, ArithmeticError, numpy.linalg.LinAlgError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        _synchronize_error(comm, error, "Snapshot observable failed")
        local = numpy.concatenate(
            (wf @ obs, [wf.sum(), numpy.abs(wf).sum(), numpy.sum(numpy.abs(wf) ** 2)])
        )
        total = numpy.empty_like(local)
        comm.Allreduce(local, total)
        nobs = len(self.observables.names)
        return dict(
            numer=total[:nobs],
            denom=total[nobs],
            absolute_weight=total[nobs + 1].real,
            squared_weight=total[nobs + 2].real,
            shift=shift,
            eta=eta,
        )

    def _measure_block(self):
        comm, walkers = self.mpi_handler.comm, self.walkers
        error = None
        try:
            trace_log, greens = self.trial.calc_log_overlap_and_greens_function_scaled_batched(
                walkers.Qmat, walkers.log_d, walkers.Tmat, 0.0
            )
            # The measurements run on the host.
            trace_log, greens = to_host(trace_log), to_host(greens)
        except (ValueError, ArithmeticError, numpy.linalg.LinAlgError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        _synchronize_error(comm, error, "Trace contraction failed")
        alive = walkers.weight > 0
        bad = alive & (~numpy.isfinite(trace_log) | ~numpy.isfinite(greens).all(axis=(1, 2, 3)))
        if comm.allreduce(int(bad.sum())):
            raise FloatingPointError("Nonfinite trace measurement on a live walker")
        logq = trace_log - walkers.log_ovlp
        shift = float(numpy.max(logq.real[alive])) if alive.any() else -numpy.inf
        shift = comm.allreduce(shift, op=MPI.MAX)
        if not numpy.isfinite(shift):
            raise RuntimeError("All Hubbard walkers died; increase population or revise guide")
        weights = numpy.zeros(walkers.nwalkers, dtype=numpy.complex128)
        weights[alive] = walkers.weight[alive] * numpy.exp(logq[alive] - shift)
        endpoint = numpy.zeros((walkers.nwalkers, len(self.observables.names)), dtype=complex)
        error = None
        try:
            for iw in numpy.flatnonzero(alive):
                endpoint[iw] = self.observables.measure(greens[iw])
        except (ValueError, ArithmeticError, numpy.linalg.LinAlgError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        _synchronize_error(comm, error, "Endpoint observable failed")
        averaged = endpoint.copy()
        if self.measure_one_rdm:
            rdm_averaged = greens.copy()
        # Mixed estimator: the guide itself is the bra, Tr[Lambda H B_w]/Tr[Lambda B_w] with
        # Lambda = exp(-fixed_guide_time k_T) (identity in completion mode).  Its weights are the
        # plain walker weights, no end-point correction Q.  Exact when k_T is the exact one-body
        # Hamiltonian; otherwise a trial-dependent estimate of the energy at imaginary time
        # beta_walk + fixed_guide_time.
        mixed_scale = (
            self.propagator.fixed_guide_time if self.propagator.guide_mode == "fixed" else 0.0
        )
        error = None
        try:
            mixed_log, greens_mixed = (
                self.trial.calc_log_overlap_and_greens_function_scaled_batched(
                    walkers.Qmat, walkers.log_d, walkers.Tmat, mixed_scale
                )
            )
            mixed_log, greens_mixed = to_host(mixed_log), to_host(greens_mixed)
        except (ValueError, ArithmeticError, numpy.linalg.LinAlgError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        _synchronize_error(comm, error, "Mixed (guide-bra) contraction failed")
        bad = alive & (
            ~numpy.isfinite(mixed_log)
            | ~numpy.isfinite(greens_mixed.reshape(len(greens_mixed), -1)).all(axis=1)
        )
        if comm.allreduce(int(bad.sum())):
            raise FloatingPointError("Nonfinite mixed measurement on a live walker")
        mixed = numpy.zeros((walkers.nwalkers, len(self.observables.names)), dtype=complex)
        error = None
        try:
            if self.measure_mixed:
                for iw in numpy.flatnonzero(alive):
                    mixed[iw] = self.observables.measure(greens_mixed[iw])
        except (ValueError, ArithmeticError, numpy.linalg.LinAlgError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        _synchronize_error(comm, error, "Mixed observable failed")
        mixed_weights = numpy.zeros(walkers.nwalkers, dtype=numpy.complex128)
        mixed_weights[alive] = walkers.weight[alive]
        if alive.any():
            # The cached guide overlap must agree with the fresh contraction of the same guide.
            mixed_dev = float(
                numpy.max(numpy.abs(numpy.exp(mixed_log[alive] - walkers.log_ovlp[alive]) - 1.0))
            )
        else:
            mixed_dev = float("nan")
        mixed_dev = comm.allreduce(mixed_dev, op=MPI.MAX)
        wrapdev = recdev = 0.0
        if self.wrap_stride is not None:
            averaged.fill(0)
            if self.measure_one_rdm:
                rdm_averaged.fill(0)
            counts = numpy.zeros(walkers.nwalkers, dtype=int)

            def measure(iw, green, cut):
                averaged[iw] += self.observables.measure(green)
                if self.measure_one_rdm:
                    rdm_averaged[iw] += green
                counts[iw] += 1

            error = None
            try:
                wrapdev, log0 = cyclic_average_hubbard(
                    walkers,
                    self.trial,
                    self.propagator,
                    self.nslices,
                    self.wrap_stride,
                    measure,
                    mem_budget=self.wrap_memory,
                    stab_every=self.wrap_stabilize_freq,
                )
            except (ValueError, ArithmeticError, numpy.linalg.LinAlgError) as exc:
                error = f"{type(exc).__name__}: {exc}"
            _synchronize_error(comm, error, "Cyclic-cut measurement failed")
            expected = len(range(0, self.nslices, self.wrap_stride))
            bad_cuts = int(numpy.any(counts[alive] != expected))
            if comm.allreduce(bad_cuts):
                raise FloatingPointError(
                    "Incomplete cyclic-cut measurement; refusing biased average"
                )
            averaged[alive] /= counts[alive, None]
            if self.measure_one_rdm:
                rdm_averaged[alive] /= counts[alive, None, None, None]
            if alive.any():
                recdev = float(
                    numpy.max(numpy.abs(principal_log_phase(log0[alive] - trace_log[alive])))
                )
            wrapdev = comm.allreduce(float(wrapdev), op=MPI.MAX)
            recdev = comm.allreduce(recdev, op=MPI.MAX)
            if (
                not numpy.isfinite(wrapdev + recdev)
                or max(wrapdev, recdev) > self.reconstruction_tolerance
            ):
                raise FloatingPointError(
                    f"Path reconstruction failed: wrap={wrapdev:g}, endpoint={recdev:g}"
                )
        nobs = len(self.observables.names)
        local = numpy.concatenate(
            (
                weights @ averaged,
                weights @ endpoint,
                [weights.sum(), numpy.abs(weights).sum(), numpy.sum(numpy.abs(weights) ** 2)],
                mixed_weights @ mixed,
                [mixed_weights.sum()],
            )
        )
        total = numpy.empty_like(local)
        comm.Allreduce(local, total)
        if not numpy.isfinite(total).all():
            raise FloatingPointError("Nonfinite Hubbard observable accumulation")
        result = dict(
            numer=total[:nobs],
            numer_endpoint=total[nobs : 2 * nobs],
            denom=total[2 * nobs],
            absolute_weight=total[2 * nobs + 1].real,
            squared_weight=total[2 * nobs + 2].real,
            log_shift=shift
            + (
                self.population_log_normalization if self.preserve_population_normalization else 0.0
            ),
            population_log_normalization=self.population_log_normalization,
            wrapdev=wrapdev,
            recdev=recdev,
            mixed_overlap_dev=mixed_dev,
        )
        if self.measure_mixed:
            result.update(
                numer_mixed=total[2 * nobs + 3 : 3 * nobs + 3], denom_mixed=total[3 * nobs + 3].real
            )
        if self.measure_one_rdm:
            # Use the identical endpoint weights for every cyclic cut. Never
            # diagonalize or Hermitize a walker's transition density here.
            local_rdm = numpy.asarray(
                [
                    numpy.einsum("w,wsij->sij", weights[alive], rdm_averaged[alive]),
                    numpy.einsum("w,wsij->sij", weights[alive], greens[alive]),
                ]
            )
            total_rdm = numpy.empty_like(local_rdm)
            comm.Allreduce(local_rdm, total_rdm)
            if not numpy.isfinite(total_rdm).all():
                raise FloatingPointError("Nonfinite one-body density accumulation")
            result.update(one_rdm_numer=total_rdm[0], one_rdm_numer_endpoint=total_rdm[1])
        return result

    def one_rdm_summary(self, block_size=1, estimator="cyclic"):
        """Signed ensemble density and componentwise delete-group errors.

        No per-walker/cut Hermitization, clipping, or particle-number adjustment
        is applied. These raw trace estimates are distinct from any subsequent
        regularization used to construct a natural-orbital trial.
        """
        if not self.measure_one_rdm or not self.results:
            raise ValueError("Run with measure_one_rdm=True before requesting its summary")
        if estimator not in ("cyclic", "endpoint"):
            raise ValueError("estimator must be cyclic or endpoint")
        nblocks = len(self.results)
        if int(block_size) != block_size or block_size < 1 or nblocks % int(block_size):
            raise ValueError("block_size must be positive and divide the block count")
        block_size = int(block_size)
        key = "one_rdm_numer" if estimator == "cyclic" else "one_rdm_numer_endpoint"
        shifts = numpy.array([r["log_shift"] for r in self.results])
        scale = numpy.exp(shifts - shifts.max())
        numer = numpy.array([r[key] for r in self.results]) * scale[:, None, None, None]
        denom = numpy.array([r["denom"] for r in self.results]) * scale
        if abs(denom.sum()) == 0:
            raise ValueError("The total trace denominator vanishes")
        mean = numer.sum(axis=0) / denom.sum()
        groups = nblocks // block_size
        real_error = numpy.full(mean.shape, numpy.nan)
        imag_error = real_error.copy()
        if groups > 1:
            gn = numer.reshape(groups, block_size, *mean.shape).sum(axis=1)
            gd = denom.reshape(groups, block_size).sum(axis=1)
            ln, ld = delete_group_sums(gn, gd)
            if numpy.any(numpy.abs(ld) == 0):
                raise ValueError("A jackknife deletion has zero trace denominator")
            leave = ln / ld[:, None, None, None]
            real_error = numpy.sqrt(
                (groups - 1)
                / groups
                * numpy.sum((leave.real - leave.real.mean(axis=0)) ** 2, axis=0)
            )
            imag_error = numpy.sqrt(
                (groups - 1)
                / groups
                * numpy.sum((leave.imag - leave.imag.mean(axis=0)) ** 2, axis=0)
            )
        return dict(
            mean=mean,
            error_real=real_error,
            error_imag=imag_error,
            nblocks=nblocks,
            jackknife_groups=groups,
            block_size=block_size,
        )

    def run(
        self,
        walkers=None,
        estimator_filename=None,
        verbose: bool = True,
        discard_weights_aftereq=False,
        additional_estimators=None,
        output=None,
        checkpoint_freq=10,
    ):
        """Run the blocks; all processes must call it.

        Parameters
        ----------
        walkers : :class:`ThermofieldWalkers`
            Walkers. Default the driver's walkers.
        estimator_filename : str
            .h5 file for the block sums. Default None: no file.
        verbose : bool
            Print one line per block. Default True.
        output : str or Path
            NPZ checkpoint of the raw block sums and metadata (an analysis
            file, not a restart), written every `checkpoint_freq` blocks. An
            existing file is refused on a fresh run. Default None.
        checkpoint_freq : int
            Checkpoint frequency in blocks. Default 10.

        Returns
        -------
        results : list
            Block results, one dict of global sums per block.
        """
        self.setup_timers()
        start = time.time()
        comm = self.mpi_handler.comm
        if walkers is not None:
            self.walkers = walkers
        if int(checkpoint_freq) != checkpoint_freq or checkpoint_freq < 1:
            raise ValueError("checkpoint_freq must be a positive integer.")
        if output is not None:
            output = Path(output)
            exists = comm.bcast(output.exists() if comm.rank == 0 else None, root=0)
            if exists and not self.results:
                raise FileExistsError(output)
        self.setup_estimators(estimator_filename, additional_estimators=additional_estimators)
        self.tsetup += time.time() - start
        for block in range(len(self.results), self.params.num_blocks):
            start_step = time.time()
            self._snaps = {}
            propagate_population_cp(
                self,
                self.walkers,
                self.trial,
                self.propagator,
                self.nslices,
                self.pcontrol,
                weight_cap=0.0,
                on_slice=self._measure_snapshots if self.snapshots else None,
            )
            self.tprop += time.time() - start_step
            names = (
                "log_weight_bound_count",
                "population_weight_bound_count",
                "invalid_weight_count",
            )
            local_counts = numpy.array(
                [getattr(self.propagator, name) for name in names], dtype=numpy.int64
            )
            counts = numpy.empty_like(local_counts)
            comm.Allreduce(local_counts, counts)
            if counts.any():
                detail = dict(zip(names, map(int, counts)))
                raise FloatingPointError(
                    f"Numerical weight limiter activated; refusing block {block + 1}: {detail}. "
                    "Choose an energy_shift near the grand energy and inspect population control."
                )
            start = time.time()
            result = self._measure_block()
            for k, snap in self._snaps.items():
                for key, value in snap.items():
                    result[f"snap{k}_{key}"] = value
            self.results.append(result)
            self.estimators["hubbard"].set_block(self.results[-1], comm)
            # The accumulator sums with `xp`; the walker weights live on the host.
            self.accumulators.update(
                SimpleNamespace(
                    weight=xp.asarray(self.walkers.weight),
                    unscaled_weight=xp.asarray(self.walkers.unscaled_weight),
                    hybrid_energy=xp.asarray(self.walkers.hybrid_energy),
                )
            )
            self.estimators.compute_estimators(
                hamiltonian=self.hamiltonian, trial=self.trial, walker_batch=self.walkers
            )
            self.estimators.print_block(comm, block + 1, self.accumulators)
            self.accumulators.zero()
            self.testim += time.time() - start
            if output is not None and (
                (block + 1) % checkpoint_freq == 0 or block + 1 == self.params.num_blocks
            ):
                self.save(output)
            if verbose and comm.rank == 0:
                last = self.results[-1]
                sign = last["denom"].real / last["absolute_weight"]
                print(
                    f"# Hubbard block {block+1}/{self.params.num_blocks}: sign={sign:.5f}, "
                    f"wrapdev={last['wrapdev']:.2e}, recdev={last['recdev']:.2e}",
                    flush=True,
                )
            self.tstep += time.time() - start_step
        return self.results

    def setup_estimators(self, filename, additional_estimators=None):
        self.accumulators = WalkerAccumulator(
            ["Weight", "WeightFactor", "HybridEnergy"], self.params.num_steps_per_block
        )
        comm = self.mpi_handler.comm
        self.estimators = ThermofieldEstimatorHandler(
            comm,
            self.hamiltonian,
            self.trial,
            eta_members=(),
            measure_trace=False,
            walker_state=self.accumulators,
            verbose=(comm.rank == 0 and self.verbose),
            filename=filename,
        )
        self.estimators["hubbard"] = HubbardTraceEstimator(self.observables)
        if additional_estimators is not None:
            for k, v in additional_estimators.items():
                self.estimators[k] = v
        self.estimators.json_string = json.dumps(self.metadata, default=_json_default)
        self.estimators.initialize(comm)

    def save(self, output):
        """Atomically save globally reduced complex raw sums on rank zero."""
        comm = self.mpi_handler.comm
        error = None
        if comm.rank == 0:
            try:
                output = Path(output)
                output.parent.mkdir(parents=True, exist_ok=True)
                arrays = {
                    key: numpy.asarray([row[key] for row in self.results])
                    for key in self.results[0]
                }
                arrays["metadata"] = json.dumps(self.metadata, default=_json_default)
                arrays["observable_names"] = numpy.asarray(self.observables.names)
                arrays["observable_metadata"] = json.dumps(
                    self.observables.metadata, default=_json_default
                )
                arrays["trial_k"] = self.trial.k_trial
                arrays["hamiltonian_h1"] = self.hamiltonian.H1
                arrays["hopping"] = self.observables.hop
                temporary = output.with_name(output.name + ".tmp")
                with temporary.open("wb") as stream:
                    numpy.savez(stream, **arrays)
                temporary.replace(output)
            except (OSError, ValueError, TypeError) as exc:
                error = f"{type(exc).__name__}: {exc}"
        _synchronize_error(comm, error, "Checkpoint write failed")

    def finalise(self, verbose=True, block_size=1):
        """Return ROM results with jackknife errors and reweighting diagnostics."""
        if len(self.results) < 2:
            raise ValueError("At least two blocks are needed for uncertainty estimates")
        shifts = numpy.array([row["log_shift"] for row in self.results])
        denoms = numpy.array([row["denom"] for row in self.results])
        summary = {}
        for label, key in (("cyclic", "numer"), ("endpoint", "numer_endpoint")):
            summary[label] = self.observables.summary(
                numpy.array([row[key] for row in self.results]),
                denoms,
                shifts,
                block_size=block_size,
            )
        scale = numpy.exp(shifts - shifts.max())
        abs_weights = numpy.array([row["absolute_weight"] for row in self.results])
        squares = numpy.array([row["squared_weight"] for row in self.results])
        summary["mean_sign"] = float((scale @ denoms).real / (scale @ abs_weights))
        # Weight concentration only; genealogy means this is not an independent sample count.
        summary["weight_concentration_count"] = float(
            (scale @ abs_weights) ** 2 / (scale**2 @ squares)
        )
        summary["max_wrapdev"] = max(row["wrapdev"] for row in self.results)
        summary["max_recdev"] = max(row["recdev"] for row in self.results)
        if "numer_mixed" in self.results[0]:
            # Guide-bra (mixed) estimator: positive weights, block normalizers only.
            mixed_denoms = numpy.array([row["denom_mixed"] for row in self.results], dtype=complex)
            pop_shifts = numpy.array([row["population_log_normalization"] for row in self.results])
            summary["mixed"] = self.observables.summary(
                numpy.array([row["numer_mixed"] for row in self.results]),
                mixed_denoms,
                pop_shifts,
                block_size=block_size,
            )
        summary["max_mixed_overlap_dev"] = max(row["mixed_overlap_dev"] for row in self.results)
        if verbose and self.mpi_handler.comm.rank == 0:
            print(json.dumps(summary, default=_json_default, indent=2), flush=True)
        return summary


def tune_chemical_potential(evaluate_density, target, bracket, tolerance=0.002, max_iter=20):
    """Bracket a physical ``+mu N`` chemical potential using measured trace density.

    ``evaluate_density(mu)`` must run an independent pilot and return its ROM
    (density, standard_error).  Trial chemical potential is a separate input.
    Success requires |n-target| + 2*error <= tolerance.  An unresolved noisy
    pilot raises, rather than silently accepting the trial's target density.
    Returns (mu, history), with every pilot retained for provenance.
    """
    lo, hi = map(float, bracket)
    if not (
        numpy.isfinite([lo, hi, target, tolerance]).all()
        and lo < hi
        and 0 < target < 2
        and tolerance > 0
    ):
        raise ValueError("Need an ordered finite bracket, 0 < target < 2, and positive tolerance")
    history = []

    def evaluate(mu):
        density, error = map(float, evaluate_density(mu))
        if not numpy.isfinite([density, error]).all() or error < 0:
            raise ValueError("Pilot returned invalid density or uncertainty")
        history.append(dict(mu=mu, density=density, error=error))
        return density, error

    nlo, elo = evaluate(lo)
    nhi, ehi = evaluate(hi)
    for mu, density, error in ((lo, nlo, elo), (hi, nhi, ehi)):
        if abs(density - target) + 2 * error <= tolerance:
            return mu, history
    if nlo - 2 * elo < target or nhi + 2 * ehi > target:
        raise ValueError(
            "Pilot densities do not confidently bracket target; widen mu bracket or improve statistics"
        )
    for _ in range(max_iter):
        mu = (lo + hi) / 2
        density, error = evaluate(mu)
        residual = density - target
        if abs(residual) + 2 * error <= tolerance:
            return mu, history
        if abs(residual) <= 2 * error:
            raise RuntimeError("Density pilot is too noisy; increase pilot blocks/walkers")
        if residual > 0:
            lo = mu
        else:
            hi = mu
    raise RuntimeError("Chemical-potential tuning did not converge")
