# Copyright 2022 The ipie Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Independent correlated AFQMC driver for paired systems A and B."""

import h5py
import numpy
import time
import math
import json
import uuid
from dataclasses import replace
from typing import Dict, Optional, Tuple

from ipie.config import MPI
from ipie.estimators.handler import EstimatorHandler
from ipie.estimators.estimator_base import EstimatorBase
from ipie.estimators.energy import CorrelatedEnergyEstimator
from ipie.propagation.propagator import Propagator
from ipie.qmc.options import QMCParams
from ipie.qmc.utils import set_rng_seed
from ipie.propagation.correlated_propagator import CorrelatedPropagator
from ipie.systems.generic import Generic
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import get_host_memory, synchronize
from ipie.utils.io import to_json
from ipie.utils.misc import get_git_info, print_env_info
from ipie.utils.mpi import MPIHandler
from ipie.walkers.base_walkers import WalkerAccumulator
from ipie.walkers.correlated_walkers import CorrelatedWalkers
from ipie.walkers.pop_controller import PopController
from ipie.walkers.walkers_dispatch import UHFWalkersTrial, get_initial_walker


class CorrelatedAFQMC:
    """Driver for correlated paired-system propagation.

    This driver is independent from AFQMC/AFQMCBase and is built around:
    - CorrelatedWalkers: wrapper for (walkers_a, walkers_b)
    - CorrelatedPropagator: wrapper for (propagator_a, propagator_b)

    Channel A and B are propagated separately in each step. Aggregate walker
    weight and overlap are maintained as products of per-channel quantities.
    For correlated sampling, both channels are propagated with the same
    timestep sign.
    """

    def __init__(
        self,
        systemA,
        systemB,
        hamiltonianA,
        hamiltonianB,
        trialA,
        trialB,
        walkers: CorrelatedWalkers,
        propagator: CorrelatedPropagator,
        mpi_handler,
        params: QMCParams,
        eq_propagator: Optional[CorrelatedPropagator] = None,
        verbose: int = 0,
    ):
        if hamiltonianB is None:
            hamiltonianB = hamiltonianA

        self.systemA = systemA
        self.systemB = systemB
        self.hamiltonianA = hamiltonianA
        self.hamiltonianB = hamiltonianB
        self.trialA = trialA
        self.trialB = trialB
        self.mpi_handler = mpi_handler
        self.shared_comm = self.mpi_handler.shared_comm
        self.verbose = verbose
        self.verbosity = int(verbose)
        self._init_time = time.time()

        self.paramsA = replace(
            params,
            correlated_samp=False,
            reference_run=False,
            walkermap_filepath=None,
        )
        self.paramsB = replace(
            params,
            correlated_samp=False,
            reference_run=False,
            walkermap_filepath=None,
        )
        self.params = self.paramsA

        self._parallel_rng_seed = set_rng_seed(self.params.rng_seed, self.mpi_handler.comm)

        if not isinstance(walkers, CorrelatedWalkers):
            raise TypeError("walkers must be a CorrelatedWalkers instance.")
        if not isinstance(propagator, CorrelatedPropagator):
            raise TypeError("propagator must be a CorrelatedPropagator instance.")
        if eq_propagator is not None and not isinstance(eq_propagator, CorrelatedPropagator):
            raise TypeError("eq_propagator must be a CorrelatedPropagator instance or None.")

        self.walkers = walkers
        self.propagator = propagator
        self.eq_propagator = eq_propagator if eq_propagator is not None else propagator

        self.eshiftA = 0.0
        self.eshiftB = 0.0
        self.walker_weights_filename = None
        self.setup_timers()

    @staticmethod
    def _to_host_array(values):
        if isinstance(values, numpy.ndarray):
            return numpy.asarray(values.real, dtype=numpy.float64)
        if hasattr(values, "get"):
            return numpy.asarray(values.get().real, dtype=numpy.float64)
        if hasattr(xp, "asnumpy"):
            return numpy.asarray(xp.asnumpy(values).real, dtype=numpy.float64)
        return numpy.asarray(values.real, dtype=numpy.float64)

    def _initialize_weight_dump(self, filename):
        self.walker_weights_filename = filename
        if self.mpi_handler.rank != 0:
            return

        with h5py.File(filename, "w") as fh5:
            fh5["num_blocks"] = self.params.num_blocks
            fh5["num_walkers_per_rank"] = self.params.num_walkers
            fh5["num_ranks"] = self.mpi_handler.size
            fh5["num_walkers_total"] = self.params.total_num_walkers
            fh5.create_dataset(
                "weight",
                shape=(self.params.total_num_walkers, self.params.num_blocks),
                dtype=numpy.float64,
            )
            fh5.create_dataset(
                "weight_A",
                shape=(self.params.total_num_walkers, self.params.num_blocks),
                dtype=numpy.float64,
            )
            fh5.create_dataset(
                "weight_B",
                shape=(self.params.total_num_walkers, self.params.num_blocks),
                dtype=numpy.float64,
            )

    def _dump_block_weights(self, block):
        if self.walker_weights_filename is None:
            return

        comm = self.mpi_handler.comm
        local_weight = self._to_host_array(self.walkers.weight)
        local_weight_a = self._to_host_array(self.walkers.walkers_A.weight)
        local_weight_b = self._to_host_array(self.walkers.walkers_B.weight)

        gathered_weight = None
        gathered_weight_a = None
        gathered_weight_b = None
        if comm.rank == 0:
            gathered_weight = numpy.empty((comm.size, self.walkers.nwalkers), dtype=local_weight.dtype)
            gathered_weight_a = numpy.empty(
                (comm.size, self.walkers.nwalkers), dtype=local_weight_a.dtype
            )
            gathered_weight_b = numpy.empty(
                (comm.size, self.walkers.nwalkers), dtype=local_weight_b.dtype
            )

        comm.Gather(local_weight, gathered_weight, root=0)
        comm.Gather(local_weight_a, gathered_weight_a, root=0)
        comm.Gather(local_weight_b, gathered_weight_b, root=0)

        if comm.rank == 0:
            with h5py.File(self.walker_weights_filename, "r+") as fh5:
                fh5["weight"][:, block] = numpy.asarray(
                    gathered_weight.reshape(-1), dtype=numpy.float64
                )
                fh5["weight_A"][:, block] = numpy.asarray(
                    gathered_weight_a.reshape(-1), dtype=numpy.float64
                )
                fh5["weight_B"][:, block] = numpy.asarray(
                    gathered_weight_b.reshape(-1), dtype=numpy.float64
                )

    @staticmethod
    def build(
        num_elec: Tuple[int, int],
        hamiltonianA,
        trial_wavefunctionA,
        num_elecB: Optional[Tuple[int, int]] = None,
        hamiltonianB=None,
        trial_wavefunctionB=None,
        walkers: Optional[CorrelatedWalkers] = None,
        num_walkers: int = 100,
        seed: Optional[int] = None,
        num_steps_per_block: int = 25,
        num_blocks: int = 100,
        timestep: float = 0.005,
        stabilize_freq=5,
        eq_stabilize_freq=2,
        pop_control_method="stochastic_reconfiguration_independent_repairing",
        pop_control_freq=5,
        eq_pop_control_freq=2,
        eq_timestep=None,
        eq_num_steps_per_block=None,
        num_eq_blocks: int = 0,
        ene_bound_const: float = 2.0,
        fb_bound: float = 1.0,
        verbose=True,
        mpi_handler=None,
    ) -> "CorrelatedAFQMC":
        """Factory method to build a correlated AFQMC driver.
        """
        if mpi_handler is None:
            mpi_handler = MPIHandler()
            comm = mpi_handler.comm
        else:
            comm = mpi_handler.comm

        if hamiltonianB is None:
            hamiltonianB = hamiltonianA
        if trial_wavefunctionB is None:
            trial_wavefunctionB = trial_wavefunctionA
        if num_elecB is None:
            num_elecB = num_elec

        params = QMCParams(
            num_walkers=num_walkers,
            total_num_walkers=num_walkers * comm.size,
            num_blocks=num_blocks,
            num_steps_per_block=num_steps_per_block,
            timestep=timestep,
            num_stblz=stabilize_freq,
            pop_control_method=pop_control_method,
            num_eq_stblz=eq_stabilize_freq,
            pop_control_freq=pop_control_freq,
            eq_pop_control_freq=eq_pop_control_freq,
            rng_seed=seed,
            eq_timestep=eq_timestep,
            eq_num_steps_per_block=eq_num_steps_per_block,
            num_eq_blocks=num_eq_blocks,
            fb_bound=fb_bound,
            ene_bound_const=ene_bound_const,
            correlated_samp=False,
            reference_run=False,
            walkermap_filepath=None,
        )

        systemA = Generic(num_elec)
        systemB = Generic(num_elecB)

        if trial_wavefunctionA.compute_trial_energy:
            trial_wavefunctionA.calculate_energy(systemA, hamiltonianA)
            trial_wavefunctionA.e1b = comm.bcast(trial_wavefunctionA.e1b, root=0)
            trial_wavefunctionA.e2b = comm.bcast(trial_wavefunctionA.e2b, root=0)
        if trial_wavefunctionB is not trial_wavefunctionA and trial_wavefunctionB.compute_trial_energy:
            trial_wavefunctionB.calculate_energy(systemB, hamiltonianB)
            trial_wavefunctionB.e1b = comm.bcast(trial_wavefunctionB.e1b, root=0)
            trial_wavefunctionB.e2b = comm.bcast(trial_wavefunctionB.e2b, root=0)
        comm.barrier()

        if walkers is None:
            _, initial_walker_A = get_initial_walker(trial_wavefunctionA)
            walkersA = UHFWalkersTrial(
                trial_wavefunctionA,
                initial_walker_A,
                systemA.nup,
                systemA.ndown,
                hamiltonianA.nbasis,
                num_walkers,
                mpi_handler,
            )
            walkersA.build(trial_wavefunctionA)

            _, initial_walker_B = get_initial_walker(trial_wavefunctionB)
            walkersB = UHFWalkersTrial(
                trial_wavefunctionB,
                initial_walker_B,
                systemB.nup,
                systemB.ndown,
                hamiltonianB.nbasis,
                num_walkers,
                mpi_handler,
            )
            walkersB.build(trial_wavefunctionB)
            walkers = CorrelatedWalkers(walkersA, walkersB)
        elif not isinstance(walkers, CorrelatedWalkers):
            raise TypeError("walkers must be a CorrelatedWalkers instance or None.")

        propagatorA = Propagator[type(hamiltonianA)](
            params.timestep,
            params.ene_bound_const,
            params.fb_bound,
        )
        propagatorA.build(
            hamiltonianA,
            trial_wavefunctionA,
            walkers.walkers_A,
            mpi_handler,
        )
        propagatorB = Propagator[type(hamiltonianB)](
            params.timestep,
            params.ene_bound_const,
            params.fb_bound,
        )
        propagatorB.build(
            hamiltonianB,
            trial_wavefunctionB,
            walkers.walkers_B,
            mpi_handler,
        )
        propagator = CorrelatedPropagator(propagatorA, propagatorB)

        if not math.isclose(params.timestep, params.eq_timestep, rel_tol=1e-8):
            eq_propagatorA = Propagator[type(hamiltonianA)](
                params.eq_timestep,
                params.ene_bound_const,
                params.fb_bound,
            )
            eq_propagatorA.build(
                hamiltonianA,
                trial_wavefunctionA,
                walkers.walkers_A,
                mpi_handler,
            )
            eq_propagatorB = Propagator[type(hamiltonianB)](
                params.eq_timestep,
                params.ene_bound_const,
                params.fb_bound,
            )
            eq_propagatorB.build(
                hamiltonianB,
                trial_wavefunctionB,
                walkers.walkers_B,
                mpi_handler,
            )
            eq_propagator = CorrelatedPropagator(eq_propagatorA, eq_propagatorB)
        else:
            eq_propagator = propagator

        return CorrelatedAFQMC(
            systemA=systemA,
            systemB=systemB,
            hamiltonianA=hamiltonianA,
            hamiltonianB=hamiltonianB,
            trialA=trial_wavefunctionA,
            trialB=trial_wavefunctionB,
            walkers=walkers,
            propagator=propagator,
            mpi_handler=mpi_handler,
            params=params,
            eq_propagator=eq_propagator,
            verbose=(verbose and comm.rank == 0),
        )

    def setup_timers(self):
        # Keep timer fields aligned with AFQMCBase.finalise formatting.
        self.tsetup = 0.0
        self.tortho = 0.0
        self.tprop = 0.0

        self.tprop_fbias = 0.0
        self.tprop_ovlp = 0.0
        self.tprop_update = 0.0
        self.tprop_gf = 0.0
        self.tprop_vhs = 0.0
        self.tprop_gemm = 0.0
        self.tprop_clip = 0.0
        self.tprop_barrier = 0.0

        self.testim = 0.0
        self.tpopc = 0.0
        self.tpopc_comm = 0.0
        self.tpopc_non_comm = 0.0
        self.tstep = 0.0

    def copy_to_gpu(self):
        if hasattr(self.propagator, "cast_to_cupy"):
            self.propagator.cast_to_cupy(verbose=self.verbose)
        if hasattr(self.eq_propagator, "cast_to_cupy"):
            self.eq_propagator.cast_to_cupy(verbose=self.verbose)
        if hasattr(self.hamiltonianA, "cast_to_cupy"):
            self.hamiltonianA.cast_to_cupy(self.verbose)
        if hasattr(self.hamiltonianB, "cast_to_cupy"):
            self.hamiltonianB.cast_to_cupy(self.verbose)
        if hasattr(self.trialA, "cast_to_cupy"):
            self.trialA.cast_to_cupy(self.verbose)
        if hasattr(self.trialB, "cast_to_cupy"):
            self.trialB.cast_to_cupy(self.verbose)
        self.walkers.cast_to_cupy(verbose=self.verbose)

    def get_env_info(self):
        this_uuid = str(uuid.uuid1())
        try:
            sha1, branch, local_mods = get_git_info()
        except Exception:
            sha1 = "None"
            branch = "None"
            local_mods = []
        if self.verbose:
            self.sys_info = print_env_info(
                sha1, branch, local_mods, this_uuid, self.mpi_handler.size
            )
            mem_avail = get_host_memory()
            print(f"# MPI communicator : {type(self.mpi_handler.comm)}")
            print(f"# Available memory on the node is {mem_avail:4.3f} GB")

    def setup_estimators(
        self, filename, additional_estimators: Optional[Dict[str, EstimatorBase]] = None
    ):
        self.accumulators = WalkerAccumulator(
            ["Weight", "WeightFactor", "HybridEnergy"], self.params.num_steps_per_block
        )
        self.accumulatorsA = WalkerAccumulator(
            ["Weight", "WeightFactor", "HybridEnergy"], self.params.num_steps_per_block
        )
        self.accumulatorsB = WalkerAccumulator(
            ["Weight", "WeightFactor", "HybridEnergy"], self.params.num_steps_per_block
        )
        comm = self.mpi_handler.comm
        # Correlated driver does not register single-system predefined estimators.
        # Additional correlated estimators can be attached via additional_estimators.
        self.estimators = EstimatorHandler(
            self.mpi_handler.comm,
            (self.systemA, self.systemB),
            (self.hamiltonianA, self.hamiltonianB),
            (self.trialA, self.trialB),
            walker_state=self.accumulators,
            verbose=(comm.rank == 0 and self.verbose),
            filename=filename,
            observables=(),
        )
        # Register default correlated energy-difference estimator so block output
        # includes EDiff / E1BodyDiff / E2BodyDiff without extra script wiring.
        self.estimators["corr_energy"] = CorrelatedEnergyEstimator()
        if additional_estimators is not None:
            for k, v in additional_estimators.items():
                self.estimators[k] = v

        json.encoder.FLOAT_REPR = lambda o: format(o, ".6f")
        json_string = to_json(self)
        self.estimators.json_string = json_string

        self.estimators.initialize(comm)
        self.estimators.compute_estimators(
            (self.systemA, self.systemB),
            (self.hamiltonianA, self.hamiltonianB),
            (self.trialA, self.trialB),
            self.walkers,
        )
        self.accumulators.update(self.walkers)
        self.accumulatorsA.update(self.walkers.walkers_A)
        self.accumulatorsB.update(self.walkers.walkers_B)
        self.estimators.print_block(comm, 0, self.accumulators)
        self.accumulators.zero()
        self.accumulatorsA.zero()
        self.accumulatorsB.zero()

    def _compute_channel_eshift(self, accumulator):
        comm = self.mpi_handler.comm
        local_vals = accumulator.buffer.copy()
        global_vals = xp.zeros_like(local_vals)
        comm.Reduce(local_vals, global_vals, op=MPI.SUM, root=0)

        if comm.rank == 0:
            weight = global_vals[accumulator.get_index("Weight")]
            hybrid = global_vals[accumulator.get_index("HybridEnergy")]
            shift = hybrid / weight if abs(weight) > 1e-16 else 0.0
        else:
            shift = None

        return comm.bcast(shift, root=0).real

    def run(
        self,
        walkers=None,
        estimator_filename=None,
        verbose=True,
        discard_weights_aftereq=False,
        walker_weights_filename=None,
        additional_estimators: Optional[Dict[str, EstimatorBase]] = None,
    ):
        """Perform correlated AFQMC simulation using open-ended random walk."""
        self.setup_timers()
        tzero_setup = time.time()

        if walkers is not None:
            if isinstance(walkers, CorrelatedWalkers):
                self.walkers = walkers
            elif isinstance(walkers, (tuple, list)) and len(walkers) == 2:
                self.walkers = CorrelatedWalkers(walkers[0], walkers[1])
            else:
                raise ValueError(
                    "walkers must be CorrelatedWalkers or a 2-tuple/list (walkersA, walkersB)."
                )

        self.setup_timers()
        eshiftA = 0.0
        eshiftB = 0.0
        self.walkers.orthogonalise()

        self.pcontrol_eq = PopController(
            self.params.num_walkers,
            self.params.num_steps_per_block,
            self.mpi_handler,
            pop_control_method=self.params.pop_control_method,
            verbose=self.verbose,
        )

        self.pcontrol = PopController(
            self.params.num_walkers,
            self.params.num_steps_per_block,
            self.mpi_handler,
            pop_control_method=self.params.pop_control_method,
            verbose=self.verbose,
        )

        self.get_env_info()
        self.copy_to_gpu()
        self.setup_estimators(estimator_filename, additional_estimators=additional_estimators)
        if walker_weights_filename is not None:
            self._initialize_weight_dump(walker_weights_filename)

        num_eqlb_steps = self.params.num_eq_blocks * self.params.eq_num_steps_per_block
        total_steps = self.params.num_steps_per_block * self.params.num_blocks + num_eqlb_steps

        synchronize()
        comm = self.mpi_handler.comm
        self.tsetup += time.time() - tzero_setup

        for step in range(1, total_steps + 1):
            synchronize()
            start_step = time.time()

            if step <= num_eqlb_steps:
                if step % self.params.num_eq_stblz == 0:
                    start = time.time()
                    self.walkers.orthogonalise()
                    synchronize()
                    self.tortho += time.time() - start
            else:
                if step % self.params.num_stblz == 0:
                    start = time.time()
                    self.walkers.orthogonalise()
                    synchronize()
                    self.tortho += time.time() - start

            start = time.time()
            prop = self.eq_propagator if step <= num_eqlb_steps else self.propagator

            if discard_weights_aftereq and step == num_eqlb_steps + 1:
                self.walkers.weight.fill(1.0)

            prop.propagate_walkers(
                self.walkers,
                self.hamiltonianA,
                self.hamiltonianB,
                self.trialA,
                self.trialB,
                eshiftA,
                eshiftB,
            )

            timer_a = prop.timer_a
            timer_b = prop.timer_b
            if timer_a is not None:
                self.tprop_fbias += getattr(timer_a, "tfbias", 0.0)
                self.tprop_ovlp += getattr(timer_a, "tovlp", 0.0)
                self.tprop_update += getattr(timer_a, "tupdate", 0.0)
                self.tprop_gf += getattr(timer_a, "tgf", 0.0)
                self.tprop_vhs += getattr(timer_a, "tvhs", 0.0)
                self.tprop_gemm += getattr(timer_a, "tgemm", 0.0)
            if timer_b is not None:
                self.tprop_fbias += getattr(timer_b, "tfbias", 0.0)
                self.tprop_ovlp += getattr(timer_b, "tovlp", 0.0)
                self.tprop_update += getattr(timer_b, "tupdate", 0.0)
                self.tprop_gf += getattr(timer_b, "tgf", 0.0)
                self.tprop_vhs += getattr(timer_b, "tvhs", 0.0)
                self.tprop_gemm += getattr(timer_b, "tgemm", 0.0)

            start_clip = time.time()
            if step > 1 and step <= num_eqlb_steps:
                wbound = self.pcontrol_eq.total_weight * 0.10
                xp.nan_to_num(self.walkers.walkers_A.weight, copy=False)
                xp.clip(self.walkers.walkers_A.weight, a_min=-wbound, a_max=wbound, out=self.walkers.walkers_A.weight)
                xp.nan_to_num(self.walkers.walkers_B.weight, copy=False)
                xp.clip(self.walkers.walkers_B.weight, a_min=-wbound, a_max=wbound, out=self.walkers.walkers_B.weight)
                self.walkers.sync_combined_state()
            elif step > num_eqlb_steps and step > 1:
                wbound = self.pcontrol.total_weight * 0.10
                xp.nan_to_num(self.walkers.walkers_A.weight, copy=False)
                xp.clip(self.walkers.walkers_A.weight, a_min=-wbound, a_max=wbound, out=self.walkers.walkers_A.weight)
                xp.nan_to_num(self.walkers.walkers_B.weight, copy=False)
                xp.clip(self.walkers.walkers_B.weight, a_min=-wbound, a_max=wbound, out=self.walkers.walkers_B.weight)
                self.walkers.sync_combined_state()
            synchronize()
            self.tprop_clip += time.time() - start_clip

            start_barrier = time.time()
            if step % self.params.pop_control_freq == 0:
                comm.Barrier()
            self.tprop_barrier += time.time() - start_barrier

            self.tprop += time.time() - start

            if step <= num_eqlb_steps:
                if step % self.params.eq_pop_control_freq == 0:
                    start = time.time()
                    self.pcontrol_eq.pop_control(self.walkers, comm)
                    synchronize()
                    self.tpopc += time.time() - start
                    self.tpopc_comm = self.pcontrol_eq.timer.communication_time
                    self.tpopc_non_comm = self.pcontrol_eq.timer.non_communication_time
            else:
                if step % self.params.pop_control_freq == 0:
                    start = time.time()
                    self.pcontrol.pop_control(self.walkers, comm)
                    synchronize()
                    self.tpopc += time.time() - start
                    self.tpopc_comm = self.pcontrol.timer.communication_time
                    self.tpopc_non_comm = self.pcontrol.timer.non_communication_time

            start = time.time()
            self.accumulators.update(self.walkers)
            self.accumulatorsA.update(self.walkers.walkers_A)
            self.accumulatorsB.update(self.walkers.walkers_B)
            synchronize()
            self.testim += time.time() - start

            start = time.time()
            if step > num_eqlb_steps:
                if step % self.params.num_steps_per_block == 0:
                    block = (step - num_eqlb_steps) // self.params.num_steps_per_block - 1
                    self._dump_block_weights(block)
                    self.estimators.compute_estimators(
                        (self.systemA, self.systemB),
                        (self.hamiltonianA, self.hamiltonianB),
                        (self.trialA, self.trialB),
                        self.walkers,
                    )
                    self.estimators.print_block(
                        comm,
                        (step - num_eqlb_steps) // self.params.num_steps_per_block,
                        self.accumulators,
                    )
                    self.eshiftA = self._compute_channel_eshift(self.accumulatorsA)
                    self.eshiftB = self._compute_channel_eshift(self.accumulatorsB)
                    self.accumulators.zero()
                    self.accumulatorsA.zero()
                    self.accumulatorsB.zero()
            else:
                if step % self.params.eq_num_steps_per_block == 0:
                    self.estimators.compute_estimators(
                        (self.systemA, self.systemB),
                        (self.hamiltonianA, self.hamiltonianB),
                        (self.trialA, self.trialB),
                        self.walkers,
                    )
                    self.estimators.print_block(
                        comm,
                        step // self.params.eq_num_steps_per_block,
                        self.accumulators,
                    )
                    self.eshiftA = self._compute_channel_eshift(self.accumulatorsA)
                    self.eshiftB = self._compute_channel_eshift(self.accumulatorsB)
                    self.accumulators.zero()
                    self.accumulatorsA.zero()
                    self.accumulatorsB.zero()
            synchronize()
            self.testim += time.time() - start

            if self.walkers.write_restart:
                if self.walkers.write_freq is not None:
                    if step % self.walkers.write_freq == 0:
                        self.walkers.write_walkers_batch(comm)
                else:
                    assert self.walkers.write_time is not None
                    if step == self.walkers.write_time:
                        self.walkers.write_walkers_batch(comm)

            if step < num_eqlb_steps:
                eshiftA = self.eshiftA
                eshiftB = self.eshiftB
            else:
                eshiftA += self.eshiftA - eshiftA
                eshiftB += self.eshiftB - eshiftB

            synchronize()
            self.tstep += time.time() - start_step

    def finalise(self, verbose=False):
        """Tidy up.

        Parameters
        ----------
        verbose : bool
            If true print out some information to stdout.
        """
        nsteps = max(self.params.num_steps_per_block, 1)
        nblocks = max(self.params.num_blocks, 1)
        nstblz = max(nsteps // self.params.num_stblz, 1)
        npcon = max(nsteps // self.params.pop_control_freq, 1)
        if self.mpi_handler.rank == 0:
            if verbose:
                print(f"# End Time: {time.asctime():s}")
                print(f"# Running time : {time.time() - self._init_time:.6f} seconds")
                print("# Timing breakdown (per call, total calls per block, total blocks):")
                print(f"# - Setup: {self.tsetup:.6f} s")
                print(
                    "# - Block: {:.6f} s / block for {} total blocks".format(
                        self.tstep / (nblocks), nblocks
                    )
                )
                print(
                    "# - Propagation: {:.6f} s / call for {} call(s) in each of {} blocks".format(
                        self.tprop / (nblocks * nsteps), nsteps, nblocks
                    )
                )
                print(
                    "#     -       Force bias: {:.6f} s / call for {} call(s) in each of {} blocks".format(
                        self.tprop_fbias / (nblocks * nsteps), nsteps, nblocks
                    )
                )
                print(
                    "#     -              VHS: {:.6f} s / call for {} call(s) in each of {} blocks".format(
                        self.tprop_vhs / (nblocks * nsteps), nsteps, nblocks
                    )
                )
                print(
                    "#     - Green's Function: {:.6f} s / call for {} call(s) in each of {} blocks".format(
                        self.tprop_gf / (nblocks * nsteps), nsteps, nblocks
                    )
                )
                print(
                    "#     -          Overlap: {:.6f} s / call for {} call(s) in each of {} blocks".format(
                        self.tprop_ovlp / (nblocks * nsteps), nsteps, nblocks
                    )
                )
                print(
                    "#     -   Weights Update: {:.6f} s / call for {} call(s) in each of {} blocks".format(
                        (self.tprop_update + self.tprop_clip) / (nblocks * nsteps), nsteps, nblocks
                    )
                )
                print(
                    "#     -  GEMM operations: {:.6f} s / call for {} call(s) in each of {} blocks".format(
                        self.tprop_gemm / (nblocks * nsteps), nsteps, nblocks
                    )
                )
                print(
                    "#     -          Barrier: {:.6f} s / call for {} call(s) in each of {} blocks".format(
                        self.tprop_barrier / (nblocks * nsteps), nsteps, nblocks
                    )
                )
                print(
                    "# - Estimators: {:.6f} s / call for {} call(s)".format(
                        self.testim / nblocks, nblocks
                    )
                )
                print(
                    "# - Orthogonalisation: {:.6f} s / call for {} call(s) in each of {} blocks".format(
                        self.tortho / (nstblz * nblocks), nstblz, nblocks
                    )
                )
                print(
                    "# - Population control: {:.6f} s / call for {} call(s) in each of {} blocks".format(
                        self.tpopc / (npcon * nblocks), npcon, nblocks
                    )
                )
                print(
                    "#       -     Commnication: {:.6f} s / call for {} call(s) in each of {} blocks".format(
                        self.tpopc_comm / (npcon * nblocks), npcon, nblocks
                    )
                )
                print(
                    "#       - Non-Commnication: {:.6f} s / call for {} call(s) in each of {} blocks".format(
                        self.tpopc_non_comm / (npcon * nblocks), npcon, nblocks
                    )
                )
