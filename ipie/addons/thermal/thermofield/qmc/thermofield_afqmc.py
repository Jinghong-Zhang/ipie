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
"""Driver to perform thermofield AFQMC calculation"""

import json
import time
from types import SimpleNamespace
from typing import Dict, Optional

import numpy

from ipie.addons.thermal.thermofield.estimators.handler import ThermofieldEstimatorHandler
from ipie.addons.thermal.thermofield.hamiltonians.ueg import UEGSparseChol
from ipie.addons.thermal.thermofield.propagation.propagator import Propagator
from ipie.addons.thermal.thermofield.qmc.options import ThermofieldQMCParams
from ipie.addons.thermal.thermofield.trial.thermofield import ThermofieldThermalTrial
from ipie.addons.thermal.thermofield.walkers.pop_controller import ThermofieldPopController
from ipie.addons.thermal.thermofield.walkers.uhf_walkers import ThermofieldWalkers
from ipie.config import config, MPI
from ipie.estimators.estimator_base import EstimatorBase
from ipie.qmc.afqmc import AFQMCBase
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import synchronize
from ipie.utils.io import to_json
from ipie.utils.mpi import MPIHandler
from ipie.walkers.base_walkers import WalkerAccumulator


class ThermofieldAFQMC(AFQMCBase):
    r"""Thermofield AFQMC driver.

    Every block is one walk of the whole population from the infinite
    temperature purification :math:`\Delta = I` over :math:`[0, \beta]`. At
    :math:`\theta = \beta - \eta` the mixed estimator with the trial at time
    eta is measured for every member, and at the end of the walk the closed
    trace. Blocks are independent; combine them as a pooled ratio of means
    with :func:`ipie.addons.thermal.thermofield.analysis.thermofield_analysis.summarize`.

    Parameters
    ----------
    hamiltonian :
        Hamiltonian describing the system.
    trial : :class:`ThermofieldThermalTrial`
        Guide and trial.
    walkers : :class:`ThermofieldWalkers`
        Walkers.
    propagator : :class:`ThermofieldPhaseless`
        Propagator.
    mpi_handler : :class:`ipie.utils.mpi.MPIHandler`
        MPI handler.
    params : :class:`ThermofieldQMCParams`
        Parameters of simulation.
    verbose : int
        How much information to print.

    Attributes
    ----------
    eta_members : :class:`numpy.ndarray`
        Trial times of the members, largest first.
    nslices : int
        Number of time slices in beta.
    member_slices : :class:`numpy.ndarray`
        Slice after which every member is measured.
    nwalk : int
        Number of slices propagated per block.
    """

    def __init__(
        self,
        hamiltonian,
        trial,
        walkers,
        propagator,
        mpi_handler,
        params: ThermofieldQMCParams,
        verbose: int = 0,
    ):
        super().__init__(
            None, hamiltonian, trial, walkers, propagator, mpi_handler, params, verbose
        )
        beta, dt = params.beta, params.timestep
        self.nslices = self._num_slices(beta / dt, "beta / timestep")
        self.eta_members = numpy.array(params.eta_members, dtype=float)
        # Slice after which each member is measured: theta = beta - eta.
        self.member_slices = numpy.array(
            [
                self.nslices - self._num_slices(eta / dt, f"eta = {eta} / timestep")
                for eta in self.eta_members
            ],
            dtype=int,
        )
        if numpy.any(self.member_slices < 1) or numpy.any(self.member_slices > self.nslices):
            raise ValueError("eta members must satisfy 0 <= eta < beta.")
        # The trace needs the whole walk; members alone end at the last one.
        self.nwalk = self.nslices if params.measure_trace else int(self.member_slices.max())
        # On the GPU set_rng_seed seeds the device stream only; population
        # control draws from the host stream.
        if config.get_option("use_gpu"):
            numpy.random.seed(self._parallel_rng_seed)

    @staticmethod
    def _num_slices(x, what):
        n = int(round(x))
        if abs(n - x) > 1e-8 * max(1.0, abs(x)):
            raise ValueError(f"{what} must be a whole number of slices (got {x}).")
        return n

    @staticmethod
    def build(
        mu: float,
        beta: float,
        hamiltonian,
        trial: Optional[ThermofieldThermalTrial] = None,
        nwalkers: int = 32,
        nblocks: int = 10,
        timestep: float = 0.01,
        seed: Optional[int] = None,
        stabilize_freq: int = 5,
        pop_control_freq: int = 5,
        pop_control_method: str = "pair_branch",
        phaseless: bool = True,
        eta_members=None,
        measure_trace: bool = True,
        weight_cap: float = 0.10,
        mpi_handler=None,
        verbose: int = 0,
    ) -> "ThermofieldAFQMC":
        r"""Factory method to build thermofield AFQMC driver from hamiltonian and trial.

        Parameters
        ----------
        mu : float
            Chemical potential.
        beta : float
            Inverse temperature.
        hamiltonian :
            Hamiltonian describing the system.
        trial : :class:`ThermofieldThermalTrial`
            Guide and trial. Default the noninteracting trial
            :math:`k_T = H_1 - \mu` with guide time beta / 2.
        nwalkers : int
            Number of walkers per MPI process. Default 32.
        nblocks : int
            Number of blocks (walks). Default 10.
        timestep : float
            Imaginary timestep. Default 0.01.
        seed : int
            Random number seed. Default None: drawn on the root process.
        stabilize_freq : int
            Frequency of the QR refactorization of the walkers (in time slices).
            Default 5.
        pop_control_freq : int
            Frequency of population control (in time slices). Default 5.
        pop_control_method : str
            Population control method. Default "pair_branch".
        phaseless : bool
            Phaseless walk; False is free projection. Default True.
        eta_members : tuple
            Trial times of the mixed estimators. Default (beta/2, beta/4, beta/8).
        measure_trace : bool
            Measure the closed trace at the end of the walk. Default True.
        weight_cap : float
            Weight cap as a fraction of the total weight. Default 0.10.
        verbose : int
            Log verbosity. Default 0.
        """
        if mpi_handler is None:
            mpi_handler = MPIHandler()
        comm = mpi_handler.comm

        # pylint: disable = no-value-for-parameter
        params = ThermofieldQMCParams(
            mu=mu,
            beta=beta,
            num_walkers=nwalkers,
            total_num_walkers=nwalkers * comm.size,
            num_blocks=nblocks,
            timestep=timestep,
            num_stblz=stabilize_freq,
            pop_control_freq=pop_control_freq,
            pop_control_method=pop_control_method,
            rng_seed=seed,
            eta_members=eta_members,
            measure_trace=measure_trace,
            weight_cap=weight_cap,
        )

        if trial is None:
            trial = ThermofieldThermalTrial(hamiltonian, beta, mu=mu, verbose=verbose)

        walkers = ThermofieldWalkers(
            trial, hamiltonian.nbasis, nwalkers, mpi_handler=mpi_handler, verbose=verbose
        )
        propagator = Propagator[type(hamiltonian)](
            timestep, mu, phaseless=phaseless, verbose=verbose
        )
        propagator.build(
            hamiltonian, trial=trial, walkers=walkers, mpi_handler=mpi_handler, verbose=verbose
        )
        return ThermofieldAFQMC(
            hamiltonian, trial, walkers, propagator, mpi_handler, params, verbose=verbose
        )

    def run(
        self,
        walkers=None,
        estimator_filename=None,
        verbose: bool = True,
        discard_weights_aftereq=False,
        additional_estimators: Optional[Dict[str, EstimatorBase]] = None,
    ):
        """Perform thermofield AFQMC simulation.

        Parameters
        ----------
        walkers : :class:`ThermofieldWalkers`
            Walkers. Default the driver's walkers.
        estimator_filename : str
            .h5 file to write the block estimates to. Default None: no file.
        additional_estimators : dict
            Dictionary of additional estimators, evaluated at the end of every walk.

        Returns
        -------
        results : dict
            Per-block sums over all processes: ``eta`` (members),
            ``numer_mix``, ``navnum_mix``, ``denom_mix`` (complex, shape
            (nblocks, nmembers)), ``log_scale_mix`` (shift plus population
            normalization), the same with ``_trace`` (shape (nblocks,)) when the
            trace is measured, and the block ratios ``e_mix``, ``nav_mix``,
            ``e_trace``, ``nav_trace``. Combine blocks with `summarize`, not by
            averaging the ratios.
        """
        # Setup.
        self.setup_timers()
        ft_setup = time.time()
        if walkers is not None:
            self.walkers = walkers
        if config.get_option("use_gpu") and not isinstance(self.hamiltonian, UEGSparseChol):
            raise NotImplementedError(
                "The thermofield GPU path supports the sparse plane-wave Hamiltonian only."
            )

        self.copy_to_gpu()
        self.pcontrol = ThermofieldPopController(
            self.params.num_walkers,
            self.nwalk,
            self.mpi_handler,
            self.params.pop_control_method,
            verbose=False,
        )
        self.get_env_info()
        self.setup_estimators(estimator_filename, additional_estimators=additional_estimators)

        synchronize()
        comm = self.mpi_handler.comm
        self.tsetup += time.time() - ft_setup

        nblocks = self.params.num_blocks
        nmembers = len(self.eta_members)
        phaseless = self.propagator.phaseless
        member_names = [f"eta={eta:g}" for eta in self.eta_members]
        at_slice = {}
        for k, s in enumerate(self.member_slices):
            at_slice.setdefault(int(s), []).append(k)
        results = {"eta": self.eta_members.copy()}
        for name in ("numer", "navnum", "denom"):
            results[f"{name}_mix"] = numpy.zeros((nblocks, nmembers), dtype=numpy.complex128)
            results[f"{name}_trace"] = numpy.zeros(nblocks, dtype=numpy.complex128)
        results["log_scale_mix"] = numpy.zeros((nblocks, nmembers))
        results["log_scale_trace"] = numpy.zeros(nblocks)

        # Propagate.
        for block in range(nblocks):
            synchronize()
            start_step = time.time()
            self.walkers.reset(self.trial)
            pop_log_norm = 0.0
            for t in range(self.nwalk):
                if self.verbosity >= 2 and comm.rank == 0:
                    print(f" # Timeslice {t} of {self.nwalk}.")

                start = time.time()
                self.propagator.propagate_walkers(self.walkers, self.hamiltonian, self.trial)
                self.tprop_fbias = self.propagator.timer.tfbias
                self.tprop_update = self.propagator.timer.tupdate
                self.tprop_vhs = self.propagator.timer.tvhs
                self.tprop_gemm = self.propagator.timer.tgemm
                self.tprop_ovlp = self.propagator.timer.tovlp

                start_clip = time.time()
                if t > 0:
                    numpy.nan_to_num(
                        self.walkers.weight, copy=False, nan=0.0, posinf=0.0, neginf=0.0
                    )
                if phaseless and t > 0 and self.params.weight_cap > 0.0:
                    wbound = self.pcontrol.total_weight * self.params.weight_cap
                    numpy.clip(
                        self.walkers.weight, a_min=-wbound, a_max=wbound, out=self.walkers.weight
                    )  # In-place clipping.
                synchronize()
                self.tprop_clip += time.time() - start_clip
                self.tprop += time.time() - start

                if t > 0 and (t + 1) % self.params.num_stblz == 0:
                    start = time.time()
                    self.walkers.stabilize()
                    synchronize()
                    self.tortho += time.time() - start

                start = time.time()
                for k in at_slice.get(t + 1, ()):
                    est = self.estimators[member_names[k]]
                    est.measure(
                        self.walkers, self.hamiltonian, self.trial, comm, log_norm=pop_log_norm
                    )
                    results["denom_mix"][block, k] = est.global_sums[0]
                    results["numer_mix"][block, k] = est.global_sums[1]
                    results["navnum_mix"][block, k] = est.global_sums[2]
                    results["log_scale_mix"][block, k] = est.log_scale
                self.testim += time.time() - start

                if phaseless and t > 0 and t % self.params.pop_control_freq == 0:
                    start = time.time()
                    pop_log_norm += self.pop_control()
                    synchronize()
                    self.tpopc += time.time() - start

            # Measure the trace and write the block.
            start = time.time()
            if self.params.measure_trace:
                est = self.estimators["trace"]
                est.measure(self.walkers, self.hamiltonian, self.trial, comm, log_norm=pop_log_norm)
                results["denom_trace"][block] = est.global_sums[0]
                results["numer_trace"][block] = est.global_sums[1]
                results["navnum_trace"][block] = est.global_sums[2]
                results["log_scale_trace"][block] = est.log_scale
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
            synchronize()
            self.testim += time.time() - start
            self.tstep += time.time() - start_step

        with numpy.errstate(divide="ignore", invalid="ignore"):
            for est in ("mix", "trace"):
                results[f"e_{est}"] = (results[f"numer_{est}"] / results[f"denom_{est}"]).real
                results[f"nav_{est}"] = (results[f"navnum_{est}"] / results[f"denom_{est}"]).real
        if not self.params.measure_trace:
            for key in ("numer", "navnum", "denom", "log_scale", "e", "nav"):
                del results[f"{key}_trace"]
        self.results = results
        return results

    # Compatibility with the per-array GPU interface of earlier versions.
    def to_device(self):
        """Copy the walkers, propagator, Hamiltonian and trial to the GPU. Use `copy_to_gpu`."""
        if not config.get_option("use_gpu"):
            raise RuntimeError("to_device needs the GPU backend: set IPIE_USE_GPU=1.")
        self.copy_to_gpu()
        return self

    def pop_control(self):
        """Population control of the walkers.

        Returns
        -------
        log_norm : float
            Log of the total weight over the target weight, the factor that
            population control removes.
        """
        comm = self.mpi_handler.comm
        log_norm = 0.0
        if self.pcontrol.ntot_walkers > 1:
            total = comm.allreduce(float(numpy.abs(self.walkers.weight).sum()), op=MPI.SUM)
            if not numpy.isfinite(total) or total <= 0.0:
                raise FloatingPointError("Walker population is extinct.")
            log_norm = numpy.log(total / self.pcontrol.target_weight)
        self.pcontrol.pop_control(self.walkers, comm)
        return log_norm

    def setup_estimators(
        self, filename, additional_estimators: Optional[Dict[str, EstimatorBase]] = None
    ):
        self.accumulators = WalkerAccumulator(
            ["Weight", "WeightFactor", "HybridEnergy"], self.params.num_steps_per_block
        )
        comm = self.mpi_handler.comm
        self.estimators = ThermofieldEstimatorHandler(
            comm,
            self.hamiltonian,
            self.trial,
            eta_members=self.eta_members,
            measure_trace=self.params.measure_trace,
            walker_state=self.accumulators,
            verbose=(comm.rank == 0 and self.verbose),
            filename=filename,
        )

        if additional_estimators is not None:
            for k, v in additional_estimators.items():
                self.estimators[k] = v

        json.encoder.FLOAT_REPR = lambda o: format(o, ".6f")
        json_string = to_json(self)
        self.estimators.json_string = json_string
        self.estimators.initialize(comm)
        # Path estimators: the initial purification has nothing to measure, so
        # there is no block 0.
