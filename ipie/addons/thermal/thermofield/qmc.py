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
#
# Author: Jinghong Zhang <jinghongzhang@fas.harvard.edu>
#
"""Minimal driver for prototype thermofield-guided FT-AFQMC.

Each block propagates two independent walker populations (replicas L and R)
from the identity purification Delta = I to theta = beta / 2 with the same
fixed target-beta guide, applying population control along the way, then
measures the one-sided mixed estimator (diagnostic) and the two-replica
estimator (the intended finite-temperature estimator) and resets.
"""

from types import SimpleNamespace

import numpy

from ipie.addons.thermal.qmc.options import ThermalQMCParams
from ipie.addons.thermal.thermofield.estimators import (
    local_energy_thermofield_mixed,
    particle_number_thermofield_mixed,
    replica_energy_estimate,
)
from ipie.addons.thermal.thermofield.propagation import ThermofieldPhaseless
from ipie.addons.thermal.thermofield.trial import ThermofieldThermalTrial
from ipie.addons.thermal.thermofield.walkers import ThermofieldWalkers
from ipie.config import MPI
from ipie.utils.mpi import MPIHandler
from ipie.walkers.pop_controller import PopController


def _allgather_thermofield_walkers(walkers, comm):
    """Collect the walker fields needed by replica estimators on every rank."""
    if comm.size == 1:
        return walkers

    fields = {}
    for name in ("Qmat", "log_d", "Tmat", "log_ovlp", "weight", "phase"):
        local = numpy.ascontiguousarray(getattr(walkers, name))
        gathered = numpy.empty((comm.size,) + local.shape, dtype=local.dtype)
        comm.Allgather(local, gathered)
        fields[name] = gathered.reshape(
            (comm.size * local.shape[0],) + local.shape[1:]
        )

    return SimpleNamespace(
        nbasis=walkers.nbasis,
        nwalkers=comm.size * walkers.nwalkers,
        **fields,
    )


class ThermofieldAFQMC:
    """Prototype thermofield FT-AFQMC driver with two replica populations."""

    def __init__(
        self,
        hamiltonian,
        trial: ThermofieldThermalTrial,
        walkers_left: ThermofieldWalkers,
        walkers_right: ThermofieldWalkers,
        propagator: ThermofieldPhaseless,
        mpi_handler,
        params: ThermalQMCParams,
        pairing: str = "random_permutation",
        measure_mode: str = "complex",
        verbose: int = 0,
    ):
        self.hamiltonian = hamiltonian
        self.trial = trial
        self.walkers_left = walkers_left
        self.walkers_right = walkers_right
        self.propagator = propagator
        self.mpi_handler = mpi_handler
        self.params = params
        self.pairing = pairing
        self.measure_mode = measure_mode
        self.verbose = verbose

    @staticmethod
    def build(
        mu: float,
        beta: float,
        hamiltonian,
        trial: ThermofieldThermalTrial = None,
        nwalkers: int = 32,
        nblocks: int = 10,
        timestep: float = 0.01,
        seed=None,
        stabilize_freq: int = 5,
        pop_control_freq: int = 5,
        pop_control_method: str = "pair_branch",
        phaseless: bool = True,
        pairing: str = "random_permutation",
        measure_mode: str = "complex",
        mpi_handler=None,
        verbose: int = 0,
    ) -> "ThermofieldAFQMC":
        """Factory mirroring `ThermalAFQMC.build`.

        Note beta / (2 * timestep) must be (close to) an integer: walkers are
        propagated over theta in [0, beta / 2].
        """
        if mpi_handler is None:
            mpi_handler = MPIHandler()
        comm = mpi_handler.comm

        if seed is not None:
            numpy.random.seed(seed + 7 * comm.rank)

        # pylint: disable = no-value-for-parameter
        params = ThermalQMCParams(
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
        )

        if trial is None:
            trial = ThermofieldThermalTrial(hamiltonian, beta, mu=mu, verbose=verbose)

        walkers_left = ThermofieldWalkers(
            trial, hamiltonian.nbasis, nwalkers, mpi_handler=mpi_handler, verbose=verbose
        )
        walkers_right = ThermofieldWalkers(
            trial, hamiltonian.nbasis, nwalkers, mpi_handler=mpi_handler, verbose=verbose
        )
        propagator = ThermofieldPhaseless(timestep, mu, phaseless=phaseless, verbose=verbose)
        propagator.build(
            hamiltonian,
            trial=trial,
            walkers=walkers_right,
            mpi_handler=mpi_handler,
            verbose=verbose,
        )
        return ThermofieldAFQMC(
            hamiltonian,
            trial,
            walkers_left,
            walkers_right,
            propagator,
            mpi_handler,
            params,
            pairing=pairing,
            measure_mode=measure_mode,
            verbose=verbose,
        )

    def propagate_population(self, walkers, nslices_half, pcontrol):
        """Propagate one population from Delta = I to theta = beta / 2."""
        comm = self.mpi_handler.comm
        walkers.reset(self.trial)
        for t in range(nslices_half):
            self.propagator.propagate_walkers(walkers, self.hamiltonian, self.trial)
            if self.propagator.phaseless and t > 0:
                # Cap outlier weights, mirroring the ThermalAFQMC driver.
                wbound = pcontrol.total_weight * 0.10
                numpy.clip(walkers.weight, a_min=-wbound, a_max=wbound, out=walkers.weight)
            if t > 0 and (t + 1) % self.params.num_stblz == 0:
                walkers.stabilize()
            if self.propagator.phaseless and t > 0 and t % self.params.pop_control_freq == 0:
                pcontrol.pop_control(walkers, comm)

    def run(self, verbose=None):
        """Run the block loop; returns per-block estimates as numpy arrays.

        Each block is an independent pair of imaginary-time sweeps.  `e_mix`
        is the one-sided mixed diagnostic (right population); `e_rep` / and
        `nav_rep` come from the two-replica estimator.
        """
        if verbose is None:
            verbose = self.verbose
        comm = self.mpi_handler.comm
        free_projection = not self.propagator.phaseless

        nslices_half = int(numpy.rint(0.5 * self.params.beta / self.params.timestep))
        assert (
            abs(nslices_half * self.params.timestep - 0.5 * self.params.beta) < 1e-12
        ), "beta / (2 timestep) must be an integer."

        pcontrol_left = PopController(
            self.params.num_walkers,
            nslices_half,
            self.mpi_handler,
            self.params.pop_control_method,
            verbose=False,
        )
        pcontrol_right = PopController(
            self.params.num_walkers,
            nslices_half,
            self.mpi_handler,
            self.params.pop_control_method,
            verbose=False,
        )

        rng = numpy.random.default_rng(
            self.params.rng_seed + 13 * comm.rank if self.params.rng_seed is not None else None
        )

        results = {k: [] for k in ("e_mix", "nav_mix", "e_rep", "e1b_rep", "e2b_rep", "nav_rep")}
        if verbose and comm.rank == 0:
            print("# Block     E_mix         Nav_mix       E_rep         Nav_rep")

        for block in range(self.params.num_blocks):
            self.propagate_population(self.walkers_left, nslices_half, pcontrol_left)
            self.propagate_population(self.walkers_right, nslices_half, pcontrol_right)

            # One-sided mixed diagnostic on the right population.
            energies = local_energy_thermofield_mixed(self.hamiltonian, self.walkers_right)
            navs = particle_number_thermofield_mixed(self.walkers_right)
            weights = self.walkers_right.weight.astype(numpy.complex128)
            if free_projection:
                weights = weights * self.walkers_right.phase
            mix_numer = numpy.array(
                [numpy.sum(weights * energies[:, 0]), numpy.sum(weights * navs)]
            )
            mix_denom = numpy.array([numpy.sum(weights)])
            # For all-pairs measurement, pair each local left walker with the
            # global right population. The final reduction then contains every
            # global pair exactly once.
            replica_walkers_right = self.walkers_right
            if self.pairing == "all_pairs":
                replica_walkers_right = _allgather_thermofield_walkers(
                    self.walkers_right, comm
                )
            rep = replica_energy_estimate(
                self.hamiltonian,
                self.walkers_left,
                replica_walkers_right,
                pairing=self.pairing,
                mode=self.measure_mode,
                rng=rng,
                free_projection=free_projection,
            )
            # Per-rank replica sums carry a rank-dependent factor
            # exp(-log_shift); rescale to the global maximum shift before
            # summing across ranks.
            shift_max = comm.allreduce(rep["log_shift"], op=MPI.MAX)
            fac = numpy.exp(rep["log_shift"] - shift_max)
            local = numpy.array(
                [
                    mix_numer[0],
                    mix_numer[1],
                    mix_denom[0],
                    fac * rep["numer"],
                    fac * rep["denom"],
                    fac * rep["numer_e1b"],
                    fac * rep["numer_e2b"],
                    fac * rep["numer_nav"],
                ],
                dtype=numpy.complex128,
            )
            glob = numpy.zeros_like(local)
            comm.Allreduce(local, glob)

            results["e_mix"].append((glob[0] / glob[2]).real)
            results["nav_mix"].append((glob[1] / glob[2]).real)
            results["e_rep"].append((glob[3] / glob[4]).real)
            results["e1b_rep"].append((glob[5] / glob[4]).real)
            results["e2b_rep"].append((glob[6] / glob[4]).real)
            results["nav_rep"].append((glob[7] / glob[4]).real)

            if verbose and comm.rank == 0:
                print(
                    f"  {block:5d}  {results['e_mix'][-1]: .8f}  {results['nav_mix'][-1]: .8f}"
                    f"  {results['e_rep'][-1]: .8f}  {results['nav_rep'][-1]: .8f}"
                )

        return {k: numpy.array(v) for k, v in results.items()}
