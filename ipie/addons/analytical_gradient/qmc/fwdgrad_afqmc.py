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
"""Serial driver for forward-mode analytical-gradient phaseless AFQMC.

Mirrors the AD-block structure of ipie.addons.adafqmc.qmc.adafqmc: walkers are
detached (tangents reset) at block boundaries, a fresh propagator (with fresh
non-detached trial-energy shift) is built per block, sub-block energies are
combined weighted by their total weights, and stochastic reconfiguration runs
between blocks.  Returns per-block weight gradients (wtsgrad) as well, for the
weight-gradient bias analysis.
"""

import numpy as np

from ipie.addons.analytical_gradient.estimators.estimator import block_average_with_tangent
from ipie.addons.analytical_gradient.hamiltonians.hamiltonian import shifted_copy
from ipie.addons.analytical_gradient.propagation.propagator import GradPropagator
from ipie.addons.analytical_gradient.trial_wavefunction.sdtrial import SDTrial
from ipie.addons.analytical_gradient.utils.fields import RandomFields
from ipie.addons.analytical_gradient.walkers.rhf_walkers import (
    initialize_walkers,
    reorthogonalize,
    stochastic_reconfiguration,
)


class GradParams:
    def __init__(
        self,
        num_walkers,
        num_steps_per_block,
        ad_block_size,
        num_ad_blocks,
        timestep,
        stabilize_freq,
        pop_control_freq,
        pop_control_freq_eq,
        seed,
        num_eqlb_steps=None,
    ):
        self.num_walkers = num_walkers
        self.num_steps_per_block = num_steps_per_block
        self.ad_block_size = ad_block_size
        self.num_ad_blocks = num_ad_blocks
        self.timestep = timestep
        self.stabilize_freq = stabilize_freq
        self.pop_control_freq = pop_control_freq
        self.pop_control_freq_eq = pop_control_freq_eq
        self.seed = seed
        if num_eqlb_steps is None:
            num_eqlb_steps = int(2.0 / timestep) * 30
        self.num_eqlb_steps = num_eqlb_steps
        if ad_block_size % num_steps_per_block != 0:
            raise ValueError("ad_block_size must be divisible by num_steps_per_block")


class FwdGradAFQMC:
    def __init__(self, params, fields=None, debug=False):
        self.params = params
        self.fields = fields if fields is not None else RandomFields(params.seed)
        self.debug = debug
        self.last_diagnostics = None

    @staticmethod
    def build(
        num_walkers=50,
        num_steps_per_block=50,
        ad_block_size=800,
        num_ad_blocks=100,
        timestep=0.005,
        stabilize_freq=5,
        pop_control_freq=5,
        pop_control_freq_eq=5,
        seed=114,
        num_eqlb_steps=None,
        fields=None,
        debug=False,
    ):
        params = GradParams(
            num_walkers,
            num_steps_per_block,
            ad_block_size,
            num_ad_blocks,
            timestep,
            stabilize_freq,
            pop_control_freq,
            pop_control_freq_eq,
            seed,
            num_eqlb_steps,
        )
        return FwdGradAFQMC(params, fields=fields, debug=debug)

    def equilibrate_walkers(self, ham, trial):
        """Value-only equilibration (zero-tangent Hamiltonian copy).

        Mirrors adafqmc equilibrate_walkers: extra per-step weight bound from
        step 1 on, reconfiguration every pop_control_freq_eq steps.
        """
        ham0 = shifted_copy(ham, 0.0)
        trial0 = SDTrial(trial.psi, ham.nelec0)
        trial0.half_rot(ham0)
        walkers = initialize_walkers(trial0, self.params.num_walkers)
        prop = GradPropagator(
            self.params.timestep, ham0, trial0, self.params.num_steps_per_block
        )
        for step in range(self.params.num_eqlb_steps):
            if step % self.params.stabilize_freq == self.params.stabilize_freq - 1:
                walkers = reorthogonalize(walkers)
            x = self.fields.normal(walkers.nwalkers, ham0.nchol)
            walkers = prop.propagate_walkers(walkers, ham0, trial0, x)
            if step >= 1:
                wbound = 0.1 * np.sum(walkers.weight)
                walkers.weight = np.where(walkers.weight < wbound, walkers.weight, wbound)
            if step % self.params.pop_control_freq_eq == self.params.pop_control_freq_eq - 1:
                walkers, _ = stochastic_reconfiguration(walkers, self.fields.uniform())
        return walkers

    def ad_block(self, ham, trial, walkers, eshift_override=None):
        """One AD block: detach tangents, propagate, return block (E, dE, W, dW).

        The returned walkers are detached value copies, exactly as adafqmc
        detaches between ad_block_gradient calls.  eshift_override (array over
        sub-blocks) freezes the detached energy-shift sequence, for
        common-random-number finite-difference checks; the sub-block etots of
        a run are stored on last_subblock_etots for that purpose.
        """
        walkers = walkers.detached_copy()
        prop = GradPropagator(
            self.params.timestep,
            ham,
            trial,
            self.params.num_steps_per_block,
            debug=self.debug,
        )
        nsub = self.params.ad_block_size // self.params.num_steps_per_block
        etots = np.zeros(nsub)
        detots = np.zeros(nsub)
        wts = np.zeros(nsub)
        dwts = np.zeros(nsub)
        for i in range(nsub):
            walkers, etot, detot, totw, dtotw = prop.propagate_block(
                i,
                walkers,
                ham,
                trial,
                self.params.stabilize_freq,
                self.params.pop_control_freq,
                self.fields,
                eshift_override=None if eshift_override is None else eshift_override[i],
            )
            etots[i] = etot
            detots[i] = detot
            wts[i] = totw
            dwts[i] = dtotw
        if self.debug:
            self.last_diagnostics = prop.diagnostics
        self.last_subblock_etots = etots.copy()
        E, dE = block_average_with_tangent(etots, detots, wts, dwts)
        return E, dE, np.sum(wts), np.sum(dwts), walkers.detached_copy()

    def run_along_path(
        self, ham, trial, num_measurements, walkers=None, obs_const=0.0, sr_replay=None
    ):
        """Path-continuous forward mode: tangents ride the whole trajectory.

        No AD blocks: dphi and dweight are never reset, the energy-shift
        feedback is differentiated through (denergy_estimate = detot at each
        update), and the gradient is sampled at every measurement (every
        num_steps_per_block steps) exactly like the energy.  Stochastic
        reconfiguration keeps its exact pathwise semantics (weight tangents
        identically zero after resampling, state tangents gathered).

        Verification identity: with the same field stream and the SR map
        frozen (sr_replay = the base run's recorded index arrays, exposed on
        self.sr_record), the central finite difference of every measured
        E_i and W_i across lambda = +/- eps equals the returned dE_i and dW_i,
        for arbitrarily many measurements along the path.

        Walkers are not equilibrated here; pass equilibrated walkers (e.g.
        from equilibrate_walkers) or the trial-initialized default is used.
        Returns (energies, gradients, weights, wtsgrads, walkers) with one
        entry per measurement; gradients include obs_const.
        """
        if walkers is None:
            walkers = initialize_walkers(trial, self.params.num_walkers)
        prop = GradPropagator(
            self.params.timestep,
            ham,
            trial,
            self.params.num_steps_per_block,
            debug=self.debug,
        )
        sr_queue = None if sr_replay is None else [np.asarray(x) for x in sr_replay]
        self.sr_record = []
        energies = np.zeros(num_measurements)
        gradients = np.zeros(num_measurements)
        weights = np.zeros(num_measurements)
        wtsgrads = np.zeros(num_measurements)
        for i in range(num_measurements):
            walkers, etot, detot, totw, dtotw = prop.propagate_block(
                i,
                walkers,
                ham,
                trial,
                self.params.stabilize_freq,
                self.params.pop_control_freq,
                self.fields,
                detach_eshift=False,
                sr_indices_queue=sr_queue,
                sr_record=self.sr_record,
            )
            energies[i] = etot
            gradients[i] = detot + obs_const
            weights[i] = totw
            wtsgrads[i] = dtotw
        if self.debug:
            self.last_diagnostics = prop.diagnostics
        return energies, gradients, weights, wtsgrads, walkers

    def run(self, ham, trial, obs_const=0.0, verbose=False):
        """Full calculation: equilibrate then num_ad_blocks gradient blocks.

        Returns (energies, gradients, weights, wtsgrads) per block; gradients
        already include obs_const.  Averaging across blocks (weighted by
        weights) is left to the caller, as in adafqmc analysis.
        """
        if self.params.num_eqlb_steps > 0:
            walkers = self.equilibrate_walkers(ham, trial)
        else:
            walkers = initialize_walkers(trial, self.params.num_walkers)
        nb = self.params.num_ad_blocks
        energies = np.zeros(nb)
        gradients = np.zeros(nb)
        weights = np.zeros(nb)
        wtsgrads = np.zeros(nb)
        for b in range(nb):
            if b > 0:
                walkers, _ = stochastic_reconfiguration(walkers, self.fields.uniform())
            E, dE, W, dW, walkers = self.ad_block(ham, trial, walkers)
            energies[b] = E
            gradients[b] = dE + obs_const
            weights[b] = W
            wtsgrads[b] = dW
            if verbose:
                print(f"# iblock = {b}, etot = {E}, obs = {gradients[b]}")
        return energies, gradients, weights, wtsgrads
