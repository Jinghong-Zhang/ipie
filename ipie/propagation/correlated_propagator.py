# Copyright 2022 The ipie Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Correlated propagator wrapper for paired-system propagation."""

import time

from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import synchronize


class CorrelatedPropagator:
    """Propagate walker channels A and B in lockstep.

    Force bias and mean-field shifts are evaluated separately by each inner
    propagator while sharing the same auxiliary fields in both channels.
    """

    def __init__(self, propagator_a, propagator_b):
        self.propagator_a = propagator_a
        self.propagator_b = propagator_b

    def cast_to_cupy(self, verbose=False):
        if hasattr(self.propagator_a, "cast_to_cupy"):
            self.propagator_a.cast_to_cupy(verbose=verbose)
        if hasattr(self.propagator_b, "cast_to_cupy"):
            self.propagator_b.cast_to_cupy(verbose=verbose)

    def _propagate_with_shared_xi(self, propagator, walkers, hamiltonian, trial, eshift, xi):
        synchronize()
        start_time = time.time()
        ovlp = trial.calc_greens_function(walkers)
        synchronize()
        propagator.timer.tgf += time.time() - start_time

        propagator.propagate_walkers_one_body(walkers)

        start_time = time.time()
        propagator.vbias = trial.calc_force_bias(hamiltonian, walkers, walkers.mpi_handler)
        xbar = -propagator.sqrt_dt * (1j * propagator.vbias - propagator.mf_shift)
        synchronize()
        propagator.timer.tfbias += time.time() - start_time

        xbar = propagator.apply_bound_force_bias(xbar, propagator.fbbound)
        xshifted = xi - xbar

        cmf = -propagator.sqrt_dt * xp.einsum("wx,x->w", xshifted, propagator.mf_shift)
        cfb = xp.einsum("wx,wx->w", xi, xbar) - 0.5 * xp.einsum("wx,wx->w", xbar, xbar)

        propagator.apply_VHS(walkers, hamiltonian, xshifted.T.copy())

        propagator.propagate_walkers_one_body(walkers)

        start_time = time.time()
        ovlp_new = trial.calc_overlap(walkers)
        synchronize()
        propagator.timer.tovlp += time.time() - start_time

        start_time = time.time()
        propagator.update_weight(walkers, ovlp, ovlp_new, cfb, cmf, eshift)
        synchronize()
        propagator.timer.tupdate += time.time() - start_time

    def propagate_walkers(
        self,
        correlated_walkers,
        hamiltonian_a,
        hamiltonian_b,
        trial_a,
        trial_b,
        eshift_a,
        eshift_b,
    ):
        """Propagate A and B, then update combined product observables."""
        shared_xi = xp.random.normal(
            0.0,
            1.0,
            hamiltonian_a.nfields * correlated_walkers.walkers_A.nwalkers,
        ).reshape(correlated_walkers.walkers_A.nwalkers, hamiltonian_a.nfields)

        self._propagate_with_shared_xi(
            self.propagator_a,
            correlated_walkers.walkers_A,
            hamiltonian_a,
            trial_a,
            eshift_a,
            shared_xi,
        )
        self._propagate_with_shared_xi(
            self.propagator_b,
            correlated_walkers.walkers_B,
            hamiltonian_b,
            trial_b,
            eshift_b,
            shared_xi,
        )

        correlated_walkers.sync_combined_state()

    @property
    def timer_a(self):
        return getattr(self.propagator_a, "timer", None)

    @property
    def timer_b(self):
        return getattr(self.propagator_b, "timer", None)
