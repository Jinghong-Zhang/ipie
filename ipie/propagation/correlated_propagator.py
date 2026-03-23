# Copyright 2022 The ipie Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Joint correlated propagator for paired-system propagation."""

from __future__ import annotations

import time
from dataclasses import dataclass

from ipie.propagation.continuous_base import PropagatorTimer
from ipie.propagation.operations import propagate_one_body
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import cast_to_device, synchronize


@dataclass
class _ChannelState:
    expH1: xp.ndarray
    mf_shift: xp.ndarray
    kernel: object
    vbias: xp.ndarray | None = None


class CorrelatedPropagator:
    """Propagate a paired walker with one shared auxiliary-field draw."""

    def __init__(self, propagator_a, propagator_b):
        self.dt = propagator_a.dt
        self.sqrt_dt = propagator_a.sqrt_dt
        self.isqrt_dt = propagator_a.isqrt_dt
        self.fbbound = propagator_a.fbbound
        self.ebound = propagator_a.ebound
        self.verbose = getattr(propagator_a, "verbose", False)
        self.timer = PropagatorTimer()

        self.channel_a = _ChannelState(
            expH1=propagator_a.expH1,
            mf_shift=propagator_a.mf_shift,
            kernel=self._make_vhs_kernel(propagator_a),
        )
        self.channel_b = _ChannelState(
            expH1=propagator_b.expH1,
            mf_shift=propagator_b.mf_shift,
            kernel=self._make_vhs_kernel(propagator_b),
        )

    def _make_vhs_kernel(self, propagator):
        kernel = type(propagator)(
            propagator.dt,
            ebound_const=propagator.ebound * propagator.dt,
            fbbound=propagator.fbbound,
            verbose=getattr(propagator, "verbose", False),
        )
        kernel.timer = self.timer
        if hasattr(propagator, "exp_nmax"):
            kernel.exp_nmax = propagator.exp_nmax
        if hasattr(propagator, "mpi_handler"):
            kernel.mpi_handler = propagator.mpi_handler
        return kernel

    def cast_to_cupy(self, verbose=False):
        cast_to_device(self, verbose=verbose)
        if hasattr(self.channel_a.kernel, "cast_to_cupy"):
            self.channel_a.kernel.cast_to_cupy(verbose=verbose)
        if hasattr(self.channel_b.kernel, "cast_to_cupy"):
            self.channel_b.kernel.cast_to_cupy(verbose=verbose)

    def _propagate_walkers_one_body(self, walkers, expH1):
        start_time = time.time()
        walkers.phia = propagate_one_body(walkers.phia, expH1[0])
        if walkers.ndown > 0 and not walkers.rhf:
            walkers.phib = propagate_one_body(walkers.phib, expH1[1])
        synchronize()
        self.timer.tgemm += time.time() - start_time

    def _apply_bound_force_bias(self, xbar):
        absxbar = xp.abs(xbar)
        idx_to_rescale = absxbar > self.fbbound
        nonzeros = absxbar > 1e-13
        xbar_rescaled = xbar.copy()
        xbar_rescaled[nonzeros] = xbar_rescaled[nonzeros] / absxbar[nonzeros]
        return xp.where(idx_to_rescale, xbar_rescaled, xbar)

    def _apply_bound_hybrid(self, ehyb, eshift):
        if abs(eshift) < 1e-10:
            return ehyb
        emax = eshift.real + self.ebound
        emin = eshift.real - self.ebound
        xp.clip(ehyb.real, a_min=emin, a_max=emax, out=ehyb.real)
        synchronize()
        return ehyb

    def _calc_overlap_ratio(self, ovlp, ovlp_new):
        if isinstance(ovlp, tuple):
            sgn_ovlp, log_ovlp = ovlp
            sgn_ovlpnew, log_ovlpnew = ovlp_new
            return sgn_ovlpnew / sgn_ovlp * xp.exp(log_ovlpnew - log_ovlp)
        return ovlp_new / ovlp

    def _combine_overlap(self, ovlp_a, ovlp_b):
        if isinstance(ovlp_a, tuple):
            sgn_a, log_a = ovlp_a
            sgn_b, log_b = ovlp_b
            return (sgn_a * sgn_b, log_a + log_b)
        return ovlp_a * ovlp_b

    def _calc_overlap_and_gf(self, trial, walkers):
        synchronize()
        start_time = time.time()
        ovlp = trial.calc_greens_function(walkers)
        synchronize()
        self.timer.tgf += time.time() - start_time
        return ovlp

    def _calc_overlap(self, trial, walkers):
        start_time = time.time()
        ovlp = trial.calc_overlap(walkers)
        synchronize()
        self.timer.tovlp += time.time() - start_time
        return ovlp

    def _calc_force_bias(self, channel, walkers, hamiltonian, trial):
        start_time = time.time()
        channel.vbias = trial.calc_force_bias(hamiltonian, walkers, walkers.mpi_handler)
        xbar = -self.sqrt_dt * (1j * channel.vbias - channel.mf_shift)
        synchronize()
        self.timer.tfbias += time.time() - start_time
        return self._apply_bound_force_bias(xbar)

    def _update_weight(
        self,
        correlated_walkers,
        ovlp_a,
        ovlp_b,
        ovlp_new_a,
        ovlp_new_b,
        cfb_a,
        cfb_b,
        cmf_a,
        cmf_b,
        eshift,
    ):
        ovlp_ratio = self._calc_overlap_ratio(ovlp_a, ovlp_new_a) * self._calc_overlap_ratio(
            ovlp_b, ovlp_new_b
        )
        cfb = cfb_a + cfb_b
        cmf = cmf_a + cmf_b

        hybrid_energy = -(xp.log(ovlp_ratio) + cfb + cmf) / self.dt
        hybrid_energy = self._apply_bound_hybrid(hybrid_energy, eshift)
        importance_function = xp.exp(
            -self.dt * (0.5 * (hybrid_energy + correlated_walkers.hybrid_energy) - eshift)
        )
        magn = xp.abs(importance_function)
        dtheta = (-self.dt * hybrid_energy - cfb).imag
        cosine_fac = xp.cos(dtheta)
        xp.clip(cosine_fac, a_min=0.0, a_max=None, out=cosine_fac)

        correlated_walkers.weight = correlated_walkers.weight * magn * cosine_fac
        correlated_walkers.ovlp = self._combine_overlap(ovlp_new_a, ovlp_new_b)
        correlated_walkers.hybrid_energy = hybrid_energy

        correlated_walkers.walkers_A.ovlp = ovlp_new_a
        correlated_walkers.walkers_B.ovlp = ovlp_new_b
        if isinstance(ovlp_new_a, tuple):
            correlated_walkers.walkers_A.sgn_ovlp = ovlp_new_a[0]
            correlated_walkers.walkers_A.log_ovlp = ovlp_new_a[1]
            correlated_walkers.walkers_B.sgn_ovlp = ovlp_new_b[0]
            correlated_walkers.walkers_B.log_ovlp = ovlp_new_b[1]
        correlated_walkers.sync_combined_state()

    def propagate_walkers(
        self,
        correlated_walkers,
        hamiltonian_a,
        hamiltonian_b,
        trial_a,
        trial_b,
        eshift,
    ):
        """Propagate A and B together using a single shared Gaussian field."""
        walkers_a = correlated_walkers.walkers_A
        walkers_b = correlated_walkers.walkers_B

        if hamiltonian_a.nfields != hamiltonian_b.nfields:
            raise ValueError("Correlated propagation requires A and B to have the same nfields.")

        ovlp_a = self._calc_overlap_and_gf(trial_a, walkers_a)
        ovlp_b = self._calc_overlap_and_gf(trial_b, walkers_b)

        self._propagate_walkers_one_body(walkers_a, self.channel_a.expH1)
        self._propagate_walkers_one_body(walkers_b, self.channel_b.expH1)

        xbar_a = self._calc_force_bias(self.channel_a, walkers_a, hamiltonian_a, trial_a)
        xbar_b = self._calc_force_bias(self.channel_b, walkers_b, hamiltonian_b, trial_b)

        xi = xp.random.normal(
            0.0,
            1.0,
            hamiltonian_a.nfields * walkers_a.nwalkers,
        ).reshape(walkers_a.nwalkers, hamiltonian_a.nfields)

        xshifted_a = xi - xbar_a
        xshifted_b = xi - xbar_b

        cmf_a = -self.sqrt_dt * xp.einsum("wx,x->w", xshifted_a, self.channel_a.mf_shift)
        cmf_b = -self.sqrt_dt * xp.einsum("wx,x->w", xshifted_b, self.channel_b.mf_shift)
        cfb_a = xp.einsum("wx,wx->w", xi, xbar_a) - 0.5 * xp.einsum("wx,wx->w", xbar_a, xbar_a)
        cfb_b = xp.einsum("wx,wx->w", xi, xbar_b) - 0.5 * xp.einsum("wx,wx->w", xbar_b, xbar_b)

        self.channel_a.kernel.apply_VHS(walkers_a, hamiltonian_a, xshifted_a.T.copy())
        self.channel_b.kernel.apply_VHS(walkers_b, hamiltonian_b, xshifted_b.T.copy())

        self._propagate_walkers_one_body(walkers_a, self.channel_a.expH1)
        self._propagate_walkers_one_body(walkers_b, self.channel_b.expH1)

        ovlp_new_a = self._calc_overlap(trial_a, walkers_a)
        ovlp_new_b = self._calc_overlap(trial_b, walkers_b)

        start_time = time.time()
        self._update_weight(
            correlated_walkers,
            ovlp_a,
            ovlp_b,
            ovlp_new_a,
            ovlp_new_b,
            cfb_a,
            cfb_b,
            cmf_a,
            cmf_b,
            eshift,
        )
        synchronize()
        self.timer.tupdate += time.time() - start_time

    @property
    def timer_a(self):
        return self.timer

    @property
    def timer_b(self):
        return None
