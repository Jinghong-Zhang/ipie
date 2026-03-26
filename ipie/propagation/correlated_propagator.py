# Copyright 2022 The ipie Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Correlated propagator wrapper for paired-system propagation."""

import time

import numpy

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

    @staticmethod
    def _to_numpy(values):
        if values is None:
            return None
        if isinstance(values, numpy.ndarray):
            return values
        if hasattr(xp, "asnumpy"):
            return xp.asnumpy(values)
        return numpy.asarray(values)

    @classmethod
    def _first_three(cls, values):
        if values is None:
            return None
        return cls._to_numpy(values)[:3]

    @staticmethod
    def _format_values(values):
        if values is None:
            return "None"
        return numpy.array2string(values, precision=8, suppress_small=False)

    @classmethod
    def _walker_norms(cls, walkers):
        if hasattr(walkers, "phi") and walkers.phi is not None:
            phi = cls._to_numpy(walkers.phi)
            return numpy.linalg.norm(phi, axis=(1, 2))

        norm_sq = None
        if hasattr(walkers, "phia") and walkers.phia is not None:
            phia = cls._to_numpy(walkers.phia)
            norm_sq = numpy.sum(numpy.abs(phia) ** 2, axis=(1, 2))
        if hasattr(walkers, "phib") and walkers.phib is not None:
            phib = cls._to_numpy(walkers.phib)
            phib_norm_sq = numpy.sum(numpy.abs(phib) ** 2, axis=(1, 2))
            norm_sq = phib_norm_sq if norm_sq is None else norm_sq + phib_norm_sq
        if norm_sq is None:
            return None
        return numpy.sqrt(norm_sq)

    @classmethod
    def _print_channel_state(cls, label, walkers):
        print(
            f"# {label}: "
            f"weight[:3]={cls._format_values(cls._first_three(walkers.weight))} "
            f"ovlp[:3]={cls._format_values(cls._first_three(walkers.ovlp))} "
            f"norm[:3]={cls._format_values(cls._first_three(cls._walker_norms(walkers)))}"
        )

    @classmethod
    def _compute_weight_diagnostics(cls, propagator, walkers, ovlp, ovlp_new, cfb, cmf, eshift):
        if isinstance(ovlp, tuple):
            sgn_ovlp, log_ovlp = ovlp
            sgn_ovlpnew, log_ovlpnew = ovlp_new
            ovlp_ratio = sgn_ovlpnew / sgn_ovlp * xp.exp(log_ovlpnew - log_ovlp)
        else:
            ovlp_ratio = ovlp_new / ovlp

        hybrid_energy = -(xp.log(ovlp_ratio) + cfb + cmf) / propagator.dt
        bounded_hybrid_energy = propagator.apply_bound_hybrid(hybrid_energy.copy(), eshift)
        importance_function = xp.exp(
            -propagator.dt * (0.5 * (bounded_hybrid_energy + walkers.hybrid_energy) - eshift)
        )
        importance_magn = xp.abs(importance_function)
        dtheta = (-propagator.dt * bounded_hybrid_energy - cfb).imag
        cosine_fac = xp.cos(dtheta)
        xp.clip(cosine_fac, a_min=0.0, a_max=None, out=cosine_fac)
        return bounded_hybrid_energy, importance_magn, dtheta, cosine_fac

    @classmethod
    def _print_weight_diagnostics(
        cls, channel_name, walkers, hybrid_energy, importance_magn, cosine_fac
    ):
        print(
            f"# {channel_name} weight update: "
            f"hybrid_energy[:3]={cls._format_values(cls._first_three(hybrid_energy))} "
            f"importance_magn[:3]={cls._format_values(cls._first_three(importance_magn))} "
            f"cosine_fac[:3]={cls._format_values(cls._first_three(cosine_fac))}"
        )
        print(
            f"# {channel_name} updated weight[:3]="
            f"{cls._format_values(cls._first_three(walkers.weight))}"
        )

    def cast_to_cupy(self, verbose=False):
        if hasattr(self.propagator_a, "cast_to_cupy"):
            self.propagator_a.cast_to_cupy(verbose=verbose)
        if hasattr(self.propagator_b, "cast_to_cupy"):
            self.propagator_b.cast_to_cupy(verbose=verbose)

    def _propagate_with_shared_xi(
        self, channel_name, propagator, walkers, hamiltonian, trial, eshift, xi
    ):
        synchronize()
        start_time = time.time()
        ovlp = trial.calc_greens_function(walkers)
        synchronize()
        propagator.timer.tgf += time.time() - start_time
        self._print_channel_state(f"{channel_name} start", walkers)

        propagator.propagate_walkers_one_body(walkers)
        self._print_channel_state(f"{channel_name} after first one-body", walkers)

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
        self._print_channel_state(f"{channel_name} after two-body", walkers)

        propagator.propagate_walkers_one_body(walkers)
        self._print_channel_state(f"{channel_name} after second one-body", walkers)

        start_time = time.time()
        ovlp_new = trial.calc_overlap(walkers)
        synchronize()
        propagator.timer.tovlp += time.time() - start_time

        hybrid_energy, importance_magn, dtheta, cosine_fac = self._compute_weight_diagnostics(
            propagator, walkers, ovlp, ovlp_new, cfb, cmf, eshift
        )

        start_time = time.time()
        propagator.update_weight(walkers, ovlp, ovlp_new, cfb, cmf, eshift)
        synchronize()
        propagator.timer.tupdate += time.time() - start_time
        self._print_weight_diagnostics(
            channel_name, walkers, hybrid_energy, importance_magn, cosine_fac
        )
        return dtheta, cosine_fac

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

        dtheta_a, cosine_fac_a = self._propagate_with_shared_xi(
            "A",
            self.propagator_a,
            correlated_walkers.walkers_A,
            hamiltonian_a,
            trial_a,
            eshift_a,
            shared_xi,
        )
        dtheta_b, cosine_fac_b = self._propagate_with_shared_xi(
            "B",
            self.propagator_b,
            correlated_walkers.walkers_B,
            hamiltonian_b,
            trial_b,
            eshift_b,
            shared_xi,
        )

        correlated_walkers.sync_combined_state()
        print(
            "# Combined correlated state: "
            f"weight_A[:3]={self._format_values(self._first_three(correlated_walkers.weight_A))} "
            f"weight_B[:3]={self._format_values(self._first_three(correlated_walkers.weight_B))} "
            f"weight_A*weight_B[:3]={self._format_values(self._first_three(correlated_walkers.weight))}"
        )
        cosine_fac_sum = xp.cos(dtheta_a + dtheta_b)
        xp.clip(cosine_fac_sum, a_min=0.0, a_max=None, out=cosine_fac_sum)
        cosine_fac_product = cosine_fac_a * cosine_fac_b
        print(
            "# Separate cosine comparison: "
            f"cos(theta_A+theta_B)[:3]={self._format_values(self._first_three(cosine_fac_sum))} "
            f"cos(theta_A)*cos(theta_B)[:3]={self._format_values(self._first_three(cosine_fac_product))}"
        )

    @property
    def timer_a(self):
        return getattr(self.propagator_a, "timer", None)

    @property
    def timer_b(self):
        return getattr(self.propagator_b, "timer", None)
