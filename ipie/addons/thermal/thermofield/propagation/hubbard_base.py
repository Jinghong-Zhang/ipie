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

from abc import ABC, abstractmethod

import numpy
import scipy.linalg

from ipie.addons.thermal.thermofield.utils.linalg import inv_stack
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import to_host


def update_trace_weight(old_weight, log_increment, sign=1.0):
    """Multiply weights by sign * exp(log_increment) without clipping.

    Ordinary increments use the plain product. A large finite increment may
    compensate a very small previous weight, so then only the combined
    logarithm decides whether the result is representable. A live weight that
    overflows, is not finite or underflows to zero is set to zero and counted
    as a failure; dead input weights stay dead without a count.

    Parameters
    ----------
    old_weight : :class:`numpy.ndarray`
        Weights.
    log_increment : :class:`numpy.ndarray`
        Log magnitude of the factors.
    sign : float or :class:`numpy.ndarray`
        Signs of the factors. Default 1.0.

    Returns
    -------
    updated : :class:`numpy.ndarray`
        New weights.
    failures : int
        Number of live weights that failed.
    """
    old, increment, signs = numpy.broadcast_arrays(
        numpy.asarray(old_weight, dtype=float),
        numpy.asarray(log_increment, dtype=float),
        numpy.asarray(sign, dtype=float),
    )
    updated = numpy.zeros_like(old)
    active = old != 0.0
    ordinary = active & (numpy.abs(increment) <= 60.0)
    extreme = active & ~ordinary
    with numpy.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        updated[ordinary] = old[ordinary] * (signs[ordinary] * numpy.exp(increment[ordinary]))
        combined_log = numpy.log(numpy.abs(old[extreme])) + increment[extreme]
        updated[extreme] = numpy.sign(old[extreme]) * signs[extreme] * numpy.exp(combined_log)
    failed = active & ((updated == 0.0) | ~numpy.isfinite(updated))
    updated[failed] = 0.0
    return updated, int(failed.sum())


class ThermofieldHubbardBase(ABC):
    r"""Base class for constrained-path propagators of Hubbard thermofield walkers.

    A slice is the symmetric Trotter split :math:`B = K D(x) K` with
    :math:`K = e^{-\Delta\tau h/2}` and the field factors :math:`D(x)` of a
    subclass. Slice l takes the walkers from :math:`\theta = l \Delta\tau` to
    :math:`(l + 1)\Delta\tau`; the cached overlaps and Green's functions are those
    of the guide at the current theta, and a walker whose guide overlap ratio
    has a negative real part is killed (the constraint).

    Subclasses set `discrete`, `gamma`, `hs_type`, `field_proposal` and
    `importance_guide`, and sample the fields.

    Parameters
    ----------
    h_kin : :class:`numpy.ndarray`
        Real symmetric one-body Hamiltonian per spin with the chemical potential
        and the U/2 shift included, shape (2, nsites, nsites).
    U : float
        Hubbard interaction.
    timestep : float
        Imaginary timestep.
    beta : float
        Inverse temperature of the walk.
    record_fields : bool
        Store the fields of every slice in `walkers.field_hist`. Default False.
    guide_mode : str
        "completion": the guide time after slice l is beta - (l + 1) dt;
        "fixed": `fixed_guide_time` at every slice. Default "completion".
    fixed_guide_time : float
        Guide time of the fixed guide. Required for guide_mode "fixed".
    batched : bool
        Propagate all walkers with stacked kernels. Default True.
    cache_qinv : bool
        Keep the Q inverses and log determinants of the walkers in
        `walkers.Qinv` and `walkers.log_det_Q`, updated by products instead of
        inversions. Default True.
    energy_shift : float
        Energy reference added as timestep * energy_shift to every log weight
        update. It cancels from normalized observables and keeps the weights
        away from the numerical guards. Default 0.0.

    Attributes
    ----------
    constraint : bool
        Apply the constraint; False is free projection with signed weights.
    kill_count, step_count : int
        Walkers killed by the constraint and walker steps taken.
    log_weight_bound_count, invalid_weight_count : int
        Numerical guard events; a driver should refuse a block that has any.
    """

    def __init__(
        self,
        h_kin,
        U,
        timestep,
        beta,
        record_fields=False,
        guide_mode="completion",
        fixed_guide_time=None,
        batched=True,
        cache_qinv=True,
        energy_shift=0.0,
    ):
        h_kin = numpy.asarray(h_kin)
        if (
            h_kin.ndim != 3
            or h_kin.shape[0] != 2
            or h_kin.shape[1] != h_kin.shape[2]
            or h_kin.shape[1] == 0
            or not numpy.all(numpy.isfinite(h_kin))
            or numpy.any(numpy.iscomplex(h_kin))
            or not numpy.allclose(h_kin, h_kin.swapaxes(-1, -2), atol=1e-12)
        ):
            raise ValueError("h_kin must contain two finite real symmetric matrices.")
        h_kin = h_kin.real
        if not numpy.isfinite(U) or U < 0:
            raise ValueError("U must be finite and non-negative.")
        if not numpy.isfinite(timestep) or timestep <= 0:
            raise ValueError("timestep must be finite and positive.")
        if not numpy.isfinite(beta) or beta <= 0:
            raise ValueError("beta must be finite and positive.")
        if not numpy.isfinite(energy_shift):
            raise ValueError("energy_shift must be finite.")
        self.energy_shift = float(energy_shift)
        self.log_weight_bound_count = 0
        self.population_weight_bound_count = 0
        self.invalid_weight_count = 0
        self.max_log_weight = -numpy.inf
        self.min_log_weight = numpy.inf
        self.batched = bool(batched)
        self.cache_qinv = bool(cache_qinv)
        self.dt = timestep
        self.beta = beta
        self.U = float(U)
        self.BK2 = numpy.array([scipy.linalg.expm(-0.5 * timestep * h_kin[s]) for s in range(2)])
        self.BK2inv = numpy.array([scipy.linalg.expm(+0.5 * timestep * h_kin[s]) for s in range(2)])
        self.nfields = h_kin.shape[-1]
        self.record_fields = record_fields
        if guide_mode not in ("completion", "fixed"):
            raise ValueError("guide_mode must be completion or fixed.")
        self.guide_mode = guide_mode
        if fixed_guide_time is None:
            if guide_mode == "fixed":
                raise ValueError("fixed_guide_time is required when guide_mode='fixed'.")
            fixed_guide_time = beta  # Unused in completion mode.
        if not numpy.isfinite(fixed_guide_time) or fixed_guide_time < 0.0:
            raise ValueError("fixed_guide_time must be non-negative.")
        self.fixed_guide_time = float(fixed_guide_time)
        self.constraint = True
        self.kill_count = 0
        self.step_count = 0
        # Log determinants of the kinetic halves, real with positive determinant.
        self.logdet_BK2 = numpy.array([numpy.linalg.slogdet(self.BK2[s])[1] for s in range(2)])

    @abstractmethod
    def construct_slice_propagators(self, fields):
        """Slice propagators of recorded fields, shape (nwalkers, 2, nbasis, nbasis)."""

    def seed_qinv_cache(self, walkers):
        """Compute the cached Q inverses and log determinants of all walkers.

        Needed at the start of a walk and after population control, which
        replaces Q factors.
        """
        Q = walkers.Qmat
        walkers.Qinv = inv_stack(Q)
        with numpy.errstate(divide="ignore", invalid="ignore"):
            sgn, log = xp.linalg.slogdet(Q)
            walkers.log_det_Q = log + xp.log(sgn)

    def refresh_qinv_after_stabilize(self, walkers):
        """Update the Q caches after `walkers.stabilize()`.

        The Q factors are then unitary, so the inverse is the conjugate
        transpose; only the phase of the log determinant needs a slogdet.
        """
        Q = walkers.Qmat
        walkers.Qinv = xp.conj(xp.swapaxes(Q, -1, -2))
        with numpy.errstate(divide="ignore", invalid="ignore"):
            sgn, log = xp.linalg.slogdet(Q)
            walkers.log_det_Q = log + xp.log(sgn)

    def guide_scale(self, l):
        """Guide time after slice `l`."""
        if self.guide_mode == "fixed":
            return self.fixed_guide_time
        # Protect the exact endpoint against a negative time from rounding.
        return max(0.0, self.beta - (l + 1) * self.dt)

    def init_walker_cache(self, walkers, trial):
        """Set the guide overlaps and Green's functions at theta = 0.

        Walkers whose cache is not finite get zero weight.
        """
        scale0 = self.fixed_guide_time if self.guide_mode == "fixed" else self.beta
        if self.batched:
            lo, G = trial.calc_log_overlap_and_greens_function_scaled_batched(
                walkers.Qmat, walkers.log_d, walkers.Tmat, scale0
            )
            ok = xp.isfinite(lo) & xp.isfinite(G).all(axis=(1, 2, 3))
            # Weights and overlaps live on the host, Green's functions on the backend.
            ok_h = to_host(ok)
            walkers.weight[~ok_h] = 0.0
            I = xp.eye(walkers.nbasis)
            walkers.log_ovlp[ok_h] = to_host(lo)[ok_h]
            walkers.Ga[ok] = I - G[ok, 0]
            walkers.Gb[ok] = I - G[ok, 1]
            return
        for iw in range(walkers.nwalkers):
            try:
                log_ovlp, G = trial.calc_log_overlap_and_greens_function_scaled(
                    walkers.Qmat[iw], walkers.log_d[iw], walkers.Tmat[iw], scale=scale0
                )
            except (ZeroDivisionError, numpy.linalg.LinAlgError):
                walkers.weight[iw] = 0.0
                continue
            if not numpy.isfinite(log_ovlp) or not numpy.isfinite(G).all():
                walkers.weight[iw] = 0.0
                continue
            walkers.set_walker_cache(iw, log_ovlp, G)

    def propagate_walkers(self, walkers, trial, l):
        """Propagate all walkers over slice `l` with constrained-path importance sampling.

        Parameters
        ----------
        walkers : :class:`ThermofieldWalkers`
            Walkers, updated in place.
        trial : :class:`ThermofieldCPTrial`
            Guide.
        l : int
            Slice index.
        """
        if self.batched:
            self._propagate_walkers_batched(walkers, trial, l)
        else:
            self._propagate_walkers_scalar(walkers, trial, l)

    @abstractmethod
    def _propagate_walkers_scalar(self, walkers, trial, l):
        """Per-walker reference implementation of :meth:`propagate_walkers`."""

    @abstractmethod
    def _propagate_walkers_batched(self, walkers, trial, l):
        """Stacked :meth:`propagate_walkers`."""

    # Compatibility with earlier versions.
    def propagate_slice(self, walkers, trial, l):
        self._propagate_walkers_scalar(walkers, trial, l)

    def propagate_slice_batched(self, walkers, trial, l):
        self._propagate_walkers_batched(walkers, trial, l)

    def refresh_cache_after_stabilize(self, walkers, trial, l):
        """Recompute the guide overlaps and Green's functions after `walkers.stabilize()`."""
        scale = self.guide_scale(l)
        for iw in range(walkers.nwalkers):
            if walkers.weight[iw] == 0.0:
                continue
            try:
                log_ovlp, G = trial.calc_log_overlap_and_greens_function_scaled(
                    walkers.Qmat[iw], walkers.log_d[iw], walkers.Tmat[iw], scale=scale
                )
            except (ZeroDivisionError, numpy.linalg.LinAlgError):
                walkers.weight[iw] = 0.0
                continue
            if not numpy.isfinite(log_ovlp) or not numpy.isfinite(G).all():
                walkers.weight[iw] = 0.0
                continue
            walkers.set_walker_cache(iw, log_ovlp, G)
