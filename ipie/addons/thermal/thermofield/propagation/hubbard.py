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

import numpy
import scipy.linalg

from ipie.addons.thermal.thermofield.utils.linalg import inv_stack
from ipie.config import config
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


class ThermofieldHubbardCP:
    r"""Constrained-path propagation of thermofield walkers for the Hubbard model.

    Continuous spin-channel fields with the symmetric Trotter split

    .. math::

        B(x) = K e^{\pm\gamma x} K, \qquad K = e^{-\Delta\tau h/2}, \qquad
        \gamma = \sqrt{\Delta\tau U},

    the force bias of the guide, :math:`\bar{x}_i = \gamma (n_{i\uparrow} - n_{i\downarrow})`
    clipped to [-1, 1], and the constraint that the real part of the guide
    overlap ratio stays positive. Slice l takes the walkers from
    :math:`\theta = l \Delta\tau` to :math:`(l + 1)\Delta\tau`; the cached overlaps and
    Green's functions are those of the guide at the current theta.

    With `trace_branching` only the branching magnitude changes, to
    :math:`|T'/T|^\epsilon |S'/S|^{1 - \epsilon}` with :math:`T = \det(I + \Delta)` and
    :math:`\epsilon` = `trace_exponent`; the force bias and the constraint still
    use the guide, so the endpoint correction is
    :math:`\mathrm{sign}(T/S) |T/S|^{1 - \epsilon}`. Negative traces stay alive; zero
    or non-finite ones kill the walker and are counted separately.

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
    rng :
        Ignored: fields are drawn from `xp.random`, seeded by `set_rng_seed`.
        Kept so that calls that pass a generator still work.
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
    trace_branching : bool
        Branch on the closed trace (see above). Default False.
    energy_shift : float
        Energy reference added as timestep * energy_shift to every log weight
        update. It cancels from normalized observables and keeps the weights
        away from the numerical guards. Default 0.0.
    force_bias_mode : str
        "guide", or "trace" for the force bias of the closed trace (requires
        trace branching). Default "guide".
    trace_exponent : float
        Exponent epsilon of the trace branching, in [0, 1]. Default 1.0.

    Attributes
    ----------
    discrete : bool
        Fields are discrete Hirsch fields. False here.
    constraint : bool
        Apply the constraint; False is free projection with signed weights.
    kill_count, step_count : int
        Walkers killed by the constraint and walker steps taken.
    log_weight_bound_count, invalid_weight_count : int
        Numerical guard events; a driver should refuse a block that has any.
    """

    discrete = False

    def __init__(
        self,
        h_kin,
        U,
        timestep,
        beta,
        rng=None,
        record_fields=False,
        guide_mode="completion",
        fixed_guide_time=None,
        batched=True,
        cache_qinv=True,
        trace_branching=False,
        energy_shift=0.0,
        force_bias_mode="guide",
        trace_exponent=1.0,
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
        self.trace_branching = bool(trace_branching)
        if not numpy.isfinite(trace_exponent) or not 0.0 <= trace_exponent <= 1.0:
            raise ValueError("trace_exponent must lie in [0, 1].")
        self.trace_exponent = float(trace_exponent)
        if force_bias_mode not in ("guide", "trace"):
            raise ValueError("force_bias_mode must be guide or trace.")
        if force_bias_mode == "trace" and not self.trace_branching:
            raise ValueError("Trace force bias requires trace-magnitude branching.")
        self.force_bias_mode = force_bias_mode
        self.trace_invalid_count = 0
        self.trace_log_weight_bound_count = 0
        self.trace_population_bound_count = 0
        self.trace_weight_invalid_count = 0
        self.dt = timestep
        self.beta = beta
        self.U = float(U)
        self.hs_type = "continuous_spin"
        self.field_proposal = "force_biased_gaussian"
        self.importance_guide = "same"
        self.gamma = numpy.sqrt(timestep * U)
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

    def construct_slice_propagators(self, fields):
        r"""Slice propagators :math:`B(x)` for recorded fields.

        Parameters
        ----------
        fields : :class:`numpy.ndarray`
            Fields of one slice, shape (nwalkers, nfields).

        Returns
        -------
        B : :class:`numpy.ndarray`
            Shape (nwalkers, 2, nbasis, nbasis).
        """
        dv = numpy.exp(self.gamma * fields)
        B = numpy.empty((len(fields), 2, self.nfields, self.nfields), dtype=numpy.complex128)
        B[:, 0] = self.BK2[0] @ (dv[:, :, None] * self.BK2[0])
        B[:, 1] = self.BK2[1] @ ((1.0 / dv)[:, :, None] * self.BK2[1])
        return B

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
        """Set the guide overlaps and Green's functions at theta = 0 and the closed traces.

        Walkers whose cache is not finite get zero weight.
        """
        if self.trace_branching:
            walkers.log_tr[:] = to_host(
                trial.calc_log_trace_batched(walkers.Qmat, walkers.log_d, walkers.Tmat)
            )
            good_trace = numpy.isfinite(walkers.log_tr)
            self.trace_invalid_count += int((~good_trace).sum())
            walkers.weight[~good_trace] = 0.0
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

    def trace_force_bias(self, walkers, trial, indices):
        r"""Force bias of the closed trace :math:`\det(I + \Delta)` for walkers `indices`.

        Changes only the Gaussian proposal: the guide still sets the
        constraint, and the weight includes the actual shift.

        Returns
        -------
        xbar : :class:`numpy.ndarray`
            Shape (len(indices), nfields).
        """
        indices = numpy.asarray(indices)
        cached = self.cache_qinv and walkers.Qinv is not None
        _, green = trial.calc_log_overlap_and_greens_function_scaled_batched(
            walkers.Qmat[indices],
            walkers.log_d[indices],
            walkers.Tmat[indices],
            0.0,
            Qinv=walkers.Qinv[indices] if cached else None,
            log_det_Q=walkers.log_det_Q[indices] if cached else None,
        )
        if not numpy.isfinite(green).all():
            raise FloatingPointError("Nonfinite live closed-trace force bias.")
        density = numpy.diagonal(green, axis1=-2, axis2=-1).real
        return numpy.clip(self.gamma * (density[:, 0] - density[:, 1]), -1.0, 1.0)

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

    def _propagate_walkers_scalar(self, walkers, trial, l):
        """Per-walker reference implementation of :meth:`propagate_walkers`."""
        if config.get_option("use_gpu"):
            raise NotImplementedError("Continuous Hubbard fields run on the CPU only.")
        scale_new = self.guide_scale(l)
        for iw in range(walkers.nwalkers):
            if walkers.weight[iw] == 0.0:
                continue
            # Force bias from the cached mixed 1RDM of the previous theta,
            # P[s] = (I - G[s])^T and xbar_i = gamma (P_up,ii - P_down,ii).
            nup = 1.0 - numpy.diagonal(walkers.Ga[iw]).real
            ndn = 1.0 - numpy.diagonal(walkers.Gb[iw]).real
            xbar = (
                self.trace_force_bias(walkers, trial, [iw])[0]
                if self.force_bias_mode == "trace"
                else numpy.clip(self.gamma * (nup - ndn), -1.0, 1.0)
            )
            x = xbar + xp.random.standard_normal(self.nfields)

            if self.record_fields:
                walkers.field_hist[iw, l] = x
            d_up = numpy.exp(self.gamma * x)
            Q_new = numpy.empty_like(walkers.Qmat[iw])
            Q_new[0] = self.BK2[0] @ (d_up[:, None] * (self.BK2[0] @ walkers.Qmat[iw, 0]))
            Q_new[1] = self.BK2[1] @ ((1.0 / d_up)[:, None] * (self.BK2[1] @ walkers.Qmat[iw, 1]))

            try:
                log_ovlp_new, G_new = trial.calc_log_overlap_and_greens_function_scaled(
                    Q_new, walkers.log_d[iw], walkers.Tmat[iw], scale=scale_new
                )
            except (ZeroDivisionError, numpy.linalg.LinAlgError):
                walkers.weight[iw] = 0.0
                continue

            if not numpy.isfinite(log_ovlp_new) or not numpy.isfinite(G_new).all():
                walkers.weight[iw] = 0.0
                continue

            # Real constrained-path ratio; the guide time changed too.
            log_ratio = (log_ovlp_new - walkers.log_ovlp[iw]).real
            ovlp_sign_new = numpy.cos((log_ovlp_new - walkers.log_ovlp[iw]).imag)
            self.step_count += 1
            if ovlp_sign_new <= 0.0:
                self.kill_count += 1
                if self.constraint:
                    walkers.weight[iw] = 0.0
                    continue
            log_tr_new = None
            if self.trace_branching:
                try:
                    log_tr_new = trial.calc_log_trace_batched(
                        Q_new[None], walkers.log_d[iw][None], walkers.Tmat[iw][None]
                    )[0]
                except (ZeroDivisionError, numpy.linalg.LinAlgError):
                    log_tr_new = numpy.nan + 0.0j
                if not numpy.isfinite(log_tr_new):
                    self.trace_invalid_count += 1
                    walkers.weight[iw] = 0.0
                    continue
                # Negative traces stay alive: the constraint is on the guide.
                log_ratio = (1.0 - self.trace_exponent) * log_ratio + self.trace_exponent * (
                    log_tr_new - walkers.log_tr[iw]
                ).real
            log_w = log_ratio - x @ xbar + 0.5 * xbar @ xbar
            if self.energy_shift != 0.0:
                log_w += self.dt * self.energy_shift
            if numpy.isfinite(log_w):
                self.max_log_weight = max(self.max_log_weight, float(log_w))
                self.min_log_weight = min(self.min_log_weight, float(log_w))
            sgn = numpy.sign(ovlp_sign_new) if not self.constraint else 1.0
            if self.trace_branching:
                updated, failures = update_trace_weight(walkers.weight[iw], log_w, sgn)
                walkers.weight[iw] = updated.item()
                self.log_weight_bound_count += failures
                self.trace_log_weight_bound_count += failures
            else:
                if abs(log_w) > 60.0:
                    self.log_weight_bound_count += 1
                walkers.weight[iw] *= sgn * numpy.exp(numpy.clip(log_w, -60.0, 60.0))
            walkers.Qmat[iw] = Q_new
            walkers.set_walker_cache(iw, log_ovlp_new, G_new)
            if self.trace_branching:
                walkers.log_tr[iw] = log_tr_new

    def _propagate_walkers_batched(self, walkers, trial, l):
        """Stacked :meth:`propagate_walkers`.

        Fields are drawn for the live walkers in index order, the same random
        numbers as the per-walker loop.
        """
        if config.get_option("use_gpu"):
            raise NotImplementedError("Continuous Hubbard fields run on the CPU only.")
        scale_new = self.guide_scale(l)
        idx = numpy.flatnonzero(walkers.weight != 0.0)
        if idx.size == 0:
            return
        nup = 1.0 - numpy.diagonal(walkers.Ga[idx], axis1=1, axis2=2).real
        ndn = 1.0 - numpy.diagonal(walkers.Gb[idx], axis1=1, axis2=2).real
        xbar = (
            self.trace_force_bias(walkers, trial, idx)
            if self.force_bias_mode == "trace"
            else numpy.clip(self.gamma * (nup - ndn), -1.0, 1.0)
        )
        x = xbar + xp.random.standard_normal((idx.size, self.nfields))
        if self.record_fields:
            walkers.field_hist[idx, l] = x
        d_up = numpy.exp(self.gamma * x)
        Q = walkers.Qmat[idx]
        Q_new = numpy.empty_like(Q)
        Q_new[:, 0] = self.BK2[0] @ (d_up[:, :, None] * (self.BK2[0] @ Q[:, 0]))
        Q_new[:, 1] = self.BK2[1] @ ((1.0 / d_up)[:, :, None] * (self.BK2[1] @ Q[:, 1]))
        Qi_new = ldQ_new = None
        use_cache = self.cache_qinv and walkers.Qinv is not None
        if use_cache:
            # Q'^{-1} = Q^{-1} B^{-1} with B^{-1} = K^{-1} diag(1/dv) K^{-1}, and
            # log det Q' = log det Q + 2 log det K +- gamma sum(x): products and
            # sums instead of stacked inversions and slogdets.
            Qi = walkers.Qinv[idx]
            Qi_new = numpy.empty_like(Qi)
            Qi_new[:, 0] = ((Qi[:, 0] @ self.BK2inv[0]) * (1.0 / d_up)[:, None, :]) @ self.BK2inv[0]
            Qi_new[:, 1] = ((Qi[:, 1] @ self.BK2inv[1]) * d_up[:, None, :]) @ self.BK2inv[1]
            sx = self.gamma * x.sum(axis=1)
            ldQ = walkers.log_det_Q[idx]
            ldQ_new = numpy.empty_like(ldQ)
            ldQ_new[:, 0] = ldQ[:, 0] + 2.0 * self.logdet_BK2[0] + sx
            ldQ_new[:, 1] = ldQ[:, 1] + 2.0 * self.logdet_BK2[1] - sx
        lo_new, G_new = trial.calc_log_overlap_and_greens_function_scaled_batched(
            Q_new, walkers.log_d[idx], walkers.Tmat[idx], scale_new, Qinv=Qi_new, log_det_Q=ldQ_new
        )
        ok = numpy.isfinite(lo_new) & numpy.isfinite(G_new).all(axis=(1, 2, 3))
        dlo = lo_new - walkers.log_ovlp[idx]
        sgn_new = numpy.cos(dlo.imag)
        self.step_count += int(ok.sum())
        self.kill_count += int((ok & (sgn_new <= 0.0)).sum())
        log_magnitude = dlo.real
        log_tr_new = None
        if self.trace_branching:
            log_tr_new = trial.calc_log_trace_batched(Q_new, walkers.log_d[idx], walkers.Tmat[idx])
            finite_trace = numpy.isfinite(log_tr_new)
            self.trace_invalid_count += int((ok & ~finite_trace).sum())
            ok &= finite_trace
            # The sign stays that of the guide; the trace is never constrained.
            log_magnitude = (1.0 - self.trace_exponent) * log_magnitude + self.trace_exponent * (
                log_tr_new - walkers.log_tr[idx]
            ).real
        log_w = (
            log_magnitude
            - numpy.einsum("ij,ij->i", x, xbar)
            + 0.5 * numpy.einsum("ij,ij->i", xbar, xbar)
        )
        if self.energy_shift != 0.0:
            log_w = log_w + self.dt * self.energy_shift
        if self.trace_branching:
            surv = ok & (sgn_new > 0.0) if self.constraint else ok
            signs = 1.0 if self.constraint else numpy.sign(sgn_new)
            updated, bound_hits = update_trace_weight(
                numpy.where(surv, walkers.weight[idx], 0.0), log_w, signs
            )
            self.trace_log_weight_bound_count += bound_hits
        else:
            if self.constraint:
                surv = ok & (sgn_new > 0.0)
                fac = numpy.exp(numpy.clip(log_w, -60.0, 60.0))
            else:
                surv = ok
                fac = numpy.sign(sgn_new) * numpy.exp(numpy.clip(log_w, -60.0, 60.0))
            bound_hits = int((surv & (numpy.abs(log_w) > 60.0)).sum())
            updated = numpy.where(surv, walkers.weight[idx] * fac, 0.0)
        self.log_weight_bound_count += bound_hits
        finite_surv = surv & numpy.isfinite(log_w)
        if finite_surv.any():
            self.max_log_weight = max(self.max_log_weight, float(log_w[finite_surv].max()))
            self.min_log_weight = min(self.min_log_weight, float(log_w[finite_surv].min()))
        walkers.weight[idx] = updated
        up = idx[surv]
        I = numpy.eye(walkers.nbasis)
        walkers.Qmat[up] = Q_new[surv]
        walkers.log_ovlp[up] = lo_new[surv]
        walkers.Ga[up] = I - G_new[surv, 0]
        walkers.Gb[up] = I - G_new[surv, 1]
        if self.trace_branching:
            walkers.log_tr[up] = log_tr_new[surv]
        if use_cache:
            walkers.Qinv[up] = Qi_new[surv]
            walkers.log_det_Q[up] = ldQ_new[surv]

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
