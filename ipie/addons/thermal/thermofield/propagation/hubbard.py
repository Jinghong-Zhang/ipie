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

from ipie.addons.thermal.thermofield.propagation.hubbard_base import (
    ThermofieldHubbardBase,
    update_trace_weight,
)
from ipie.config import config
from ipie.utils.backend import arraylib as xp


class ThermofieldHubbardCP(ThermofieldHubbardBase):
    r"""Constrained-path propagation with continuous spin-channel fields.

    Continuous spin-channel fields with the symmetric Trotter split

    .. math::

        B(x) = K e^{\pm\gamma x} K, \qquad K = e^{-\Delta\tau h/2}, \qquad
        \gamma = \sqrt{\Delta\tau U},

    the force bias of the guide, :math:`\bar{x}_i = \gamma (n_{i\uparrow} - n_{i\downarrow})`
    clipped to [-1, 1], and the constraint that the real part of the guide
    overlap ratio stays positive. Branching on the closed trace is described in
    :class:`ThermofieldHubbardBase`.

    Parameters
    ----------
    h_kin, U, timestep, beta, record_fields, guide_mode, fixed_guide_time,
    batched, cache_qinv, trace_branching, energy_shift, trace_exponent :
        As in :class:`ThermofieldHubbardBase`.
    rng :
        Ignored: fields are drawn from `xp.random`, seeded by `set_rng_seed`.
        Kept so that calls that pass a generator still work.
    force_bias_mode : str
        "guide", or "trace" for the force bias of the closed trace (requires
        trace branching). Default "guide".

    Attributes
    ----------
    discrete : bool
        Fields are discrete Hirsch fields. False here.
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
        super().__init__(
            h_kin,
            U,
            timestep,
            beta,
            record_fields=record_fields,
            guide_mode=guide_mode,
            fixed_guide_time=fixed_guide_time,
            batched=batched,
            cache_qinv=cache_qinv,
            trace_branching=trace_branching,
            energy_shift=energy_shift,
            trace_exponent=trace_exponent,
        )
        if force_bias_mode not in ("guide", "trace"):
            raise ValueError("force_bias_mode must be guide or trace.")
        if force_bias_mode == "trace" and not self.trace_branching:
            raise ValueError("Trace force bias requires trace-magnitude branching.")
        self.force_bias_mode = force_bias_mode
        self.hs_type = "continuous_spin"
        self.field_proposal = "force_biased_gaussian"
        self.importance_guide = "same"
        self.gamma = numpy.sqrt(timestep * U)

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
