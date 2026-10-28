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

from ipie.addons.thermal.thermofield.propagation.hubbard_base import ThermofieldHubbardBase
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
    overlap ratio stays positive.

    Parameters
    ----------
    h_kin, U, timestep, beta, record_fields, guide_mode, fixed_guide_time,
    batched, cache_qinv, energy_shift :
        As in :class:`ThermofieldHubbardBase`.
    rng :
        Ignored: fields are drawn from `xp.random`, seeded by `set_rng_seed`.
        Kept so that calls that pass a generator still work.

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
        energy_shift=0.0,
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
            energy_shift=energy_shift,
        )
        self.hs_type = "continuous_spin"
        self.field_proposal = "force_biased_gaussian"
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
            xbar = numpy.clip(self.gamma * (nup - ndn), -1.0, 1.0)
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
            log_w = log_ratio - x @ xbar + 0.5 * xbar @ xbar
            if self.energy_shift != 0.0:
                log_w += self.dt * self.energy_shift
            if numpy.isfinite(log_w):
                self.max_log_weight = max(self.max_log_weight, float(log_w))
                self.min_log_weight = min(self.min_log_weight, float(log_w))
            sgn = numpy.sign(ovlp_sign_new) if not self.constraint else 1.0
            if abs(log_w) > 60.0:
                self.log_weight_bound_count += 1
            walkers.weight[iw] *= sgn * numpy.exp(numpy.clip(log_w, -60.0, 60.0))
            walkers.Qmat[iw] = Q_new
            walkers.set_walker_cache(iw, log_ovlp_new, G_new)

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
        xbar = numpy.clip(self.gamma * (nup - ndn), -1.0, 1.0)
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
        log_w = (
            dlo.real
            - numpy.einsum("ij,ij->i", x, xbar)
            + 0.5 * numpy.einsum("ij,ij->i", xbar, xbar)
        )
        if self.energy_shift != 0.0:
            log_w = log_w + self.dt * self.energy_shift
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
        if use_cache:
            walkers.Qinv[up] = Qi_new[surv]
            walkers.log_det_Q[up] = ldQ_new[surv]
