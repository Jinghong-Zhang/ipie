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
from types import SimpleNamespace

import numpy

from ipie.addons.thermal.thermofield.propagation.hubbard import (
    ThermofieldHubbardCP,
    update_trace_weight,
)
from ipie.addons.thermal.thermofield.utils.linalg import principal_log_phase
from ipie.utils.backend import arraylib as xp


class ThermofieldHubbardDiscrete(ThermofieldHubbardCP):
    r"""Constrained-path propagation with discrete Hirsch spin fields.

    For a slice :math:`B = K D(x) K` with :math:`K = e^{-\Delta\tau h/2}`, let
    :math:`\Lambda` be the guide at the end of the slice. The kinetic step and
    the change of guide come first, with overlap ratio
    :math:`S(\Lambda, K K \Delta) / S(\Lambda_{old}, \Delta)`. At the interaction
    cut the Green's function is

    .. math::

        G_{site} = K^{-1} G(\Lambda, K K \Delta) K,

    and a site update with :math:`d_\sigma = e^{\sigma\gamma x} - 1` has the
    ratios and heat-bath probabilities

    .. math::

        r(x) = \prod_\sigma \left[1 + d_\sigma G_{site, \sigma ii}\right], \qquad
        p(x) = \max(r(x) / 2, 0), \qquad x = \pm 1.

    The field is drawn from p / sum(p), the weight is multiplied by sum(p),
    and each spin is updated by the Sherman-Morrison step
    :math:`G \leftarrow G + d / (1 + d G_{ii}) (I - G)_{:, i} G_{i, :}`, which does
    not need the projector identity :math:`G^2 = G`. At the end of the slice
    :math:`\Delta \leftarrow K D K \Delta` and :math:`G \leftarrow K G_{site} K^{-1}`.

    The constraint applies to the kinetic step and to every partial update.
    Recorded fields are exactly +1 or -1 and rebuild the same B for cyclic
    averaging. With `trace_branching` the same proposal and constraint are
    kept and each completed slice acquires :math:`|Q'/Q|` with
    :math:`Q = \det(I + \Delta) / S`, so the endpoint keeps only the phase of Q.
    No Gaussian force bias or weight clipping is applied.

    Parameters
    ----------
    h_kin, U, timestep, beta, rng, record_fields, guide_mode, fixed_guide_time,
    batched, cache_qinv, energy_shift, trace_branching, trace_exponent :
        As in :class:`ThermofieldHubbardCP`.
    importance_guide : str
        "same": the guide sets both the constraint and the importance
        function. "completion": the fixed guide sets the constraint (the sign
        of every step and the allowed fields), while the allowed fields are
        drawn and the walkers branched with the magnitude of the completion
        guide :math:`|\det(I + e^{-(\beta - \theta) k_T} \Delta)|`, which is
        :math:`|\det(I + \Delta)|` at the endpoint; the endpoint correction is then
        the sign of the trace only. Requires guide_mode "fixed". Default "same".
    bridge_lambda : float
        Weight lambda in [0, 1] of the exact bridge dressing of the importance
        function (Wan and Zhang, arXiv:2507.08937),
        :math:`I = S [(1 - \lambda) + \lambda / (2M) \sum_{k, x} r_k(x)]`, a sum of
        extra sign-free fields of coupling `bridge_theta` on every site at the
        interaction cut. It lifts the zeros of the guide overlap, and the
        dressing is removed from the weight after the last slice. Default 0.0.
    bridge_theta : float
        Coupling of the bridge fields, in (0, pi). Default 1.0.
    """

    discrete = True

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
        trace_branching=False,
        importance_guide="same",
        trace_exponent=1.0,
        bridge_lambda=0.0,
        bridge_theta=1.0,
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
            trace_branching=trace_branching,
            trace_exponent=trace_exponent,
        )
        # Same two spin factors as construct_hirsch_auxiliaries, except for
        # its unshifted-Hamiltonian exp(-dt*U/2) factor (see module docstring).
        with numpy.errstate(over="ignore", invalid="ignore"):
            self.gamma = numpy.arccosh(numpy.exp(0.5 * timestep * U))
            self.auxf = numpy.exp(self.gamma * numpy.array([[1.0, -1.0], [-1.0, 1.0]]))
        if not numpy.isfinite(self.auxf).all():
            raise ValueError("Hirsch factors overflow; reduce timestep*U.")
        self.delta = self.auxf - 1.0
        self.field_log_scalar = numpy.zeros(2, dtype=complex)
        self.hs_type = "discrete_spin"
        self.field_proposal = "site_conditional_guide"
        self.force_bias_mode = "none"
        self.forbidden_field_count = 0
        # importance_guide="completion": keep the fixed guide as the path
        # constraint (kinetic-step sign and allowed site fields) but draw the
        # allowed fields and branch with the magnitude of the COMPLETION guide
        # overlap |det(1 + e^{-(beta-theta) k_T} Delta)|, which equals
        # |det(1 + Delta)| at the endpoint.  The endpoint correction Q is then
        # the sign of the physical trace only, so its heavy-tailed magnitude
        # never enters the population.  Exact for the same constrained
        # ensemble: only the importance function changes.
        if importance_guide not in ("same", "completion"):
            raise ValueError("importance_guide must be 'same' or 'completion'.")
        if importance_guide == "completion":
            if guide_mode != "fixed":
                raise ValueError("importance_guide='completion' requires guide_mode='fixed'.")
            if self.trace_branching:
                raise ValueError("importance_guide='completion' and trace_branching are exclusive.")
            self.field_proposal = "site_conditional_completion_importance"
        self.importance_guide = importance_guide
        # Exact bridge (Wan and Zhang, arXiv:2507.08937) placed in the importance
        # function: I = S * dress, with the positive dressing
        #     dress = (1 - lambda) + lambda/(2M) sum_{k, x} <extra HS field x of
        #             coupling bridge_theta on site k, at the interaction cut>,
        # each term the same sign-free field type as the propagation.  I vanishes
        # only where all 2M + 1 terms vanish, which lifts the guide-overlap zeros
        # that the field proposals otherwise avoid.  The dressing is removed from
        # the weight after the last slice, so the endpoint Q = Tr(B)/S in the
        # driver becomes Tr(B)/I, bounded by 1/(1 - lambda) for a completion guide.
        if not (numpy.isfinite(bridge_lambda) and 0.0 <= bridge_lambda <= 1.0):
            raise ValueError("bridge_lambda must lie in [0, 1].")
        if not (numpy.isfinite(bridge_theta) and 0.0 < bridge_theta < numpy.pi):
            raise ValueError("bridge_theta must lie in (0, pi).")
        self.bridge_lambda = float(bridge_lambda)
        self.bridge_theta = float(bridge_theta)
        # Which bridge check rejected a walker, and the largest relative imaginary
        # part seen in the (analytically real) dressing and dressed probabilities.
        self.bridge_rejections = dict(
            kinetic_dress=0, site_probability=0, zero_overlap=0, site_dress=0
        )
        self.bridge_max_relative_imag = 0.0
        # Relative size allowed for the rounding imaginary part of the analytically
        # real field probabilities and dressing. The dressed walk visits states near
        # guide-overlap zeros, where the cut Green function is large and rounding
        # reaches ~1e-9; a genuine error would be O(1).
        self.real_tolerance = 1e-6 if self.bridge_lambda > 0.0 else 1e-10
        if self.bridge_lambda > 0.0:
            if self.trace_branching or importance_guide != "same":
                raise ValueError("The bridge dressing requires ordinary guide branching.")
            self.bridge_nslices = int(round(beta / timestep))
            if abs(self.bridge_nslices * timestep - beta) > 1e-9 * max(1.0, beta):
                raise ValueError("The bridge dressing needs beta to be a whole number of slices.")
        self._set_bridge_factors()

    def _set_bridge_factors(self):
        """Bridge terms exp(theta x (n_up - n_dn)): real spin fields, a positive weight each."""
        signs = numpy.array([1.0, -1.0])
        self.bridge_delta = numpy.exp(self.bridge_theta * numpy.outer(signs, [1.0, -1.0])) - 1.0
        self.bridge_scalar = numpy.ones(2, dtype=complex)

    def _bridge_dress(self, green):
        """dress = I/S at the interaction cut; green is (walkers, 2, M, M)."""
        gkk = numpy.diagonal(green, axis1=2, axis2=3)
        terms = numpy.prod(1.0 + self.bridge_delta[None, :, :, None] * gkk[:, None], axis=2)
        lam = self.bridge_lambda
        return (1.0 - lam) + lam * numpy.einsum("nym,y->n", terms, self.bridge_scalar) / (
            2 * self.nfields
        )

    def _site_bridge_probabilities(self, green, site, ids, dress):
        """Dressed field probabilities: (1/2) scalar(x) I(x) / I, from exact rank-two ratios.

        For field x on `site` and bridge x' on mode k, per spin the determinant
        ratio is (1 + d G_jj)(1 + e G_kk) - d e G_jk G_kj (k != j), and
        1 + ((1 + d)(1 + e) - 1) G_jj for k = j, with G the cut Green function.
        """
        g = green[ids]
        gjj = g[:, :, site, site]
        gkk = numpy.diagonal(g, axis1=2, axis2=3)
        d = self.delta[None, :, None, :, None]
        e = self.bridge_delta[None, None, :, :, None]
        ratios = (1.0 + d * gjj[:, None, None, :, None]) * (
            1.0 + e * gkk[:, None, None]
        ) - d * e * (g[:, :, site, :] * g[:, :, :, site])[:, None, None]
        combined = (1.0 + self.delta)[:, None, :] * (1.0 + self.bridge_delta)[None, :, :] - 1.0
        ratios[..., site] = 1.0 + combined[None] * gjj[:, None, None, :]
        bridged = numpy.einsum("nxym,y->nx", numpy.prod(ratios, axis=3), self.bridge_scalar) / (
            2 * self.nfields
        )
        bare = numpy.prod(1.0 + self.delta[None] * gjj[:, None, :], axis=2)
        lam = self.bridge_lambda
        importance = (1.0 - lam) * bare + lam * bridged
        return 0.5 * numpy.exp(self.field_log_scalar)[None, :] * importance / dress[:, None]

    def init_walker_cache(self, walkers, trial):
        old_alive = walkers.weight != 0
        super().init_walker_cache(walkers, trial)
        self.invalid_weight_count += int((old_alive & (walkers.weight == 0)).sum())
        if self.bridge_lambda > 0.0:
            # log(I/S) at the last interaction cut; zero on a fresh path (I = S at theta = 0).
            if "log_dress" not in walkers.buff_names:
                walkers.add_walker_array(
                    "log_dress", numpy.zeros(walkers.nwalkers, dtype=numpy.complex128)
                )
            else:
                walkers.log_dress[:] = 0.0
        if self.importance_guide == "completion":
            # At theta = 0 the completion importance guide (time beta) and the
            # fixed constraint guide coincide: both caches start from the same
            # overlap; log_ovlp keeps only the magnitude from here on.
            cache = walkers.log_ovlp.copy()
            if "log_ovlp_constraint" not in walkers.buff_names:
                walkers.add_walker_array("log_ovlp_constraint", cache)
            else:
                walkers.log_ovlp_constraint[:] = cache
            walkers.log_ovlp[:] = cache.real

    def _completion_scale(self, step):
        """Guide time of the completion importance guide after slice ``step``."""
        return max(0.0, self.beta - (step + 1) * self.dt)

    def refresh_cache_after_stabilize(self, walkers, trial, l):
        if self.importance_guide != "completion":
            return super().refresh_cache_after_stabilize(walkers, trial, l)
        alive = numpy.flatnonzero(walkers.weight != 0.0)
        if alive.size == 0:
            return
        lo_f, G = trial.calc_log_overlap_and_greens_function_scaled_batched(
            walkers.Qmat[alive], walkers.log_d[alive], walkers.Tmat[alive], self.guide_scale(l)
        )
        lo_c, _ = trial.calc_log_overlap_and_greens_function_scaled_batched(
            walkers.Qmat[alive],
            walkers.log_d[alive],
            walkers.Tmat[alive],
            self._completion_scale(l),
        )
        ok = numpy.isfinite(lo_f) & numpy.isfinite(lo_c) & numpy.isfinite(G).all(axis=(1, 2, 3))
        walkers.weight[alive[~ok]] = 0.0
        up = alive[ok]
        eye = numpy.eye(walkers.nbasis)
        walkers.log_ovlp_constraint[up] = principal_log_phase(lo_f[ok])
        walkers.log_ovlp[up] = lo_c[ok].real
        walkers.Ga[up] = eye - G[ok, 0]
        walkers.Gb[up] = eye - G[ok, 1]

    def _site_probabilities(self, green, site, ids=None):
        density = green[:, :, site, site]
        if ids is not None:
            # Select the two diagonal entries, not a copy of every matrix.
            density = density[ids]
        return 0.5 * numpy.prod(1.0 + self.delta[None, :, :] * density[:, None, :], axis=2)

    @staticmethod
    def _site_green_update(green, site, delta):
        """Left row scaling at the interaction cut; green is (walkers,2,M,M)."""
        column = -green[:, :, :, site].copy()
        column[:, :, site] += 1.0
        row = green[:, :, site, :].copy()
        ratio = 1.0 + delta * green[:, :, site, site]
        green += (delta / ratio)[:, :, None, None] * column[:, :, :, None] * row[:, :, None, :]

    # Per-field hooks of _advance. A field is one site here (spin factors on
    # the same diagonal entry of both spin blocks); subclasses with other
    # field layouts override these four methods together.
    def _field_probabilities(self, green, field, ids=None):
        return self._site_probabilities(green, field, ids)

    def _field_green_update(self, green, field, chosen):
        self._site_green_update(green, field, self.delta[chosen])

    def _field_factors(self, factors, active, field, chosen):
        factors[active, :, field] = self.auxf[chosen]

    def _field_log_scalar(self, field, chosen):
        return self.field_log_scalar[chosen]

    def _advance(self, walkers, trial, step, ids, uniforms):
        if len(ids) == 0:
            return
        state = self._slice_start(walkers, trial, step, ids)
        self._slice_fields(walkers, step, ids, uniforms, state)
        self._slice_finish(walkers, trial, step, ids, state)

    def _slice_start(self, walkers, trial, step, ids):
        """Kinetic step, guide contraction at the interaction cut, kill and weight set-up."""
        log_s_imp = green_imp = dress = None
        eye = numpy.eye(walkers.nbasis)
        two_guides = self.importance_guide == "completion"
        q_old = walkers.Qmat[ids]
        q_kin = self.BK2 @ (self.BK2 @ q_old)
        cached = self.cache_qinv and walkers.Qinv is not None
        qi_old = walkers.Qinv[ids] if cached else None
        ld_old = walkers.log_det_Q[ids] if cached else None
        qi_kin = (qi_old @ self.BK2inv) @ self.BK2inv if cached else None
        ld_kin = ld_old + 2.0 * self.logdet_BK2 if cached else None
        # Constraint guide (the guide selected by guide_mode).
        log_s, green = trial.calc_log_overlap_and_greens_function_scaled_batched(
            q_kin,
            walkers.log_d[ids],
            walkers.Tmat[ids],
            self.guide_scale(step),
            Qinv=qi_kin,
            log_det_Q=ld_kin,
        )
        good = numpy.isfinite(log_s) & numpy.isfinite(green).all(axis=(1, 2, 3))
        if two_guides:
            # Sign constraint from the fixed guide (cache log_ovlp_constraint);
            # branching magnitude from the completion guide (cache log_ovlp).
            change = principal_log_phase(log_s - walkers.log_ovlp_constraint[ids])
            log_s_imp, green_imp = trial.calc_log_overlap_and_greens_function_scaled_batched(
                q_kin,
                walkers.log_d[ids],
                walkers.Tmat[ids],
                self._completion_scale(step),
                Qinv=qi_kin,
                log_det_Q=ld_kin,
            )
            good &= numpy.isfinite(log_s_imp) & numpy.isfinite(green_imp).all(axis=(1, 2, 3))
            log_weight = (log_s_imp - walkers.log_ovlp[ids]).real + self.dt * self.energy_shift
            # Only |S_c| is carried: its sign is recovered at the endpoint as
            # the sign of det(1 + Delta), which the driver evaluates as Q.
            log_s_imp = log_s_imp.real.astype(numpy.complex128)
            green_imp = self.BK2inv @ green_imp @ self.BK2
        else:
            change = principal_log_phase(log_s - walkers.log_ovlp[ids])
            log_weight = change.real + self.dt * self.energy_shift
        self.invalid_weight_count += int((~good).sum())
        live = good & (numpy.cos(change.imag) > 0.0)
        self.step_count += int(good.sum())
        self.kill_count += int((good & ~live).sum())
        # The contraction at D, with the final K folded into the left guide.
        green = self.BK2inv @ green @ self.BK2
        bridge = self.bridge_lambda > 0.0
        if bridge:
            # The kinetic step moves the cut: I_new/I_old = (S_new/S_old)(dress_new/dress_old).
            with numpy.errstate(over="ignore", invalid="ignore"):
                dress = self._bridge_dress(green)
            rel = numpy.abs(dress.imag) / numpy.maximum(1.0, numpy.abs(dress.real))
            good_d = numpy.isfinite(dress) & (dress.real > 0.0) & (rel <= self.real_tolerance)
            if numpy.isfinite(rel[live]).any():
                self.bridge_max_relative_imag = max(
                    self.bridge_max_relative_imag, float(numpy.nanmax(rel[live]))
                )
            self.bridge_rejections["kinetic_dress"] += int((live & ~good_d).sum())
            self.invalid_weight_count += int((live & ~good_d).sum())
            live &= good_d
            dress = numpy.where(good_d, dress.real, 1.0)
            log_weight = log_weight + numpy.log(dress) - walkers.log_dress[ids].real
        factors = numpy.ones((len(ids), 2, walkers.nbasis), dtype=self.auxf.dtype)
        return SimpleNamespace(
            eye=eye,
            two_guides=two_guides,
            bridge=bridge,
            q_old=q_old,
            cached=cached,
            qi_old=qi_old,
            ld_old=ld_old,
            log_s=log_s,
            green=green,
            log_s_imp=log_s_imp,
            green_imp=green_imp,
            log_weight=log_weight,
            live=live,
            dress=dress,
            factors=factors,
        )

    def _slice_fields(self, walkers, step, ids, uniforms, state):
        """Sequential heat-bath sampling of every field; arrays in ``state`` change in place."""
        two_guides = state.two_guides
        bridge = state.bridge
        log_s = state.log_s
        green = state.green
        log_s_imp = state.log_s_imp
        green_imp = state.green_imp
        log_weight = state.log_weight
        live = state.live
        dress = state.dress
        factors = state.factors
        for site in range(self.nfields):
            active = numpy.flatnonzero(live)
            if len(active) == 0:
                break
            with numpy.errstate(over="ignore", invalid="ignore"):
                probabilities = self._field_probabilities(green, site, active)
                if two_guides:
                    prob_imp = self._field_probabilities(green_imp, site, active)
                if bridge:
                    prob_bridge = self._site_bridge_probabilities(
                        green, site, active, dress[active]
                    )
            good_p = numpy.isfinite(probabilities).all(axis=1) & (
                numpy.abs(probabilities.imag)
                <= self.real_tolerance * numpy.maximum(1.0, numpy.abs(probabilities.real))
            ).all(axis=1)
            if bridge:
                rel = (
                    numpy.abs(prob_bridge.imag) / numpy.maximum(1.0, numpy.abs(prob_bridge.real))
                ).max(axis=1)
                if numpy.isfinite(rel).any():
                    self.bridge_max_relative_imag = max(
                        self.bridge_max_relative_imag, float(numpy.nanmax(rel))
                    )
                good_b = numpy.isfinite(prob_bridge).all(axis=1) & (rel <= self.real_tolerance)
                self.bridge_rejections["site_probability"] += int((good_p & ~good_b).sum())
                good_p &= good_b
            if two_guides:
                good_p &= numpy.isfinite(prob_imp).all(axis=1) & (
                    numpy.abs(prob_imp.imag) <= 1e-10 * numpy.maximum(1.0, numpy.abs(prob_imp.real))
                ).all(axis=1)
            self.invalid_weight_count += int((~good_p).sum())
            live[active[~good_p]] = False
            probabilities = probabilities[good_p].real
            if two_guides:
                prob_imp = prob_imp[good_p].real
            if bridge:
                prob_bridge = prob_bridge[good_p].real
            active = active[good_p]
            self.step_count += len(active)
            self.forbidden_field_count += int(
                ((prob_bridge if bridge else probabilities) <= 0).sum()
            )
            if two_guides:
                # Fields forbidden by the fixed constraint get zero probability;
                # the allowed ones are drawn with the completion-guide magnitude.
                positive = numpy.where(probabilities > 0.0, numpy.abs(prob_imp), 0.0)
            else:
                positive = numpy.maximum(prob_bridge if bridge else probabilities, 0.0)
            with numpy.errstate(over="ignore", invalid="ignore"):
                norm = positive.sum(axis=1)
            valid_norm = numpy.isfinite(norm)
            self.invalid_weight_count += int((~valid_norm).sum())
            self.kill_count += int((valid_norm & (norm == 0)).sum())
            accepted = valid_norm & (norm > 0)
            live[active[~accepted]] = False
            active = active[accepted]
            norm = norm[accepted]
            positive = positive[accepted]
            chosen = (uniforms[active, site] >= positive[:, 0] / norm).astype(int)
            # Usually every walker survives the site constraint. In that case
            # update the local Green-function stack directly. Advanced indexing
            # already makes a copy when only a subset survives.
            all_active = len(active) == len(ids)
            update = green if all_active else green[active]
            with numpy.errstate(divide="ignore", over="ignore", invalid="ignore"):
                self._field_green_update(update, site, chosen)
            valid_g = numpy.isfinite(update).all(axis=(1, 2, 3))
            if bridge:
                # The undressed overlap must stay nonzero; the dressing follows the updated cut.
                nonzero = probabilities[accepted][numpy.arange(len(chosen)), chosen] > 0.0
                self.bridge_rejections["zero_overlap"] += int((valid_g & ~nonzero).sum())
                valid_g &= nonzero
                with numpy.errstate(over="ignore", invalid="ignore"):
                    dress_new = self._bridge_dress(update)
                good_new = numpy.isfinite(dress_new) & (dress_new.real > 0.0)
                self.bridge_rejections["site_dress"] += int((valid_g & ~good_new).sum())
                valid_g &= good_new
                dress[active] = numpy.where(valid_g, dress_new.real, 1.0)
            if two_guides:
                update_imp = green_imp if all_active else green_imp[active]
                with numpy.errstate(divide="ignore", over="ignore", invalid="ignore"):
                    self._field_green_update(update_imp, site, chosen)
                valid_g &= numpy.isfinite(update_imp).all(axis=(1, 2, 3))
            self.invalid_weight_count += int((~valid_g).sum())
            live[active[~valid_g]] = False
            if not all_active:
                green[active] = update
                if two_guides:
                    green_imp[active] = update_imp
            self._field_factors(factors, active, site, chosen)
            log_weight[active] += numpy.log(norm)
            sel = numpy.arange(len(chosen))
            log_s[active] += (
                numpy.log(2.0)
                + numpy.log(probabilities[accepted][sel, chosen])
                - self._field_log_scalar(site, chosen)
            )
            if two_guides:
                log_s_imp[active] += (
                    numpy.log(2.0)
                    + numpy.log(positive[sel, chosen])
                    - self._field_log_scalar(site, chosen).real
                )
            if self.record_fields:
                walkers.field_hist[ids[active], step, site] = 1 - 2 * chosen

    def _slice_finish(self, walkers, trial, step, ids, state):
        """Final kinetic half, weights, constraint caches and Q-inverse caches."""
        eye = state.eye
        two_guides = state.two_guides
        bridge = state.bridge
        q_old = state.q_old
        cached = state.cached
        qi_old = state.qi_old
        ld_old = state.ld_old
        log_s = state.log_s
        green = state.green
        log_s_imp = state.log_s_imp
        log_weight = state.log_weight
        live = state.live
        dress = state.dress
        factors = state.factors
        kept = numpy.flatnonzero(live)
        old_weights = walkers.weight[ids].copy()
        walkers.weight[ids] = 0.0
        if len(kept) == 0:
            return
        final_g = self.BK2 @ green[kept] @ self.BK2inv
        final_q = self.BK2 @ (factors[kept, :, :, None] * (self.BK2 @ q_old[kept]))
        good_final = (
            numpy.isfinite(final_g).all(axis=(1, 2, 3))
            & numpy.isfinite(final_q).all(axis=(1, 2, 3))
            & numpy.isfinite(log_s[kept])
        )
        if two_guides:
            good_final &= numpy.isfinite(log_s_imp[kept])
        self.invalid_weight_count += int((~good_final).sum())
        log_trace = None
        if self.trace_branching:
            source = ids[kept]
            log_trace = trial.calc_log_trace_batched(
                final_q, walkers.log_d[source], walkers.Tmat[source]
            )
            finite_trace = numpy.isfinite(log_trace) & numpy.isfinite(walkers.log_tr[source])
            self.trace_invalid_count += int((good_final & ~finite_trace).sum())
            good_final &= finite_trace
            # The site heat bath already includes the full guide/proposal
            # factor. Replace only its telescoping magnitude, not the
            # guide-based field probabilities or sub-slice constraints.
            log_weight[kept] += self.trace_exponent * (
                log_trace.real
                - log_s[kept].real
                - walkers.log_tr[source].real
                + walkers.log_ovlp[source].real
            )
        if bridge:
            final_dress = dress[kept]
            if step == self.bridge_nslices - 1:
                # Endpoint reweighting 1/dress: the driver's Q = Tr(B)/S then gives Tr(B)/I.
                log_weight[kept] -= numpy.log(final_dress)
                final_dress = numpy.ones_like(final_dress)
        finite_weight = good_final & numpy.isfinite(log_weight[kept])
        if finite_weight.any():
            self.max_log_weight = max(
                self.max_log_weight, float(log_weight[kept][finite_weight].max())
            )
            self.min_log_weight = min(
                self.min_log_weight, float(log_weight[kept][finite_weight].min())
            )
        updated, failures = update_trace_weight(old_weights[kept], log_weight[kept])
        self.invalid_weight_count += failures
        if self.trace_branching:
            self.trace_weight_invalid_count += failures
        updated[~good_final] = 0.0
        surviving = updated != 0
        target = ids[kept[surviving]]
        walkers.weight[ids[kept]] = updated
        walkers.Qmat[target] = final_q[surviving]
        if two_guides:
            walkers.log_ovlp[target] = log_s_imp[kept[surviving]]
            walkers.log_ovlp_constraint[target] = principal_log_phase(log_s[kept[surviving]])
        else:
            walkers.log_ovlp[target] = principal_log_phase(log_s[kept[surviving]])
        walkers.Ga[target] = eye - final_g[surviving, 0]
        walkers.Gb[target] = eye - final_g[surviving, 1]
        if bridge:
            walkers.log_dress[target] = numpy.log(final_dress[surviving])
        if self.trace_branching:
            walkers.log_tr[target] = log_trace[surviving]
        if cached:
            subset = kept[surviving]
            walkers.Qinv[target] = (
                (qi_old[subset] @ self.BK2inv) / factors[subset, :, None, :]
            ) @ self.BK2inv
            walkers.log_det_Q[target] = (
                ld_old[subset] + 2.0 * self.logdet_BK2 + numpy.log(factors[subset]).sum(axis=-1)
            )

    def _uniforms(self, walkers):
        if not self.constraint or self.force_bias_mode != "none":
            raise ValueError(
                "Discrete Hirsch propagation requires guide CP and no Gaussian force bias."
            )
        ids = numpy.flatnonzero(walkers.weight != 0)
        # Draw all site uniforms of the live population in walker-major order,
        # so the batched and per-walker paths use the same random numbers.
        return ids, xp.random.random((len(ids), self.nfields))

    def _propagate_walkers_batched(self, walkers, trial, l):
        ids, uniforms = self._uniforms(walkers)
        self._advance(walkers, trial, l, ids, uniforms)

    def _propagate_walkers_scalar(self, walkers, trial, l):
        ids, uniforms = self._uniforms(walkers)
        for j, iw in enumerate(ids):
            self._advance(walkers, trial, l, numpy.array([iw]), uniforms[j : j + 1])


class ThermofieldHubbardCharge(ThermofieldHubbardDiscrete):
    r"""Sign-free discrete charge fields for half-filled completion or fixed guides.

    For N=n_up+n_down and cos(alpha)=exp(-dt*U/2), the local identity is

        exp[-dt*U*(n_up*n_down-N/2)]
          = 1/2 sum_x exp(dt*U/2-i*alpha*x) exp(i*alpha*x*N).

    Both spin matrices acquire the same complex phase. The scalar prefactor
    is essential: the determinant alone is generally complex. Conditional
    sampling includes this scalar, while log_ovlp caches the bare determinant
    of the one-body matrices. Scalar factors cancel from trace/guide ratios
    and normalized observables; they remain in the accumulated path weight.

    This implementation requires real bipartite hopping, zero shifted
    chemical potential, and a spin-symmetric particle-hole-symmetric trial.
    In this domain scalar times determinant is nonnegative at every partial
    update. No phaseless projection or magnitude replacement is applied.
    The guide may be the completion exp[-(beta-tau) k_T] or a fixed
    exp(-t k_T): both are particle-hole symmetric, so every partial overlap
    stays a real square and the fixed-guide endpoint correction
    Q = Tr(B)/Tr(Lambda B) is real and nonnegative.
    Doped, pinned and trace-branching uses are deliberately rejected rather
    than extending this sign-free reference implicitly.
    """

    def __init__(self, h_kin, U, timestep, beta, rng=None, **kwargs):
        super().__init__(h_kin, U, timestep, beta, rng=rng, **kwargs)
        if self.trace_branching or self.importance_guide != "same":
            raise ValueError("Charge reference requires ordinary guide branching.")
        h = numpy.asarray(h_kin).real
        if not numpy.allclose(h[0], h[1], atol=1e-12, rtol=0) or not numpy.allclose(
            numpy.diag(h[0]), 0, atol=1e-12, rtol=0
        ):
            raise ValueError(
                "Charge reference requires zero chemical potential and no spin pinning."
            )
        # Infer the particle-hole sublattices from the supplied hopping graph.
        eta = numpy.zeros(self.nfields)
        for root in range(self.nfields):
            if eta[root] != 0:
                continue
            eta[root] = 1
            pending = [root]
            while pending:
                i = pending.pop()
                for j in numpy.flatnonzero(numpy.abs(h[0, i]) > 1e-12):
                    if eta[j] == 0:
                        eta[j] = -eta[i]
                        pending.append(j)
                    elif eta[j] == eta[i]:
                        raise ValueError("Charge reference requires bipartite hopping.")
        self.sublattice_sign = eta
        self.gamma = numpy.arccos(numpy.exp(-0.5 * timestep * U))
        signs = numpy.array([1.0, -1.0])
        self.auxf = numpy.repeat(numpy.exp(1j * self.gamma * signs)[:, None], 2, axis=1)
        self.delta = self.auxf - 1.0
        self.field_log_scalar = 0.5 * timestep * U - 1j * self.gamma * signs
        self.hs_type = "discrete_charge"
        self.field_proposal = "site_conditional_charge_guide"
        self._set_bridge_factors()

    def construct_slice_propagators(self, fields):
        r"""Slice propagators :math:`B(x)` for recorded charge fields.

        Both spins acquire the same phase :math:`e^{i\gamma x}`.

        Parameters
        ----------
        fields : :class:`numpy.ndarray`
            Fields of one slice, shape (nwalkers, nfields).

        Returns
        -------
        B : :class:`numpy.ndarray`
            Shape (nwalkers, 2, nbasis, nbasis).
        """
        dv = numpy.exp(1j * self.gamma * fields)
        B = numpy.empty((len(fields), 2, self.nfields, self.nfields), dtype=numpy.complex128)
        B[:, 0] = self.BK2[0] @ (dv[:, :, None] * self.BK2[0])
        B[:, 1] = self.BK2[1] @ (dv[:, :, None] * self.BK2[1])
        return B

    def _set_bridge_factors(self):
        """Bridge terms exp(i theta x (N - 1)): extra charge fields, each a real square here."""
        signs = numpy.array([1.0, -1.0])
        self.bridge_delta = (
            numpy.repeat(numpy.exp(1j * self.bridge_theta * signs)[:, None], 2, axis=1) - 1.0
        )
        self.bridge_scalar = numpy.exp(-1j * self.bridge_theta * signs)

    def init_walker_cache(self, walkers, trial):
        k = numpy.array([(trial.C_T[s] * trial.eps_T[s]) @ trial.C_T[s].conj().T for s in range(2)])
        eta = self.sublattice_sign
        if (
            not numpy.allclose(k[0], k[1], atol=1e-10, rtol=0)
            or not numpy.allclose(k.imag, 0, atol=1e-10, rtol=0)
            or not numpy.allclose(eta[:, None] * k[0] * eta[None, :], -k[0], atol=1e-10, rtol=0)
        ):
            raise ValueError("Charge reference requires a spin-symmetric particle-hole trial.")
        super().init_walker_cache(walkers, trial)

    def _site_probabilities(self, green, site, ids=None):
        return (
            super()._site_probabilities(green, site, ids)
            * numpy.exp(self.field_log_scalar)[None, :]
        )
