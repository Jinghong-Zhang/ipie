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
"""Phaseless propagation with hand-derived forward-mode tangents.

Mirrors ipie.addons.adafqmc.propagation.propagator line for line; every
lambda-dependent quantity is carried as a (value, tangent) pair with explicit
product/chain rules.  Randomness is always injected (see utils.fields).
"""

import math

import numpy as np
import scipy.linalg

from ipie.addons.analytical_gradient.estimators.estimator import (
    local_energy_with_tangent,
    weighted_energy_with_tangent,
)
from ipie.addons.analytical_gradient.walkers.rhf_walkers import (
    GradWalkers,
    reorthogonalize,
    stochastic_reconfiguration,
)


def compute_exph1_with_tangent(h1e_mod, dh1e_mod, chol, dchol, mf_shift, dmf_shift, dt):
    """exp(-dt/2 * H1_mf) and its tangent via the Frechet derivative.

    H1_mf = h1e_mod + sum_g Im(mf_shift)_g chol_g  (mean-field subtraction).
    """
    M = h1e_mod + np.einsum("p,pij->ij", mf_shift.imag, chol)
    dM = (
        dh1e_mod
        + np.einsum("p,pij->ij", dmf_shift.imag, chol)
        + np.einsum("p,pij->ij", mf_shift.imag, dchol)
    )
    A = -0.5 * dt * M
    dA = -0.5 * dt * dM
    dtype = np.promote_types(A.dtype, dA.dtype)
    expH1, dexpH1 = scipy.linalg.expm_frechet(
        A.astype(dtype), dA.astype(dtype), compute_expm=True
    )
    return expH1, dexpH1


def apply_bound_force_bias_with_tangent(xbar, dxbar, max_bound=1.0):
    """Componentwise cap |xbar| <= max_bound by rescaling to unit modulus.

    Mirrors adafqmc apply_bound_force_bias (rescale to |x|=1, 1e-13 guard).
    Tangent of y = x/|x|:  dy = dx/|x| - x Re(conj(x) dx)/|x|^3.
    Returns (xbar, dxbar, cap_mask).
    """
    absxbar = np.abs(xbar)
    idx_to_rescale = absxbar > max_bound
    nonzeros = absxbar > 1e-13
    safe_abs = np.where(nonzeros, absxbar, 1.0)
    rescaled = np.where(nonzeros, xbar / safe_abs, xbar)
    drescaled = np.where(
        nonzeros,
        dxbar / safe_abs - xbar * np.real(np.conj(xbar) * dxbar) / safe_abs**3,
        dxbar,
    )
    xbar_out = np.where(idx_to_rescale, rescaled, xbar)
    dxbar_out = np.where(idx_to_rescale, drescaled, dxbar)
    return xbar_out, dxbar_out, idx_to_rescale


def construct_vhs_with_tangent(isqrtt, chol, dchol, xshifted, dxshifted):
    """VHS = i sqrt(dt) sum_g xshifted_g chol_g, with product-rule tangent."""
    vhs = isqrtt * np.einsum("wp,pij->wij", xshifted, chol)
    dvhs = isqrtt * (
        np.einsum("wp,pij->wij", dxshifted, chol)
        + np.einsum("wp,pij->wij", xshifted, dchol)
    )
    return vhs, dvhs


def apply_taylor_with_tangent(taylor_order, vhs, dvhs, phi, dphi):
    """exp(VHS) phi via order-n Taylor series, with the coupled tangent recursion.

    The tangent update must consume the pre-update term: dT_n uses T_{n-1}.
    Differentiates the truncated series itself (the algorithm's definition).
    """
    T = phi
    dT = dphi
    phi_out = phi.copy()
    dphi_out = dphi.copy()
    for n in range(1, taylor_order + 1):
        dT = (dvhs @ T + vhs @ dT) / n
        T = (vhs @ T) / n
        phi_out = phi_out + T
        dphi_out = dphi_out + dT
    return phi_out, dphi_out


class GradPropagator:
    """One-step phaseless propagation of (phi, dphi, w, dw), adafqmc-matched."""

    def __init__(
        self,
        dt,
        ham,
        trial,
        prop_block_size,
        taylor_order=6,
        fbbound=1.0,
        apply_weight_bound=True,
        debug=False,
    ):
        self.dt = dt
        self.sqrtdt = math.sqrt(dt)
        self.isqrtt = 1j * math.sqrt(dt)
        self.prop_block_size = prop_block_size
        self.taylor_order = taylor_order
        self.fbbound = fbbound
        self.apply_weight_bound = apply_weight_bound
        self.debug = debug
        self.diagnostics = []

        self.mf_shift = 2j * np.einsum("pij,ij->p", ham.chol, trial.G)
        self.dmf_shift = 2j * (
            np.einsum("pij,ij->p", ham.dchol, trial.G)
            + np.einsum("pij,ij->p", ham.chol, trial.dG)
        )
        self.expH1, self.dexpH1 = compute_exph1_with_tangent(
            ham.h1e_mod, ham.dh1e_mod, ham.chol, ham.dchol, self.mf_shift, self.dmf_shift, dt
        )
        self.h0shift = ham.enuc - 0.5 * np.dot(self.mf_shift.imag, self.mf_shift.imag)
        self.dh0shift = -np.dot(self.mf_shift.imag, self.dmf_shift.imag)
        # NOT detached in adafqmc (propagator.py:165): the trial energy's
        # lambda-tangent flows into constant_term for the whole first
        # sub-block, until the first sub-block energy update zeroes it.
        self.energy_estimate, self.denergy_estimate = trial.eval_energy_with_tangent(ham)

    def propagate_walkers(self, walkers, ham, trial, x):
        """One propagation step; x is the injected (nwalkers, nchol) field array."""
        phi, dphi = walkers.phi, walkers.dphi

        # Force bias from the pre-step Green's function.
        Ghalf, dGhalf, S_old, dS_old, _ = trial.get_ghalf_with_tangent(phi, dphi)
        vbias, dvbias = trial.calc_force_bias_with_tangent(Ghalf, dGhalf)
        xbar = -self.sqrtdt * (1j * vbias - self.mf_shift)
        dxbar = -self.sqrtdt * (1j * dvbias - self.dmf_shift)
        xbar, dxbar, fb_cap_mask = apply_bound_force_bias_with_tangent(
            xbar, dxbar, self.fbbound
        )

        # First half one-body step (tangent consumes pre-update phi).
        dphi = np.einsum("pq,wqr->wpr", self.dexpH1, phi) + np.einsum(
            "pq,wqr->wpr", self.expH1, dphi
        )
        phi = np.einsum("pq,wqr->wpr", self.expH1, phi)

        # Two-body step.
        xshifted = x - xbar
        dxshifted = -dxbar
        vhs, dvhs = construct_vhs_with_tangent(
            self.isqrtt, ham.chol, ham.dchol, xshifted, dxshifted
        )
        phi, dphi = apply_taylor_with_tangent(self.taylor_order, vhs, dvhs, phi, dphi)

        # Second half one-body step.
        dphi = np.einsum("pq,wqr->wpr", self.dexpH1, phi) + np.einsum(
            "pq,wqr->wpr", self.expH1, dphi
        )
        phi = np.einsum("pq,wqr->wpr", self.expH1, phi)

        # Weight factor, assembled in log space.  RHF: overlap ratio squared.
        S_new, dS_new = trial.calc_overlap_with_tangent(phi, dphi)
        sgn_old, logabs_old = np.linalg.slogdet(S_old)
        sgn_new, logabs_new = np.linalg.slogdet(S_new)
        logratio = 2.0 * (
            (logabs_new + np.log(sgn_new.astype(np.complex128)))
            - (logabs_old + np.log(sgn_old.astype(np.complex128)))
        )
        dlogratio = 2.0 * (
            np.einsum("wii->w", np.linalg.solve(S_new, dS_new))
            - np.einsum("wii->w", np.linalg.solve(S_old, dS_old))
        )
        logfb = np.einsum("wp,wp->w", x, xbar) - 0.5 * np.einsum("wp,wp->w", xbar, xbar)
        dlogfb = np.einsum("wp,wp->w", x, dxbar) - np.einsum("wp,wp->w", xbar, dxbar)
        logmf = -self.sqrtdt * np.einsum("wp,p->w", xshifted, self.mf_shift)
        dlogmf = -self.sqrtdt * (
            np.einsum("wp,p->w", dxshifted, self.mf_shift)
            + np.einsum("wp,p->w", xshifted, self.dmf_shift)
        )
        logct = self.dt * (self.energy_estimate - self.h0shift)
        dlogct = self.dt * (self.denergy_estimate - self.dh0shift)

        # Phase from overlap ratio and mean-field factor only (adafqmc line 238);
        # cos/sin evaluated branch-cut free from the accumulated Im parts.
        theta = np.imag(logratio + logmf)
        dtheta = np.imag(dlogratio + dlogmf)
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        logA = logratio + logfb + logmf + logct
        dlogA = dlogratio + dlogfb + dlogmf + dlogct
        absA = np.exp(np.real(logA))
        cos_mask = costheta > 0.0
        factor = absA * costheta * cos_mask
        dfactor = cos_mask * (factor * np.real(dlogA) - absA * sintheta * dtheta)

        weight = walkers.weight * factor
        dweight = walkers.dweight * factor + walkers.weight * dfactor

        if self.apply_weight_bound:
            wbound = 0.1 * np.sum(weight)
            dwbound = 0.1 * np.sum(dweight)
            capped = ~(weight < wbound)
            weight = np.where(capped, wbound, weight)
            dweight = np.where(capped, dwbound, dweight)
        else:
            capped = np.zeros_like(weight, dtype=bool)

        if self.debug:
            self.diagnostics.append(
                {
                    "fb_cap_mask": fb_cap_mask.copy(),
                    "weight_cap_mask": capped.copy(),
                    "cos_sign_mask": cos_mask.copy(),
                    "min_abs_cos": float(np.min(np.abs(costheta))),
                    "max_abs_xbar_precap": float(
                        np.max(np.abs(-self.sqrtdt * (1j * vbias - self.mf_shift)))
                    ),
                    "max_abs_dphi": float(np.max(np.abs(dphi))),
                }
            )

        return GradWalkers(walkers.nwalkers, phi, dphi, weight, dweight)

    def propagate_block(
        self,
        iblock,
        walkers,
        ham,
        trial,
        stabilize_freq,
        pop_control_freq,
        fields,
        eshift_override=None,
        detach_eshift=True,
        sr_indices_queue=None,
        sr_record=None,
    ):
        """A sub-block of prop_block_size steps, with the exact adafqmc schedule.

        Returns (walkers, etot, detot, totw, dtotw) from the end-of-sub-block
        mixed estimator; energy_estimate is updated to the detached value
        (tangent zeroed), mirroring adafqmc propagate_block.

        eshift_override, if given, replaces the end-of-sub-block energy_estimate
        update value.  Because the update is detached (a constant of the
        differentiated function), finite-difference checks must freeze this
        sequence at its lambda = 0 values to evaluate the same function the
        tangents differentiate.

        detach_eshift=False instead differentiates through the energy-shift
        feedback (denergy_estimate = detot), used by the path-continuous mode
        where nothing is detached; its finite-difference counterpart then needs
        no eshift freezing at all.

        sr_indices_queue, if given, replays recorded reconfiguration index
        arrays (popping one per SR event) instead of recomputing them — the
        frozen-SR-map semantics of common-random-number verification.  The
        uniform is still consumed to keep the field stream aligned.  Every SR
        index array used is appended to sr_record when provided.
        """
        etot = detot = totw = dtotw = None
        for i in range(self.prop_block_size):
            step = iblock * self.prop_block_size + i
            if step % stabilize_freq == stabilize_freq - 1:
                walkers = reorthogonalize(walkers)
            x = fields.normal(walkers.nwalkers, ham.nchol)
            walkers = self.propagate_walkers(walkers, ham, trial, x)
            if i == self.prop_block_size - 1:
                Ghalf, dGhalf, _, _, _ = trial.get_ghalf_with_tangent(
                    walkers.phi, walkers.dphi
                )
                eloc, deloc = local_energy_with_tangent(
                    trial.rh1, trial.drh1, trial.rchol, trial.drchol, Ghalf, dGhalf, ham.enuc
                )
                etot, detot, totw, dtotw = weighted_energy_with_tangent(
                    walkers.weight, walkers.dweight, eloc, deloc
                )
                self.energy_estimate = etot if eshift_override is None else eshift_override
                self.denergy_estimate = 0.0 if detach_eshift else detot
            if step % pop_control_freq == pop_control_freq - 1:
                zeta = fields.uniform()
                replay = None if sr_indices_queue is None else sr_indices_queue.pop(0)
                walkers, sr_indices = stochastic_reconfiguration(walkers, zeta, indices=replay)
                if sr_record is not None:
                    sr_record.append(np.asarray(sr_indices).copy())
                if self.debug:
                    self.diagnostics.append({"sr_indices": np.asarray(sr_indices).copy()})
        return walkers, etot, detot, totw, dtotw
