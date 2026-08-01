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
"""RHF walkers carrying forward-mode tangent state (dphi, dweight)."""

import numpy as np


class GradWalkers:
    def __init__(self, nwalkers, phi, dphi, weight, dweight):
        self.nwalkers = nwalkers
        self.phi = phi  # (nwalkers, nao, nocc) complex128
        self.dphi = dphi  # tangent d(phi)/dlambda, same shape
        self.weight = weight  # (nwalkers,) float64
        self.dweight = dweight  # tangent d(weight)/dlambda, float64

    def detached_copy(self):
        """Value copy with tangents reset to zero (the AD-block-boundary detach)."""
        return GradWalkers(
            self.nwalkers,
            self.phi.copy(),
            np.zeros_like(self.phi),
            self.weight.copy(),
            np.zeros_like(self.weight),
        )


def initialize_walkers(trial, nwalkers):
    phi = np.array([trial.psi] * nwalkers, dtype=np.complex128)
    dphi = np.zeros_like(phi)
    weight = np.ones(nwalkers, dtype=np.float64)
    dweight = np.zeros(nwalkers, dtype=np.float64)
    return GradWalkers(nwalkers, phi, dphi, weight, dweight)


def reorthogonalize(walkers, project_gauge=True):
    """phi <- Q from phi = QR; tangent dphi <- (1 - QQ^dag) dphi R^{-1}.

    Treating R as lambda-independent is exact for the block estimator: the
    estimator is invariant under right-multiplication of any walker by a fixed
    invertible matrix and the propagation is covariant under it, so the
    algorithms with R(lambda), with frozen R(0), and with no
    reorthogonalization at all define the same function of lambda.  The
    tangent produced here differs from the QR Q-factor differential (what
    reverse-mode AD uses) by a pure gauge direction along which the estimator
    has zero derivative.

    project_gauge additionally removes the in-span component Q (Q^dag dphi)
    of the tangent — also a pure gauge change (tangents of the form phi.M are
    exactly null for the estimator, and the tangent flow maps gauge to gauge),
    so all computed gradients are unchanged in exact arithmetic.  It is
    essential for conditioning in many-electron systems: the raw tangent grows
    like exp(lambda_L tau) along a Lyapunov-unstable, almost purely in-span
    direction (measured lambda_L ~ 13/a.u. for NH3/STO-3G), and without the
    projection the estimator's exact cancellation of that growth is destroyed
    by floating-point roundoff (1e-16 * |dphi|).  Reverse-mode AD through the
    QR Q-factor performs the analogous re-gauging automatically, which is why
    adafqmc does not exhibit the instability.
    """
    Q, R = np.linalg.qr(walkers.phi)
    dphi = np.linalg.solve(
        np.transpose(R, (0, 2, 1)), np.transpose(walkers.dphi, (0, 2, 1))
    ).transpose(0, 2, 1)
    if project_gauge:
        dphi = dphi - Q @ (np.transpose(Q.conj(), (0, 2, 1)) @ dphi)
    return GradWalkers(walkers.nwalkers, Q, dphi, walkers.weight, walkers.dweight)


def stochastic_reconfiguration(walkers, zeta, indices=None):
    """Systematic resampling, mirroring adafqmc semantics exactly.

    The post-SR weights are identically the pre-normalized average as a
    function of lambda, so their tangent is exactly zero; state tangents flow
    through the (locally constant) gather.  Returns (walkers, indices).

    If indices is given, the resampling map is replayed instead of recomputed
    (SR as a fixed, undifferentiated gather) — used by common-random-number
    finite-difference verification so the perturbed runs follow the base
    run's walker map exactly.
    """
    nw = walkers.nwalkers
    weights = walkers.weight / np.sum(walkers.weight) * nw
    cumulative = np.cumsum(weights)
    total = cumulative[-1]
    average = total / nw
    if indices is None:
        z = total * (np.arange(nw) + zeta) / nw
        indices = np.searchsorted(cumulative, z, side="left")
        indices = np.where(indices < nw, indices, 0)
    phi = walkers.phi[indices].copy()
    dphi = walkers.dphi[indices].copy()
    weight = np.full(nw, average, dtype=np.float64)
    dweight = np.zeros(nw, dtype=np.float64)
    return GradWalkers(nw, phi, dphi, weight, dweight), indices
