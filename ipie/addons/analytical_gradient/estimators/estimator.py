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
"""RHF local energy and weighted averages, with forward-mode tangents."""

import numpy as np

from ipie.addons.analytical_gradient.utils.linalg import flatten_ghalf, rmatmul


def local_energy_with_tangent(rh1, drh1, rchol, drchol, Ghalf, dGhalf, enuc):
    """Per-walker local energy and its tangent.

    E_loc = enuc + 2 Tr(rh1 Theta) + 1/2 (E_J - E_X)  with
    E_J = sum_g X_g^2, X_g = 2 Tr(rchol_g Theta) (unconjugated square), and
    E_X = 2 sum_g Tr(rchol_g Theta rchol_g Theta).  One-body and Coulomb
    contractions are flattened to gemms (core-ipie local-energy convention).
    """
    nw, nao, nocc = Ghalf.shape
    nchol = rchol.shape[0]
    Gt = flatten_ghalf(Ghalf)
    dGt = flatten_ghalf(dGhalf)
    rh1_flat = rh1.reshape(-1)
    drh1_flat = drh1.reshape(-1)
    rchol_flat = rchol.reshape(nchol, -1)
    drchol_flat = drchol.reshape(nchol, -1)
    has_dr = bool(drchol_flat.any())

    e1 = 2.0 * (Gt @ rh1_flat)
    de1 = 2.0 * (Gt @ drh1_flat + dGt @ rh1_flat)
    X = 2.0 * rmatmul(Gt, rchol_flat.T)
    dX = 2.0 * rmatmul(dGt, rchol_flat.T)
    if has_dr:
        dX = dX + 2.0 * rmatmul(Gt, drchol_flat.T)
    ej = np.einsum("wp,wp->w", X, X)
    dej = 2.0 * np.einsum("wp,wp->w", X, dX)

    rcholm = rchol.reshape(nchol * nocc, nao)
    T = np.matmul(rcholm, Ghalf).reshape(nw, nchol, nocc, nocc)
    dT = np.matmul(rcholm, dGhalf).reshape(nw, nchol, nocc, nocc)
    if has_dr:
        dT = dT + np.matmul(drchol.reshape(nchol * nocc, nao), Ghalf).reshape(
            nw, nchol, nocc, nocc
        )
    ex = 2.0 * np.einsum("wgij,wgji->w", T, T)
    dex = 2.0 * (
        np.einsum("wgij,wgji->w", dT, T) + np.einsum("wgij,wgji->w", T, dT)
    )
    eloc = enuc + e1 + 0.5 * (ej - ex)
    deloc = de1 + 0.5 * (dej - dex)
    return eloc, deloc


def weighted_energy_with_tangent(weight, dweight, eloc, deloc):
    """Mixed estimator E = Re[sum_w w E_loc / sum_w w] and its tangent.

    Weights are real; the complex weighted mean is differentiated by the
    quotient rule before taking the real part.
    Returns (etot, detot, totw, dtotw).
    """
    totw = np.sum(weight)
    dtotw = np.sum(dweight)
    ebar = np.sum(weight * eloc) / totw
    debar = np.sum(dweight * (eloc - ebar) + weight * deloc) / totw
    return np.real(ebar), np.real(debar), totw, dtotw


def block_average_with_tangent(etots, detots, wts, dwts):
    """AD-block combination E = sum_i E_i W_i / sum_i W_i and its tangent."""
    etots = np.asarray(etots)
    detots = np.asarray(detots)
    wts = np.asarray(wts)
    dwts = np.asarray(dwts)
    W = np.sum(wts)
    E = np.sum(etots * wts) / W
    dE = np.sum(detots * wts + (etots - E) * dwts) / W
    return E, dE
