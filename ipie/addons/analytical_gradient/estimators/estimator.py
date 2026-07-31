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


def local_energy_with_tangent(rh1, drh1, rchol, drchol, Ghalf, dGhalf, enuc):
    """Per-walker local energy and its tangent.

    E_loc = enuc + 2 Tr(rh1 Theta) + 1/2 (E_J - E_X)  with
    E_J = sum_g X_g^2, X_g = 2 Tr(rchol_g Theta) (unconjugated square), and
    E_X = 2 sum_g Tr(rchol_g Theta rchol_g Theta).
    """
    e1 = 2.0 * np.einsum("ij,wji->w", rh1, Ghalf)
    de1 = 2.0 * (
        np.einsum("ij,wji->w", drh1, Ghalf) + np.einsum("ij,wji->w", rh1, dGhalf)
    )
    X = 2.0 * np.einsum("pij,wji->wp", rchol, Ghalf)
    dX = 2.0 * (
        np.einsum("pij,wji->wp", drchol, Ghalf) + np.einsum("pij,wji->wp", rchol, dGhalf)
    )
    ej = np.einsum("wp,wp->w", X, X)
    dej = 2.0 * np.einsum("wp,wp->w", X, dX)
    T = np.einsum("gip,wpj->wgij", rchol, Ghalf)
    dT = np.einsum("gip,wpj->wgij", drchol, Ghalf) + np.einsum(
        "gip,wpj->wgij", rchol, dGhalf
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
