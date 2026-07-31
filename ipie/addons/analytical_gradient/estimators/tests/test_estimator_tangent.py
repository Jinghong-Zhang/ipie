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

import numpy as np
import pytest

from ipie.addons.analytical_gradient.estimators.estimator import (
    block_average_with_tangent,
    local_energy_with_tangent,
    weighted_energy_with_tangent,
)
from ipie.addons.analytical_gradient.hamiltonians.hamiltonian import HamTangent
from ipie.addons.analytical_gradient.trial_wavefunction.sdtrial import SDTrial
from ipie.addons.analytical_gradient.utils.testing import (
    build_random_rhf_test_system,
    random_symmetric,
)

NAO, NOCC, NCHOL, NW = 4, 2, 6, 3


def setup_system(seed=17):
    h1e, chol, enuc, obs_mat, psi = build_random_rhf_test_system(
        nao=NAO, nelec0=NOCC, nchol=NCHOL, seed=seed
    )
    rng = np.random.default_rng(seed + 1)
    dchol = np.stack(
        [random_symmetric(rng, NAO, scale=0.05) for _ in range(NCHOL)]
    )
    ham = HamTangent(NOCC, NAO, h1e, chol, enuc, dh1e=obs_mat, dchol=dchol)
    phi = rng.standard_normal((NW, NAO, NOCC)) + 1j * rng.standard_normal((NW, NAO, NOCC))
    dphi = 0.3 * (
        rng.standard_normal((NW, NAO, NOCC)) + 1j * rng.standard_normal((NW, NAO, NOCC))
    )
    return ham, psi, phi, dphi


def eloc_at(ham, psi, phi, t):
    """Value branch of the local energy with arrays and states shifted by t."""
    ham_t = HamTangent(
        NOCC, NAO, ham.h1e + t * ham.dh1e, ham.chol + t * ham.dchol, ham.enuc
    )
    trial_t = SDTrial(psi, NOCC)
    trial_t.half_rot(ham_t)
    return trial_t, ham_t


@pytest.mark.unit
def test_local_energy_tangent_vs_fd():
    ham, psi, phi, dphi = setup_system()
    trial = SDTrial(psi, NOCC)
    trial.half_rot(ham)
    Ghalf, dGhalf, _, _, _ = trial.get_ghalf_with_tangent(phi, dphi)
    _, deloc = local_energy_with_tangent(
        trial.rh1, trial.drh1, trial.rchol, trial.drchol, Ghalf, dGhalf, ham.enuc
    )

    def value(t):
        trial_t, ham_t = eloc_at(ham, psi, phi, t)
        phi_t = phi + t * dphi
        Gh, _, _, _, _ = trial_t.get_ghalf_with_tangent(phi_t, np.zeros_like(phi_t))
        el, _ = local_energy_with_tangent(
            trial_t.rh1,
            trial_t.drh1,
            trial_t.rchol,
            trial_t.drchol,
            Gh,
            np.zeros_like(Gh),
            ham_t.enuc,
        )
        return el

    eps = 1e-6
    fd = (value(eps) - value(-eps)) / (2 * eps)
    np.testing.assert_allclose(fd, deloc, rtol=1e-6, atol=1e-8)


@pytest.mark.unit
def test_force_bias_tangent_vs_fd():
    ham, psi, phi, dphi = setup_system(seed=23)
    trial = SDTrial(psi, NOCC)
    trial.half_rot(ham)
    Ghalf, dGhalf, _, _, _ = trial.get_ghalf_with_tangent(phi, dphi)
    _, dvbias = trial.calc_force_bias_with_tangent(Ghalf, dGhalf)

    def value(t):
        trial_t, _ = eloc_at(ham, psi, phi, t)
        phi_t = phi + t * dphi
        Gh, _, _, _, _ = trial_t.get_ghalf_with_tangent(phi_t, np.zeros_like(phi_t))
        vb, _ = trial_t.calc_force_bias_with_tangent(Gh, np.zeros_like(Gh))
        return vb

    eps = 1e-6
    fd = (value(eps) - value(-eps)) / (2 * eps)
    np.testing.assert_allclose(fd, dvbias, rtol=1e-6, atol=1e-9)


@pytest.mark.unit
def test_trial_energy_tangent_vs_fd():
    ham, psi, _, _ = setup_system(seed=29)

    trial = SDTrial(psi, NOCC)
    trial.half_rot(ham)
    _, de = trial.eval_energy_with_tangent(ham)

    def value(t):
        trial_t, ham_t = eloc_at(ham, psi, None, t)
        e, _ = trial_t.eval_energy_with_tangent(ham_t)
        return e

    eps = 1e-6
    fd = (value(eps) - value(-eps)) / (2 * eps)
    np.testing.assert_allclose(fd, de, rtol=1e-6, atol=1e-9)


@pytest.mark.unit
def test_weighted_energy_tangent_quotient_rule():
    rng = np.random.default_rng(31)
    w = rng.uniform(0.5, 2.0, NW)
    dw = rng.standard_normal(NW)
    eloc = rng.standard_normal(NW) + 1j * rng.standard_normal(NW)
    deloc = rng.standard_normal(NW) + 1j * rng.standard_normal(NW)
    etot, detot, totw, dtotw = weighted_energy_with_tangent(w, dw, eloc, deloc)

    def value(t):
        wt = w + t * dw
        et = eloc + t * deloc
        return np.real(np.sum(wt * et) / np.sum(wt)), np.sum(wt)

    eps = 1e-7
    fd_e = (value(eps)[0] - value(-eps)[0]) / (2 * eps)
    fd_w = (value(eps)[1] - value(-eps)[1]) / (2 * eps)
    assert abs(etot - value(0.0)[0]) < 1e-12
    assert abs(totw - np.sum(w)) < 1e-12
    np.testing.assert_allclose(fd_e, detot, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(fd_w, dtotw, rtol=1e-6, atol=1e-9)


@pytest.mark.unit
def test_block_average_tangent_quotient_rule():
    rng = np.random.default_rng(37)
    n = 4
    e = rng.standard_normal(n)
    de = rng.standard_normal(n)
    wts = rng.uniform(0.5, 2.0, n)
    dwts = rng.standard_normal(n)
    E, dE = block_average_with_tangent(e, de, wts, dwts)

    def value(t):
        et = e + t * de
        wt = wts + t * dwts
        return np.sum(et * wt) / np.sum(wt)

    eps = 1e-7
    fd = (value(eps) - value(-eps)) / (2 * eps)
    assert abs(E - value(0.0)) < 1e-12
    np.testing.assert_allclose(fd, dE, rtol=1e-6, atol=1e-9)
