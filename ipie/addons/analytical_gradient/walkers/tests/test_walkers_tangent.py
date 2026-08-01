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

from ipie.addons.analytical_gradient.estimators.estimator import local_energy_with_tangent
from ipie.addons.analytical_gradient.hamiltonians.hamiltonian import HamTangent
from ipie.addons.analytical_gradient.trial_wavefunction.sdtrial import SDTrial
from ipie.addons.analytical_gradient.utils.testing import (
    build_random_rhf_test_system,
    random_symmetric,
)
from ipie.addons.analytical_gradient.walkers.rhf_walkers import (
    GradWalkers,
    initialize_walkers,
    reorthogonalize,
    stochastic_reconfiguration,
)

NAO, NOCC, NCHOL, NW = 4, 2, 6, 5


def make_walkers_and_trial(seed=41):
    h1e, chol, enuc, obs_mat, psi = build_random_rhf_test_system(
        nao=NAO, nelec0=NOCC, nchol=NCHOL, seed=seed
    )
    rng = np.random.default_rng(seed + 1)
    dchol = np.stack([random_symmetric(rng, NAO, scale=0.05) for _ in range(NCHOL)])
    ham = HamTangent(NOCC, NAO, h1e, chol, enuc, dh1e=obs_mat, dchol=dchol)
    trial = SDTrial(psi, NOCC)
    trial.half_rot(ham)
    phi = np.array([psi[:, :NOCC]] * NW, dtype=np.complex128)
    phi += 0.2 * (
        rng.standard_normal(phi.shape) + 1j * rng.standard_normal(phi.shape)
    )
    dphi = 0.3 * (
        rng.standard_normal(phi.shape) + 1j * rng.standard_normal(phi.shape)
    )
    weight = rng.uniform(0.5, 2.0, NW)
    dweight = rng.standard_normal(NW)
    return ham, trial, GradWalkers(NW, phi, dphi, weight, dweight)


def gauge_invariants(ham, trial, walkers):
    Ghalf, dGhalf, S, dS, Sinv = trial.get_ghalf_with_tangent(walkers.phi, walkers.dphi)
    vbias, dvbias = trial.calc_force_bias_with_tangent(Ghalf, dGhalf)
    eloc, deloc = local_energy_with_tangent(
        trial.rh1, trial.drh1, trial.rchol, trial.drchol, Ghalf, dGhalf, ham.enuc
    )
    dlogdet = np.einsum("wii->w", np.linalg.solve(S, dS))
    return vbias, dvbias, eloc, deloc, dlogdet


@pytest.mark.unit
def test_reortho_gauge_invariance():
    """The linchpin of the reortho gauge: every quantity entering the estimator
    (force bias, local energy) is invariant under reortho, with or without the
    in-span gauge projection; the log-det tangent alone shifts by exactly
    -tr(M) under the projection (M = Q^dag dphi R^{-1}), which cancels in the
    step overlap-ratio tangent."""
    ham, trial, walkers = make_walkers_and_trial()
    before = gauge_invariants(ham, trial, walkers)
    after_fixed_r = gauge_invariants(ham, trial, reorthogonalize(walkers, project_gauge=False))
    for b, a in zip(before, after_fixed_r):
        np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-11)

    projected = reorthogonalize(walkers, project_gauge=True)
    after_proj = gauge_invariants(ham, trial, projected)
    for b, a in zip(before[:4], after_proj[:4]):  # vbias, dvbias, eloc, deloc
        np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-11)
    # dlogdet shifts by exactly -tr(M): reconstruct M from the two tangents.
    unprojected = reorthogonalize(walkers, project_gauge=False)
    Q = projected.phi
    M = np.einsum("wji,wjk->wik", Q.conj(), unprojected.dphi)
    np.testing.assert_allclose(
        after_proj[4], after_fixed_r[4] - np.einsum("wii->w", M), rtol=1e-10, atol=1e-11
    )
    # And the projected tangent has no in-span component.
    np.testing.assert_allclose(
        np.einsum("wji,wjk->wik", Q.conj(), projected.dphi), 0.0, atol=1e-12
    )


@pytest.mark.unit
def test_reortho_weights_untouched_and_phi_orthonormal():
    _, _, walkers = make_walkers_and_trial(seed=43)
    ortho = reorthogonalize(walkers)
    np.testing.assert_array_equal(ortho.weight, walkers.weight)
    np.testing.assert_array_equal(ortho.dweight, walkers.dweight)
    eye = np.einsum("wij,wik->wjk", ortho.phi.conj(), ortho.phi)
    np.testing.assert_allclose(eye, np.array([np.eye(NOCC)] * NW), atol=1e-12)


@pytest.mark.unit
def test_stochastic_reconfiguration_semantics():
    _, _, walkers = make_walkers_and_trial(seed=47)
    zeta = 0.37
    new, indices = stochastic_reconfiguration(walkers, zeta)

    # Reference transcription of the torch loop (rhf_walkers.py lines 29-46).
    nw = walkers.nwalkers
    weights = walkers.weight / np.sum(walkers.weight) * nw
    cumulative = np.cumsum(weights)
    total = cumulative[-1]
    ref_idx = []
    for i in range(nw):
        z = total * (i + zeta) / nw
        idx = int(np.searchsorted(cumulative, z, side="left"))
        ref_idx.append(idx if idx < nw else 0)
    np.testing.assert_array_equal(indices, ref_idx)
    np.testing.assert_allclose(new.weight, np.full(nw, total / nw))
    np.testing.assert_array_equal(new.dweight, np.zeros(nw))
    np.testing.assert_array_equal(new.phi, walkers.phi[ref_idx])
    np.testing.assert_array_equal(new.dphi, walkers.dphi[ref_idx])


@pytest.mark.unit
def test_initialize_walkers():
    _, trial, _ = make_walkers_and_trial(seed=53)
    walkers = initialize_walkers(trial, 3)
    assert walkers.phi.dtype == np.complex128
    np.testing.assert_allclose(walkers.phi[1], trial.psi)
    np.testing.assert_array_equal(walkers.weight, np.ones(3))
    np.testing.assert_array_equal(walkers.dphi, np.zeros_like(walkers.phi))
