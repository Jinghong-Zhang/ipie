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

from ipie.addons.analytical_gradient.hamiltonians.hamiltonian import (
    HamTangent,
    build_fixed_trial_tangent,
    build_relaxed_trial_tangent,
    shifted_copy,
)
from ipie.addons.analytical_gradient.utils.testing import (
    build_random_rhf_test_system,
    random_symmetric,
)


@pytest.mark.unit
def test_h1e_mod_tangent_fd():
    h1e, chol, enuc, obs_mat, _ = build_random_rhf_test_system(seed=7)
    rng = np.random.default_rng(11)
    dchol = np.stack([random_symmetric(rng, 4, scale=0.05) for _ in range(chol.shape[0])])
    ham = HamTangent(2, 4, h1e, chol, enuc, dh1e=obs_mat, dchol=dchol)
    eps = 1e-6
    hp = HamTangent(2, 4, h1e + eps * obs_mat, chol + eps * dchol, enuc)
    hm = HamTangent(2, 4, h1e - eps * obs_mat, chol - eps * dchol, enuc)
    fd = (hp.h1e_mod - hm.h1e_mod) / (2 * eps)
    np.testing.assert_allclose(fd, ham.dh1e_mod, rtol=1e-6, atol=1e-9)


@pytest.mark.unit
def test_fixed_trial_builder():
    h1e, chol, enuc, obs_mat, _ = build_random_rhf_test_system(seed=7)
    ham = build_fixed_trial_tangent(2, 4, h1e, chol, enuc, obs_mat)
    np.testing.assert_allclose(ham.dh1e, obs_mat)
    np.testing.assert_allclose(ham.dchol, 0.0)
    np.testing.assert_allclose(ham.dh1e_mod, obs_mat)


@pytest.mark.unit
def test_relaxed_trial_builder_vs_fd():
    h1e, chol, enuc, obs_mat, _ = build_random_rhf_test_system(seed=13)
    rng = np.random.default_rng(3)
    U, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    dU = 0.1 * rng.standard_normal((4, 4))
    ham = build_relaxed_trial_tangent(2, 4, h1e, chol, enuc, obs_mat, U, dU)

    def arrays(eps):
        Ue = U + eps * dU
        he = Ue.conj().T @ (h1e + eps * obs_mat) @ Ue
        Le = np.einsum("qi,aij,jp->aqp", Ue.conj().T, chol, Ue)
        return he, Le

    eps = 1e-6
    hp, Lp = arrays(eps)
    hm, Lm = arrays(-eps)
    np.testing.assert_allclose(ham.h1e, arrays(0.0)[0], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(ham.chol, arrays(0.0)[1], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose((hp - hm) / (2 * eps), ham.dh1e, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose((Lp - Lm) / (2 * eps), ham.dchol, rtol=1e-6, atol=1e-9)


@pytest.mark.unit
def test_shifted_copy():
    h1e, chol, enuc, obs_mat, _ = build_random_rhf_test_system(seed=7)
    rng = np.random.default_rng(11)
    dchol = np.stack([random_symmetric(rng, 4, scale=0.05) for _ in range(chol.shape[0])])
    ham = HamTangent(2, 4, h1e, chol, enuc, dh1e=obs_mat, dchol=dchol)
    eps = 1e-3
    shifted = shifted_copy(ham, eps)
    np.testing.assert_allclose(shifted.h1e, h1e + eps * obs_mat)
    np.testing.assert_allclose(shifted.chol, chol + eps * dchol)
    np.testing.assert_allclose(shifted.dh1e, 0.0)
    np.testing.assert_allclose(shifted.dchol, 0.0)
    assert shifted.enuc == ham.enuc
