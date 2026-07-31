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
"""Small random RHF test systems for the analytical-gradient addon tests."""

import numpy as np


def random_symmetric(rng, n, scale=1.0):
    a = rng.standard_normal((n, n))
    return scale * 0.5 * (a + a.T)


def gauge_fix_columns(mat):
    """Fix the sign of each column so its largest-magnitude entry is positive."""
    fixed = mat.copy()
    for j in range(mat.shape[1]):
        i = np.argmax(np.abs(fixed[:, j]))
        if fixed[i, j].real < 0:
            fixed[:, j] = -fixed[:, j]
    return fixed


def build_random_rhf_test_system(
    nao=4, nelec0=2, nchol=6, seed=7, chol_scale=0.3, obs_scale=0.1, enuc=0.7
):
    """Random symmetric h1e/chol/obs and an orthonormal trial (h1e eigenvectors).

    Returns (h1e, chol, enuc, obs_mat, psi) as float64 numpy arrays, with psi
    gauge-fixed so tests are deterministic across LAPACK builds.
    """
    rng = np.random.default_rng(seed)
    h1e = random_symmetric(rng, nao)
    chol = np.stack([random_symmetric(rng, nao, scale=chol_scale) for _ in range(nchol)])
    obs_mat = random_symmetric(rng, nao, scale=obs_scale)
    _, evecs = np.linalg.eigh(h1e)
    psi = gauge_fix_columns(evecs)
    return h1e, chol, float(enuc), obs_mat, psi
