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
    build_fixed_trial_tangent,
)
from ipie.addons.analytical_gradient.qmc.fwdgrad_afqmc import FwdGradAFQMC
from ipie.addons.analytical_gradient.trial_wavefunction.sdtrial import SDTrial
from ipie.addons.analytical_gradient.utils.testing import build_random_rhf_test_system


@pytest.mark.unit
def test_fwdgrad_afqmc_smoke():
    nao, nocc, nchol = 4, 2, 6
    h1e, chol, enuc, obs_mat, psi = build_random_rhf_test_system(
        nao=nao, nelec0=nocc, nchol=nchol, seed=101
    )
    ham = build_fixed_trial_tangent(nocc, nao, h1e, chol, enuc, obs_mat)
    trial = SDTrial(psi, nocc)
    trial.half_rot(ham)
    driver = FwdGradAFQMC.build(
        num_walkers=24,
        num_steps_per_block=5,
        ad_block_size=10,
        num_ad_blocks=2,
        timestep=0.005,
        stabilize_freq=5,
        pop_control_freq=5,
        pop_control_freq_eq=5,
        seed=42,
        num_eqlb_steps=10,
    )
    energies, gradients, weights, wtsgrads = driver.run(ham, trial, obs_const=0.5)
    for arr in (energies, gradients, weights, wtsgrads):
        assert arr.shape == (2,)
        assert arr.dtype == np.float64
        assert np.all(np.isfinite(arr))
    assert np.all(weights > 0)
