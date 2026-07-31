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
"""Full AD-block gradient vs common-random-number central finite differences.

The block crosses reorthogonalization, stochastic reconfiguration, and the
weight cap; validity of the FD stencil is guaranteed by asserting that every
cap/clip mask and every SR index array is identical across the lambda runs.
"""

import numpy as np
import pytest

from ipie.addons.analytical_gradient.hamiltonians.hamiltonian import (
    HamTangent,
    shifted_copy,
)
from ipie.addons.analytical_gradient.qmc.fwdgrad_afqmc import FwdGradAFQMC
from ipie.addons.analytical_gradient.trial_wavefunction.sdtrial import SDTrial
from ipie.addons.analytical_gradient.utils.fields import ScriptedFields
from ipie.addons.analytical_gradient.utils.testing import (
    build_random_rhf_test_system,
    random_symmetric,
)
from ipie.addons.analytical_gradient.walkers.rhf_walkers import initialize_walkers

NAO, NOCC, NCHOL, NW = 4, 2, 6, 24
DT = 0.005
NSTEPS_PER_SUB = 10
AD_BLOCK_SIZE = 30  # 3 sub-blocks
STABILIZE_FREQ = 5
POP_CONTROL_FREQ = 10  # SR at steps 9, 19, 29


def make_ham(relaxed_like, seed=201):
    h1e, chol, enuc, obs_mat, psi = build_random_rhf_test_system(
        nao=NAO, nelec0=NOCC, nchol=NCHOL, seed=seed
    )
    if relaxed_like:
        rng = np.random.default_rng(seed + 5)
        dchol = np.stack([random_symmetric(rng, NAO, scale=0.05) for _ in range(NCHOL)])
    else:
        dchol = np.zeros_like(chol)
    ham = HamTangent(NOCC, NAO, h1e, chol, enuc, dh1e=obs_mat, dchol=dchol)
    return ham, psi


def run_block(ham, psi, fields_content, eshift_override=None):
    """One AD block on scripted fields.

    Returns (E, dE, W, dW, diagnostics, subblock_etots).  The finite-difference
    runs must freeze the detached energy-shift sequence at its lambda = 0
    values (eshift_override) so the stencil evaluates the same function the
    tangents differentiate.
    """
    normals, uniforms = fields_content
    fields = ScriptedFields(normals, uniforms)
    trial = SDTrial(psi, NOCC)
    trial.half_rot(ham)
    driver = FwdGradAFQMC.build(
        num_walkers=NW,
        num_steps_per_block=NSTEPS_PER_SUB,
        ad_block_size=AD_BLOCK_SIZE,
        num_ad_blocks=1,
        timestep=DT,
        stabilize_freq=STABILIZE_FREQ,
        pop_control_freq=POP_CONTROL_FREQ,
        seed=0,
        num_eqlb_steps=0,
        fields=fields,
        debug=True,
    )
    walkers = initialize_walkers(trial, NW)
    E, dE, W, dW, _ = driver.ad_block(ham, trial, walkers, eshift_override=eshift_override)
    fields.assert_exhausted()
    return E, dE, W, dW, driver.last_diagnostics, driver.last_subblock_etots


def extract_masks(diagnostics):
    steps = [d for d in diagnostics if "fb_cap_mask" in d]
    srs = [d for d in diagnostics if "sr_indices" in d]
    return steps, srs


@pytest.mark.unit
@pytest.mark.parametrize("relaxed_like", [False, True])
def test_block_gradient_vs_crn_fd(relaxed_like):
    ham, psi = make_ham(relaxed_like)
    rng = np.random.default_rng(999)
    normals = [rng.standard_normal((NW, NCHOL)) for _ in range(AD_BLOCK_SIZE)]
    uniforms = [rng.random() for _ in range(AD_BLOCK_SIZE // POP_CONTROL_FREQ)]
    content = (normals, uniforms)

    E0, dE, W0, dW, diag0, eshifts0 = run_block(ham, psi, content)

    fd = {}
    diags = {}
    for eps in (1e-4, 5e-5):
        Ep, _, Wp, _, diag_p, _ = run_block(
            shifted_copy(ham, eps), psi, content, eshift_override=eshifts0
        )
        Em, _, Wm, _, diag_m, _ = run_block(
            shifted_copy(ham, -eps), psi, content, eshift_override=eshifts0
        )
        fd[eps] = ((Ep - Em) / (2 * eps), (Wp - Wm) / (2 * eps))
        diags[eps] = (diag_p, diag_m)

    # Stencil validity: every branch decision identical across all lambda runs.
    steps0, srs0 = extract_masks(diag0)
    assert min(d["min_abs_cos"] for d in steps0) > 0.05
    for eps in fd:
        for diag in diags[eps]:
            steps, srs = extract_masks(diag)
            assert len(steps) == len(steps0) and len(srs) == len(srs0)
            for s, s0 in zip(steps, steps0):
                np.testing.assert_array_equal(s["fb_cap_mask"], s0["fb_cap_mask"])
                np.testing.assert_array_equal(s["weight_cap_mask"], s0["weight_cap_mask"])
                np.testing.assert_array_equal(s["cos_sign_mask"], s0["cos_sign_mask"])
            for s, s0 in zip(srs, srs0):
                np.testing.assert_array_equal(s["sr_indices"], s0["sr_indices"])

    d1, w1 = fd[1e-4]
    d2, w2 = fd[5e-5]
    richardson_e = (4 * d2 - d1) / 3
    richardson_w = (4 * w2 - w1) / 3
    assert abs(d1 - dE) < 1e-6 + 1e-5 * abs(dE), (d1, dE)
    assert abs(richardson_e - dE) < 1e-7 + 1e-6 * abs(dE), (richardson_e, dE)
    assert abs(w1 - dW) < 1e-6 + 1e-5 * abs(dW), (w1, dW)
    assert abs(richardson_w - dW) < 1e-7 + 1e-6 * abs(dW), (richardson_w, dW)
