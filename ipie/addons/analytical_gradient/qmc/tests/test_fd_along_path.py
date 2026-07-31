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
"""Path-continuous gradient vs finite differences, at every measurement.

The defining identity of the analytical gradient: run the perturbed and
unperturbed systems with the same auxiliary-field stream and the same SR
walker map (SR is not differentiated — its index arrays are recorded from the
base run and replayed in the shifted runs), with nothing detached along the
path (the energy-shift feedback is differentiated through).  Then the central
finite difference of every measured E_i and W_i equals the analytical dE_i and
dW_i, no matter how many measurement blocks are propagated.
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

NAO, NOCC, NCHOL, NW = 4, 2, 6, 24
DT = 0.005
NSTEPS_PER_MEAS = 10
NUM_MEASUREMENTS = 6  # 60 steps total
STABILIZE_FREQ = 5
POP_CONTROL_FREQ = 10  # 6 SR events along the path


def make_ham(mode, seed=401):
    """mode: 'fixed' (dh1e only), 'relaxed_ham' (rotated-basis: dchol != 0),
    'trial' (fixed basis: dchol = 0, trial tangent dpsi != 0)."""
    h1e, chol, enuc, obs_mat, psi = build_random_rhf_test_system(
        nao=NAO, nelec0=NOCC, nchol=NCHOL, seed=seed
    )
    rng = np.random.default_rng(seed + 5)
    if mode == "relaxed_ham":
        dchol = np.stack([random_symmetric(rng, NAO, scale=0.05) for _ in range(NCHOL)])
    else:
        dchol = np.zeros_like(chol)
    dpsi = 0.1 * rng.standard_normal(psi.shape) if mode == "trial" else np.zeros_like(psi)
    return HamTangent(NOCC, NAO, h1e, chol, enuc, dh1e=obs_mat, dchol=dchol), psi, dpsi


def run_path(ham, psi, dpsi, walkers0, content, sr_replay=None):
    """walkers0: common lambda-independent initial walkers (the analytic branch
    starts with zero tangents, so the shifted value runs must start from the
    SAME states, not from their own shifted trial)."""
    normals, uniforms = content
    fields = ScriptedFields([x.copy() for x in normals], list(uniforms))
    trial = SDTrial(psi, NOCC, dpsi=dpsi)
    trial.half_rot(ham)
    driver = FwdGradAFQMC.build(
        num_walkers=NW,
        num_steps_per_block=NSTEPS_PER_MEAS,
        ad_block_size=NSTEPS_PER_MEAS,
        num_ad_blocks=1,
        timestep=DT,
        stabilize_freq=STABILIZE_FREQ,
        pop_control_freq=POP_CONTROL_FREQ,
        seed=0,
        num_eqlb_steps=0,
        fields=fields,
        debug=True,
    )
    E, dE, W, dW, _ = driver.run_along_path(
        ham, trial, NUM_MEASUREMENTS, walkers=walkers0.detached_copy(), sr_replay=sr_replay
    )
    fields.assert_exhausted()
    return E, dE, W, dW, driver.sr_record, driver.last_diagnostics


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["fixed", "relaxed_ham", "trial"])
def test_along_path_gradient_vs_frozen_sr_fd(mode):
    ham, psi, dpsi = make_ham(mode)
    rng = np.random.default_rng(1337)
    nsteps = NUM_MEASUREMENTS * NSTEPS_PER_MEAS
    normals = [rng.standard_normal((NW, NCHOL)) for _ in range(nsteps)]
    uniforms = [rng.random() for _ in range(nsteps // POP_CONTROL_FREQ)]
    content = (normals, uniforms)

    trial_base = SDTrial(psi, NOCC)
    from ipie.addons.analytical_gradient.walkers.rhf_walkers import initialize_walkers

    walkers0 = initialize_walkers(trial_base, NW)

    E0, dE, W0, dW, sr_record, diag0 = run_path(ham, psi, dpsi, walkers0, content)
    steps0 = [d for d in diag0 if "fb_cap_mask" in d]

    fd = {}
    for eps in (1e-4, 5e-5):
        # Frozen SR map: replay the base run's walker map in the shifted runs;
        # the value branches shift the Hamiltonian arrays AND the trial, but
        # start from the same lambda-independent walkers.
        Ep, _, Wp, _, _, diag_p = run_path(
            shifted_copy(ham, eps), psi + eps * dpsi, None, walkers0, content, sr_record
        )
        Em, _, Wm, _, _, diag_m = run_path(
            shifted_copy(ham, -eps), psi - eps * dpsi, None, walkers0, content, sr_record
        )
        # Stencil validity: no cap/clip branch flips anywhere along the path.
        for diag in (diag_p, diag_m):
            steps = [d for d in diag if "fb_cap_mask" in d]
            assert len(steps) == len(steps0)
            for s, s0 in zip(steps, steps0):
                np.testing.assert_array_equal(s["fb_cap_mask"], s0["fb_cap_mask"])
                np.testing.assert_array_equal(s["weight_cap_mask"], s0["weight_cap_mask"])
                np.testing.assert_array_equal(s["cos_sign_mask"], s0["cos_sign_mask"])
        fd[eps] = ((Ep - Em) / (2 * eps), (Wp - Wm) / (2 * eps))

    dE_fd1, dW_fd1 = fd[1e-4]
    dE_fd2, dW_fd2 = fd[5e-5]
    richardson_e = (4 * dE_fd2 - dE_fd1) / 3
    richardson_w = (4 * dW_fd2 - dW_fd1) / 3
    # The identity holds at EVERY measurement along the path.
    for i in range(NUM_MEASUREMENTS):
        assert abs(dE_fd1[i] - dE[i]) < 1e-6 + 1e-5 * abs(dE[i]), (i, dE_fd1[i], dE[i])
        assert abs(richardson_e[i] - dE[i]) < 1e-7 + 1e-6 * abs(dE[i]), (
            i,
            richardson_e[i],
            dE[i],
        )
        assert abs(dW_fd1[i] - dW[i]) < 1e-6 + 1e-5 * abs(dW[i]), (i, dW_fd1[i], dW[i])
        assert abs(richardson_w[i] - dW[i]) < 1e-7 + 1e-6 * abs(dW[i]), (
            i,
            richardson_w[i],
            dW[i],
        )
