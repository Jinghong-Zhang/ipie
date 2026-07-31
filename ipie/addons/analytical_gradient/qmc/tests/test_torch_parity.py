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
"""Exact-fields parity: analytical forward-mode gradient vs adafqmc torch AD.

Both codes are driven by an identical auxiliary-field stream (torch.randn and
torch.rand are monkeypatched to replay a recorded numpy stream, with shape
assertions so any draw-order mismatch fails loudly).  Per-block energy,
gradient (observable), total weight, and weight gradient must then agree to
floating-point accumulation accuracy.  This is the analytical correctness
check of the AD implementation, fixed and relaxed trial.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

# adafqmc imports must come after the importorskip (they import torch).
from ipie.addons.adafqmc.hamiltonians.hamiltonian import HamObs  # noqa: E402
from ipie.addons.adafqmc.qmc.adafqmc import ADAFQMC  # noqa: E402
from ipie.addons.adafqmc.utils.miscellaneous import trial_tangent  # noqa: E402
from ipie.addons.adafqmc.walkers.rhf_walkers import Walkers  # noqa: E402
from ipie.qmc.comm import FakeComm  # noqa: E402

from ipie.addons.analytical_gradient.hamiltonians.hamiltonian import (  # noqa: E402
    build_relaxed_trial_tangent,
)
from ipie.addons.analytical_gradient.qmc.fwdgrad_afqmc import FwdGradAFQMC  # noqa: E402
from ipie.addons.analytical_gradient.trial_wavefunction.sdtrial import SDTrial  # noqa: E402
from ipie.addons.analytical_gradient.utils.fields import ScriptedFields  # noqa: E402
from ipie.addons.analytical_gradient.utils.testing import (  # noqa: E402
    build_random_rhf_test_system,
)
from ipie.addons.analytical_gradient.walkers.rhf_walkers import GradWalkers  # noqa: E402

NAO, NOCC, NCHOL, NW = 4, 2, 6, 24
DT = 0.005
NSTEPS_PER_SUB = 10
AD_BLOCK_SIZE = 20  # 2 sub-blocks: exercises the eshift detach path
STABILIZE_FREQ = 5
POP_CONTROL_FREQ = 5  # SR events at steps 4, 9, 14, 19
OBS_CONST = 0.31


def make_inputs(relaxed, seed=301):
    h1e, chol, enuc, obs_mat, psi = build_random_rhf_test_system(
        nao=NAO, nelec0=NOCC, nchol=NCHOL, seed=seed
    )
    rng = np.random.default_rng(seed + 3)
    U0 = psi  # orthonormal rotation (relaxed-orbital coefficients at lambda=0)
    dU = 0.1 * rng.standard_normal((NAO, NAO)) if relaxed else np.zeros((NAO, NAO))
    # Initial walkers in the rotated (identity-trial) basis: perturbed identity
    # columns with unequal weights, so QR/SR/weight-cap paths are all nontrivial.
    phi0 = np.array([np.eye(NAO)[:, :NOCC]] * NW, dtype=np.complex128)
    phi0 += 0.05 * (
        rng.standard_normal(phi0.shape) + 1j * rng.standard_normal(phi0.shape)
    )
    w0 = rng.uniform(0.5, 2.0, NW)
    nsteps = AD_BLOCK_SIZE
    normals = [rng.standard_normal((NW, NCHOL)) for _ in range(nsteps)]
    uniforms = [rng.random() for _ in range(nsteps // POP_CONTROL_FREQ)]
    return h1e, chol, enuc, obs_mat, U0, dU, phi0, w0, (normals, uniforms)


def run_numpy(h1e, chol, enuc, obs_mat, U0, dU, phi0, w0, content):
    ham = build_relaxed_trial_tangent(NOCC, NAO, h1e, chol, enuc, obs_mat, U0, dU)
    trial = SDTrial(np.eye(NAO), NOCC)
    trial.half_rot(ham)
    fields = ScriptedFields(*content)
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
    )
    walkers = GradWalkers(NW, phi0.copy(), np.zeros_like(phi0), w0.copy(), np.zeros(NW))
    E, dE, W, dW, _ = driver.ad_block(ham, trial, walkers)
    fields.assert_exhausted()
    return E, dE + OBS_CONST, W, dW


def run_torch(h1e, chol, enuc, obs_mat, U0, dU, phi0, w0, content, monkeypatch):
    normals = [x.copy() for x in content[0]]
    uniforms = list(content[1])

    def fake_randn(*shape, dtype=None):
        assert normals, "torch.randn called more times than the recorded stream"
        x = normals.pop(0)
        assert x.shape == tuple(shape), (x.shape, shape)
        return torch.from_numpy(x.copy())

    def fake_rand(n):
        assert n == 1
        assert uniforms, "torch.rand called more times than the recorded stream"
        return torch.tensor([uniforms.pop(0)], dtype=torch.float64)

    monkeypatch.setattr(torch, "randn", fake_randn)
    monkeypatch.setattr(torch, "rand", fake_rand)

    hamobs = HamObs(
        NOCC,
        NAO,
        torch.tensor(h1e, dtype=torch.float64),
        torch.tensor(chol, dtype=torch.float64),
        torch.tensor([enuc], dtype=torch.float64),
        observable=(
            torch.tensor(obs_mat, dtype=torch.float64),
            torch.tensor([OBS_CONST], dtype=torch.float64),
        ),
        obs_type="dipole",
    )
    adafqmc = ADAFQMC.build(
        FakeComm(),
        trial_tangent,
        num_walkers_per_process=NW,
        num_steps_per_block=NSTEPS_PER_SUB,
        ad_block_size=AD_BLOCK_SIZE,
        num_ad_blocks=1,
        timestep=DT,
        stabilize_freq=STABILIZE_FREQ,
        pop_control_freq=POP_CONTROL_FREQ,
    )
    walkers = Walkers(
        NW,
        torch.tensor(phi0, dtype=torch.complex128),
        torch.tensor(w0, dtype=torch.float64),
    )
    trial_detached = torch.tensor(U0, dtype=torch.float64)
    tangent = torch.tensor(dU, dtype=torch.float64)
    _, e_estimate, observable, blkwts, wtsgrad = adafqmc.ad_block_gradient(
        hamobs, walkers, trial_detached, tangent
    )
    assert not normals and not uniforms, "recorded stream not fully consumed"
    return (
        float(e_estimate),
        float(observable),
        float(blkwts),
        float(wtsgrad),
    )


@pytest.mark.unit
@pytest.mark.parametrize("relaxed", [False, True])
def test_ad_block_parity_vs_adafqmc(relaxed, monkeypatch):
    inputs = make_inputs(relaxed)
    E_np, obs_np, W_np, dW_np = run_numpy(*inputs)
    E_t, obs_t, W_t, dW_t = run_torch(*inputs, monkeypatch)

    # Values: pure transcription check of the propagation.
    assert abs(E_np - E_t) < 1e-10, (E_np, E_t)
    assert abs(W_np - W_t) < 1e-9 * abs(W_t), (W_np, W_t)
    # Gradients: forward-mode tangents vs reverse-mode AD (roundoff accumulates
    # differently; QR gauge equivalence exact only in exact arithmetic).
    assert abs(obs_np - obs_t) < 1e-8 + 1e-6 * abs(obs_t), (obs_np, obs_t)
    assert abs(dW_np - dW_t) < 1e-8 + 1e-6 * abs(dW_t), (dW_np, dW_t)
