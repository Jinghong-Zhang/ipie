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
import scipy.linalg

from ipie.addons.analytical_gradient.hamiltonians.hamiltonian import (
    HamTangent,
    shifted_copy,
)
from ipie.addons.analytical_gradient.propagation.propagator import (
    GradPropagator,
    apply_bound_force_bias_with_tangent,
    apply_taylor_with_tangent,
    compute_exph1_with_tangent,
    construct_vhs_with_tangent,
)
from ipie.addons.analytical_gradient.trial_wavefunction.sdtrial import SDTrial
from ipie.addons.analytical_gradient.utils.testing import (
    build_random_rhf_test_system,
    random_symmetric,
)
from ipie.addons.analytical_gradient.walkers.rhf_walkers import GradWalkers

NAO, NOCC, NCHOL, NW = 4, 2, 6, 5
DT = 0.005


def make_ham_and_trial(seed=61):
    h1e, chol, enuc, obs_mat, psi = build_random_rhf_test_system(
        nao=NAO, nelec0=NOCC, nchol=NCHOL, seed=seed
    )
    rng = np.random.default_rng(seed + 1)
    dchol = np.stack([random_symmetric(rng, NAO, scale=0.05) for _ in range(NCHOL)])
    ham = HamTangent(NOCC, NAO, h1e, chol, enuc, dh1e=obs_mat, dchol=dchol)
    trial = SDTrial(psi, NOCC)
    trial.half_rot(ham)
    return ham, trial, rng


@pytest.mark.unit
def test_exph1_tangent_vs_fd():
    ham, trial, _ = make_ham_and_trial()

    def exph1_at(eps):
        ham_e = shifted_copy(ham, eps)
        mf = 2j * np.einsum("pij,ij->p", ham_e.chol, trial.G)
        M = ham_e.h1e_mod + np.einsum("p,pij->ij", mf.imag, ham_e.chol)
        return scipy.linalg.expm(-0.5 * DT * M)

    mf0 = 2j * np.einsum("pij,ij->p", ham.chol, trial.G)
    dmf0 = 2j * np.einsum("pij,ij->p", ham.dchol, trial.G)
    expH1, dexpH1 = compute_exph1_with_tangent(
        ham.h1e_mod, ham.dh1e_mod, ham.chol, ham.dchol, mf0, dmf0, DT
    )
    np.testing.assert_allclose(expH1, exph1_at(0.0), rtol=1e-12, atol=1e-13)
    eps = 1e-6
    fd = (exph1_at(eps) - exph1_at(-eps)) / (2 * eps)
    np.testing.assert_allclose(fd, dexpH1, rtol=1e-6, atol=1e-9)


@pytest.mark.unit
def test_force_bias_cap_tangent_vs_fd():
    rng = np.random.default_rng(67)
    moduli = np.array([[0.3, 0.7, 1.5, 3.0, 0.9, 1.2]] * 2)
    phases = rng.uniform(0, 2 * np.pi, moduli.shape)
    xbar = moduli * np.exp(1j * phases)
    dxbar = 0.3 * (rng.standard_normal(xbar.shape) + 1j * rng.standard_normal(xbar.shape))

    _, dxbar_out, mask = apply_bound_force_bias_with_tangent(xbar, dxbar, max_bound=1.0)
    np.testing.assert_array_equal(mask, moduli > 1.0)

    eps = 1e-6
    yp, _, mp = apply_bound_force_bias_with_tangent(
        xbar + eps * dxbar, np.zeros_like(dxbar), max_bound=1.0
    )
    ym, _, mm = apply_bound_force_bias_with_tangent(
        xbar - eps * dxbar, np.zeros_like(dxbar), max_bound=1.0
    )
    np.testing.assert_array_equal(mp, mm)  # no cap-boundary crossing in the stencil
    fd = (yp - ym) / (2 * eps)
    np.testing.assert_allclose(fd, dxbar_out, rtol=1e-6, atol=1e-9)


@pytest.mark.unit
def test_vhs_taylor_tangent_vs_fd():
    rng = np.random.default_rng(71)
    chol = np.stack([random_symmetric(rng, NAO, scale=0.3) for _ in range(NCHOL)])
    dchol = np.stack([random_symmetric(rng, NAO, scale=0.05) for _ in range(NCHOL)])
    xs = rng.standard_normal((NW, NCHOL)) + 1j * rng.standard_normal((NW, NCHOL))
    dxs = 0.3 * (rng.standard_normal((NW, NCHOL)) + 1j * rng.standard_normal((NW, NCHOL)))
    phi = rng.standard_normal((NW, NAO, NOCC)) + 1j * rng.standard_normal((NW, NAO, NOCC))
    dphi = 0.3 * (
        rng.standard_normal((NW, NAO, NOCC)) + 1j * rng.standard_normal((NW, NAO, NOCC))
    )
    isqrtt = 1j * np.sqrt(DT)

    vhs, dvhs = construct_vhs_with_tangent(isqrtt, chol, dchol, xs, dxs)
    _, dphi_out = apply_taylor_with_tangent(6, vhs, dvhs, phi, dphi)

    def value(t):
        vhs_t, _ = construct_vhs_with_tangent(
            isqrtt, chol + t * dchol, np.zeros_like(dchol), xs + t * dxs, np.zeros_like(dxs)
        )
        out, _ = apply_taylor_with_tangent(
            6, vhs_t, np.zeros_like(vhs_t), phi + t * dphi, np.zeros_like(dphi)
        )
        return out

    eps = 1e-6
    fd = (value(eps) - value(-eps)) / (2 * eps)
    np.testing.assert_allclose(fd, dphi_out, rtol=1e-6, atol=1e-9)


def step_at(ham, trial_psi, phi, dphi, weight, dweight, x, eps, **prop_kwargs):
    """Run one value-branch step at lambda = eps along all stored tangents."""
    ham_e = shifted_copy(ham, eps)
    trial_e = SDTrial(trial_psi, NOCC)
    trial_e.half_rot(ham_e)
    prop_e = GradPropagator(DT, ham_e, trial_e, 1, **prop_kwargs)
    walkers_e = GradWalkers(
        phi.shape[0],
        phi + eps * dphi,
        np.zeros_like(phi),
        weight + eps * dweight,
        np.zeros_like(weight),
    )
    return prop_e.propagate_walkers(walkers_e, ham_e, trial_e, x)


@pytest.mark.unit
def test_single_step_tangent_vs_fd_caps_off():
    ham, trial, rng = make_ham_and_trial(seed=73)
    phi = np.array([trial.psi] * NW, dtype=np.complex128)
    phi += 0.1 * (rng.standard_normal(phi.shape) + 1j * rng.standard_normal(phi.shape))
    dphi = 0.3 * (rng.standard_normal(phi.shape) + 1j * rng.standard_normal(phi.shape))
    weight = rng.uniform(0.5, 2.0, NW)
    dweight = rng.standard_normal(NW)
    x = rng.standard_normal((NW, NCHOL))
    kwargs = dict(fbbound=1e8, apply_weight_bound=False)

    prop = GradPropagator(DT, ham, trial, 1, **kwargs)
    walkers = GradWalkers(NW, phi, dphi, weight, dweight)
    out = prop.propagate_walkers(walkers, ham, trial, x)

    eps = 1e-5
    outp = step_at(ham, trial.psi, phi, dphi, weight, dweight, x, eps, **kwargs)
    outm = step_at(ham, trial.psi, phi, dphi, weight, dweight, x, -eps, **kwargs)
    fd_phi = (outp.phi - outm.phi) / (2 * eps)
    fd_w = (outp.weight - outm.weight) / (2 * eps)
    # No QR inside a step, so raw dphi is directly comparable (no gauge freedom).
    np.testing.assert_allclose(fd_phi, out.dphi, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(fd_w, out.dweight, rtol=1e-6, atol=1e-8)


@pytest.mark.unit
def test_xbar_override_self_consistency():
    """Overriding the force bias with the natural (capped) value reproduces the
    step exactly; last_xbar exposes the value used."""
    ham, trial, rng = make_ham_and_trial(seed=83)
    phi = np.array([trial.psi] * NW, dtype=np.complex128)
    phi += 0.1 * (rng.standard_normal(phi.shape) + 1j * rng.standard_normal(phi.shape))
    weight = rng.uniform(0.5, 2.0, NW)
    x = rng.standard_normal((NW, NCHOL))

    prop = GradPropagator(DT, ham, trial, 1)
    walkers = GradWalkers(NW, phi, np.zeros_like(phi), weight, np.zeros(NW))
    out = prop.propagate_walkers(walkers, ham, trial, x)
    xbar_used = prop.last_xbar.copy()

    prop2 = GradPropagator(DT, ham, trial, 1)
    walkers2 = GradWalkers(NW, phi.copy(), np.zeros_like(phi), weight.copy(), np.zeros(NW))
    out2 = prop2.propagate_walkers(walkers2, ham, trial, x, xbar_override=xbar_used)
    np.testing.assert_array_equal(out2.phi, out.phi)
    np.testing.assert_array_equal(out2.weight, out.weight)


@pytest.mark.unit
def test_single_step_weight_cap_tangent_vs_fd():
    ham, trial, rng = make_ham_and_trial(seed=79)
    phi = np.array([trial.psi] * NW, dtype=np.complex128)
    phi += 0.1 * (rng.standard_normal(phi.shape) + 1j * rng.standard_normal(phi.shape))
    dphi = 0.3 * (rng.standard_normal(phi.shape) + 1j * rng.standard_normal(phi.shape))
    # One dominant walker so the 0.1*sum(w) cap deterministically binds it.
    weight = np.array([10.0, 1.0, 1.0, 1.0, 1.0])
    dweight = rng.standard_normal(NW)
    x = rng.standard_normal((NW, NCHOL))
    kwargs = dict(fbbound=1e8, apply_weight_bound=True, debug=True)

    prop = GradPropagator(DT, ham, trial, 1, **kwargs)
    walkers = GradWalkers(NW, phi, dphi, weight, dweight)
    out = prop.propagate_walkers(walkers, ham, trial, x)
    cap_mask = prop.diagnostics[-1]["weight_cap_mask"]
    assert cap_mask[0] and not cap_mask[1:].any()

    eps = 1e-5
    outp = step_at(ham, trial.psi, phi, dphi, weight, dweight, x, eps, **kwargs)
    outm = step_at(ham, trial.psi, phi, dphi, weight, dweight, x, -eps, **kwargs)
    fd_w = (outp.weight - outm.weight) / (2 * eps)
    np.testing.assert_allclose(fd_w, out.dweight, rtol=1e-6, atol=1e-8)
