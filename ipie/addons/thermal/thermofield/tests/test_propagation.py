# Copyright 2026 The ipie Developers. All Rights Reserved.
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
# Author: Jinghong Zhang <jinghongzhang@fas.harvard.edu>
#
"""Propagation tests: noninteracting exactness (Test 2) and the
force-bias sign finite-difference check (Test 7)."""

import numpy
import pytest
import scipy.linalg

from ipie.addons.thermal.thermofield.propagation import ThermofieldPhaseless
from ipie.addons.thermal.thermofield.tests.ed_utils import (
    build_noninteracting_hamiltonian,
    noninteracting_exact,
)
from ipie.addons.thermal.thermofield.trial import ThermofieldThermalTrial
from ipie.addons.thermal.thermofield.walkers import ThermofieldWalkers
from ipie.hamiltonians.generic import GenericComplexChol, GenericRealChol


def build_noninteracting_setup(M, beta, mu, timestep, nwalkers=2, seed=11):
    rng = numpy.random.default_rng(seed)
    h = rng.standard_normal((M, M))
    h = 0.5 * (h + h.T)
    hamiltonian = build_noninteracting_hamiltonian(h)
    trial = ThermofieldThermalTrial(hamiltonian, beta=beta, mu=mu)
    walkers = ThermofieldWalkers(trial, M, nwalkers)
    propagator = ThermofieldPhaseless(timestep, mu)
    propagator.build(hamiltonian, trial=trial, walkers=walkers)
    return h, hamiltonian, trial, walkers, propagator


@pytest.mark.unit
@pytest.mark.parametrize("beta", [0.5, 2.0, 20.0])
def test_noninteracting_propagation_is_exact(beta):
    """With no two-body term the walk is deterministic and exact:
    Delta(beta/2) = exp(-beta (h - mu) / 2) and G_T = Fermi function."""
    M = 4
    mu = 0.2
    timestep = 0.05
    h, hamiltonian, trial, walkers, propagator = build_noninteracting_setup(M, beta, mu, timestep)

    nslices = int(numpy.rint(0.5 * beta / timestep))
    numpy.random.seed(7)
    for t in range(nslices):
        propagator.propagate_walkers(walkers, hamiltonian, trial)
        if (t + 1) % 5 == 0:
            walkers.stabilize()

    k = h - mu * numpy.eye(M)
    Delta_exact = scipy.linalg.expm(-0.5 * beta * k)
    for iw in range(walkers.nwalkers):
        physical = walkers.get_delta(iw)
        for s in range(2):
            numpy.testing.assert_allclose(physical[s], Delta_exact, rtol=1e-9, atol=1e-10)

    # Mixed Green's function equals the exact Fermi function.
    f, _, _ = noninteracting_exact(h, mu, beta)
    G = trial.calc_greens_function(walkers.Qmat[0], walkers.log_d[0], walkers.Tmat[0])
    numpy.testing.assert_allclose(G[0], f, atol=1e-10)
    numpy.testing.assert_allclose(G[1], f, atol=1e-10)

    # Weights are positive and identical across walkers (no fields).
    assert numpy.all(walkers.weight > 0)
    assert walkers.weight[0] == pytest.approx(walkers.weight[1], rel=1e-12)


@pytest.mark.unit
def test_stabilization_preserves_physical_state():
    """The QDT refactorization leaves overlap and G_T unchanged."""
    M = 4
    beta = 8.0
    _, hamiltonian, trial, walkers, propagator = build_noninteracting_setup(
        M, beta, mu=0.0, timestep=0.1
    )
    numpy.random.seed(7)
    for t in range(20):
        propagator.propagate_walkers(walkers, hamiltonian, trial)

    log_ovlp_before = trial.calc_log_overlap(walkers.Qmat[0], walkers.log_d[0], walkers.Tmat[0])
    G_before = trial.calc_greens_function(walkers.Qmat[0], walkers.log_d[0], walkers.Tmat[0])
    walkers.stabilize()
    assert numpy.any(walkers.log_d != 0.0)
    log_ovlp_after = trial.calc_log_overlap(walkers.Qmat[0], walkers.log_d[0], walkers.Tmat[0])
    G_after = trial.calc_greens_function(walkers.Qmat[0], walkers.log_d[0], walkers.Tmat[0])
    assert log_ovlp_after == pytest.approx(log_ovlp_before, abs=1e-11)
    numpy.testing.assert_allclose(G_after, G_before, atol=1e-12)


@pytest.mark.unit
def test_force_bias_sign_finite_difference_real_chol():
    """Test 7: d/dx log S_T(B(x) Delta)|_{x=0} = i sqrt(dt) sum_s Tr(L G_T,s),
    matching vbias from construct_force_bias, and the code's shifted fields
    satisfy xbar = -sqrt(dt) (i vbias - mf_shift)."""
    M = 3
    beta = 1.3
    timestep = 0.01
    rng = numpy.random.default_rng(31)
    h = rng.standard_normal((M, M))
    h = 0.5 * (h + h.T)
    L = rng.standard_normal((M, M))
    L = 0.5 * (L + L.T)
    chol = L.ravel()[:, None]
    hamiltonian = GenericRealChol(numpy.array([h, h]), chol)
    trial = ThermofieldThermalTrial(hamiltonian, beta=beta, mu=0.0)
    walkers = ThermofieldWalkers(trial, M, 1)
    # Random complex walker.
    Delta = rng.standard_normal((2, M, M)) + 1j * rng.standard_normal((2, M, M))
    walkers.set_delta(0, Delta, trial=trial)

    propagator = ThermofieldPhaseless(timestep, 0.0)
    propagator.build(hamiltonian, trial=trial, walkers=walkers)
    sqrt_dt = timestep**0.5

    # Central difference of log S_T(exp(i sqrt(dt) x L) Delta) at x = 0.
    def log_ovlp(x):
        BV = scipy.linalg.expm(1j * sqrt_dt * x * L)
        Delta_x = numpy.array([BV @ Delta[s] for s in range(2)])
        return trial.calc_log_overlap(Delta_x)

    dx = 1e-4
    deriv = (log_ovlp(dx) - log_ovlp(-dx)) / (2 * dx)

    from ipie.addons.thermal.propagation.force_bias import construct_force_bias

    vbias = construct_force_bias(hamiltonian, walkers)[0, 0]
    G = trial.calc_greens_function(Delta)
    trace = numpy.trace(L @ G[0]) + numpy.trace(L @ G[1])
    assert vbias == pytest.approx(trace, abs=1e-11)
    assert deriv == pytest.approx(1j * sqrt_dt * vbias, abs=1e-6)

    # The propagator's actual shifted fields: xbar = xi - xshifted must equal
    # -sqrt(dt) (i vbias - mf_shift), i.e. minus the overlap derivative plus
    # the mean-field-shift correction.
    numpy.random.seed(7)
    _, _, xshifted, _ = propagator.construct_two_body_propagator(
        walkers, hamiltonian, trial, debug=True
    )
    xbar = propagator.xi[0] - xshifted[:, 0]
    expected = -sqrt_dt * (1j * vbias - propagator.mf_shift)
    # The force-bias bound may rescale large components; this system is small.
    numpy.testing.assert_allclose(xbar, expected, atol=1e-10)
    numpy.testing.assert_allclose(xbar, -deriv + sqrt_dt * propagator.mf_shift, atol=1e-6)


@pytest.mark.unit
def test_force_bias_sign_finite_difference_complex_chol():
    """Test 7 for the complex-Cholesky A/B channels."""
    M = 2
    beta = 0.9
    timestep = 0.01
    rng = numpy.random.default_rng(41)
    h = rng.standard_normal((M, M))
    h = 0.5 * (h + h.T)
    Lc = 0.3 * (rng.standard_normal((M, M)) + 1j * rng.standard_normal((M, M)))
    # GenericComplexChol builds A = L + L^dag, B = i (L - L^dag).
    chol = Lc.ravel()[:, None]
    hamiltonian = GenericComplexChol(numpy.array([h, h], dtype=numpy.complex128), chol)
    trial = ThermofieldThermalTrial(hamiltonian, beta=beta, mu=0.0)
    walkers = ThermofieldWalkers(trial, M, 1)
    Delta = rng.standard_normal((2, M, M)) + 1j * rng.standard_normal((2, M, M))
    walkers.set_delta(0, Delta, trial=trial)

    propagator = ThermofieldPhaseless(timestep, 0.0)
    propagator.build(hamiltonian, trial=trial, walkers=walkers)
    sqrt_dt = timestep**0.5

    from ipie.addons.thermal.propagation.force_bias import construct_force_bias

    vbias = construct_force_bias(hamiltonian, walkers)[0]
    A = hamiltonian.A[:, 0].reshape(M, M)
    B = hamiltonian.B[:, 0].reshape(M, M)

    for op, vb in ((A, vbias[0]), (B, vbias[1])):

        def log_ovlp(x, op=op):
            BV = scipy.linalg.expm(1j * sqrt_dt * x * op)
            Delta_x = numpy.array([BV @ Delta[s] for s in range(2)])
            return trial.calc_log_overlap(Delta_x)

        dx = 1e-4
        deriv = (log_ovlp(dx) - log_ovlp(-dx)) / (2 * dx)
        assert deriv == pytest.approx(1j * sqrt_dt * vb, abs=1e-6)
