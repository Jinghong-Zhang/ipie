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

import numpy
import pytest
import scipy.linalg

import ipie.addons.thermal.thermofield.trial.thermofield as trial_module
from ipie.addons.thermal.propagation.force_bias import construct_force_bias
from ipie.addons.thermal.thermofield.propagation.phaseless_thermofield import ThermofieldPhaseless
from ipie.addons.thermal.thermofield.trial.thermofield import ThermofieldThermalTrial
from ipie.addons.thermal.thermofield.utils.testing import (
    build_noninteracting_test_case_handlers,
    noninteracting_exact,
)
from ipie.addons.thermal.thermofield.walkers.uhf_walkers import ThermofieldWalkers
from ipie.hamiltonians.generic import GenericComplexChol, GenericRealChol


@pytest.mark.unit
def test_noninteracting_propagation_is_exact():
    # System params.
    nbasis = 4

    # Thermofield AFQMC params.
    mu = 0.2
    timestep = 0.05
    stabilize_freq = 5

    # Without a two-body term the walk is deterministic and exact:
    # Delta(beta / 2) = exp(-beta (h - mu) / 2) and G_T is the Fermi function.
    for beta in (0.5, 2.0, 20.0):
        objs = build_noninteracting_test_case_handlers(nbasis, mu, beta, timestep)
        h1e = objs["h1e"]
        hamiltonian = objs["hamiltonian"]
        trial = objs["trial"]
        walkers = objs["walkers"]
        propagator = objs["propagator"]

        nslices = int(numpy.rint(0.5 * beta / timestep))
        numpy.random.seed(7)
        for t in range(nslices):
            propagator.propagate_walkers(walkers, hamiltonian, trial)
            if (t + 1) % stabilize_freq == 0:
                walkers.stabilize()

        Delta_exact = scipy.linalg.expm(-0.5 * beta * (h1e - mu * numpy.eye(nbasis)))
        for iw in range(walkers.nwalkers):
            physical = walkers.get_delta(iw)
            for s in range(2):
                numpy.testing.assert_allclose(physical[s], Delta_exact, rtol=1e-9, atol=1e-10)

        # The mixed Green's function equals the exact Fermi function.
        f, _, _ = noninteracting_exact(h1e, mu, beta)
        G = trial.calc_greens_function(walkers.Qmat[0], walkers.log_d[0], walkers.Tmat[0])
        numpy.testing.assert_allclose(G[0], f, atol=1e-10)
        numpy.testing.assert_allclose(G[1], f, atol=1e-10)

        # Weights are positive and identical across walkers (no fields).
        assert numpy.all(walkers.weight > 0)
        assert walkers.weight[0] == pytest.approx(walkers.weight[1], rel=1e-12)


@pytest.mark.unit
def test_stabilization_preserves_physical_state():
    # System params.
    nbasis = 4

    # Thermofield AFQMC params.
    mu = 0.0
    beta = 8.0
    timestep = 0.1
    nslices = 20

    objs = build_noninteracting_test_case_handlers(nbasis, mu, beta, timestep)
    hamiltonian = objs["hamiltonian"]
    trial = objs["trial"]
    walkers = objs["walkers"]
    propagator = objs["propagator"]
    numpy.random.seed(7)
    for _ in range(nslices):
        propagator.propagate_walkers(walkers, hamiltonian, trial)

    # The QDT refactorization leaves the overlap and G_T unchanged.
    Q, log_d, T = walkers.Qmat[0], walkers.log_d[0], walkers.Tmat[0]
    log_ovlp_before = trial.calc_log_overlap(Q, log_d, T)
    G_before = trial.calc_greens_function(Q, log_d, T)
    walkers.stabilize()
    assert numpy.any(walkers.log_d != 0.0)
    Q, log_d, T = walkers.Qmat[0], walkers.log_d[0], walkers.Tmat[0]
    log_ovlp_after = trial.calc_log_overlap(Q, log_d, T)
    G_after = trial.calc_greens_function(Q, log_d, T)
    assert log_ovlp_after == pytest.approx(log_ovlp_before, abs=1e-11)
    numpy.testing.assert_allclose(G_after, G_before, atol=1e-12)


@pytest.mark.unit
def test_propagation_reuses_overlap_factorization():
    # System params.
    nbasis = 4

    # Thermofield AFQMC params.
    mu = 0.1
    beta = 1.0
    timestep = 0.05
    nwalkers = 3

    objs = build_noninteracting_test_case_handlers(nbasis, mu, beta, timestep, nwalkers=nwalkers)
    hamiltonian = objs["hamiltonian"]
    trial = objs["trial"]
    walkers = objs["walkers"]
    propagator = objs["propagator"]

    # One batched factorization per spin block per slice, i.e. at most two
    # kernel calls for the whole population. Either kernel may serve the guide
    # overlap (one-sided for the frontier split, two-sided for the zero split).
    names = ("one_plus_qdy_batched", "stabilized_inverse_one_plus_batched")
    originals = {name: getattr(trial_module, name) for name in names}
    calls = 0

    def counted_factory(original):
        def counted(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        return counted

    for name in names:
        setattr(trial_module, name, counted_factory(originals[name]))
    try:
        numpy.random.seed(7)
        propagator.propagate_walkers(walkers, hamiltonian, trial)
    finally:
        for name in names:
            setattr(trial_module, name, originals[name])

    assert 1 <= calls <= 2


@pytest.mark.unit
def test_batched_propagation_matches_scalar_loop():
    # System params.
    nbasis = 4

    # Thermofield AFQMC params.
    mu = 0.1
    beta = 1.0
    timestep = 0.05
    nwalkers = 3
    nslices = 6
    seed = 5
    numpy.random.seed(seed)

    h1e = numpy.random.standard_normal((nbasis, nbasis))
    h1e = 0.5 * (h1e + h1e.T)
    L = numpy.random.standard_normal((nbasis, nbasis))
    L = 0.5 * (L + L.T)
    chol = 0.3 * L.ravel()[:, None]
    hamiltonian = GenericRealChol(numpy.array([h1e, h1e]), chol)

    # The batched path reproduces the per-walker loop, including the
    # free-projection phase.
    for phaseless in (True, False):
        results = []
        for batched in (True, False):
            trial = ThermofieldThermalTrial(hamiltonian, beta=beta, mu=mu)
            walkers = ThermofieldWalkers(trial, nbasis, nwalkers)
            propagator = ThermofieldPhaseless(timestep, mu, phaseless=phaseless)
            propagator.build(hamiltonian, trial=trial, walkers=walkers)
            propagator.batched = batched
            for t in range(nslices):
                numpy.random.seed(100 + t)
                propagator.propagate_walkers(walkers, hamiltonian, trial)
            results.append(
                (
                    walkers.weight.copy(),
                    walkers.phase.copy(),
                    walkers.Qmat.copy(),
                    walkers.log_ovlp.copy(),
                    walkers.Ga.copy(),
                )
            )
        for a, b in zip(*results):
            numpy.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-13)


@pytest.mark.unit
def test_growth_bound_clips_log_growth():
    # Thermofield AFQMC params.
    mu = 0.1
    timestep = 0.05
    growth_bound = 0.5
    growth_center = 0.02

    propagator = ThermofieldPhaseless(timestep, mu)
    propagator.mf_const_fac = 0.9
    propagator.growth_bound = growth_bound
    propagator.growth_center = growth_center
    lmin, lmax = propagator.growth_bound_range()
    assert lmax - lmin == pytest.approx(2.0 * growth_bound * numpy.sqrt(2.0 * timestep))

    # Walkers above, inside and below the range, a dead walker above it, and a
    # walker with an infinite log growth.
    log_R = numpy.array([1.0 + 0.3j, 0.05 - 0.2j, -1.0 + 0.1j, 2.0, numpy.inf])
    weight = numpy.array([1.0, 1.0, 1.0, 0.0, 1.0])
    with numpy.errstate(over="ignore", invalid="ignore"):
        magn = numpy.abs(propagator.mf_const_fac * numpy.exp(log_R))
    ref = magn.copy()
    propagator.apply_bound_growth(magn, log_R, weight)
    assert magn[0] == numpy.exp(lmax)
    assert magn[1] == ref[1]
    assert magn[2] == numpy.exp(lmin)
    assert magn[3] == numpy.exp(lmax)
    assert numpy.isinf(magn[4])
    assert propagator.ngrowth_trig == 2


@pytest.mark.unit
def test_growth_bound_batched_matches_scalar_loop():
    # System params.
    nbasis = 4

    # Thermofield AFQMC params.
    mu = 0.1
    beta = 1.0
    timestep = 0.05
    nwalkers = 3
    nslices = 6
    seed = 5
    numpy.random.seed(seed)

    h1e = numpy.random.standard_normal((nbasis, nbasis))
    h1e = 0.5 * (h1e + h1e.T)
    L = numpy.random.standard_normal((nbasis, nbasis))
    L = 0.5 * (L + L.T)
    chol = 0.3 * L.ravel()[:, None]
    hamiltonian = GenericRealChol(numpy.array([h1e, h1e]), chol)

    # A bound too wide to act leaves the walk bitwise unchanged; a tight bound
    # acts on the batched and the per-walker paths alike.
    runs = {}
    for growth_bound, batched in ((0.0, True), (1e6, True), (0.05, True), (0.05, False)):
        trial = ThermofieldThermalTrial(hamiltonian, beta=beta, mu=mu)
        walkers = ThermofieldWalkers(trial, nbasis, nwalkers)
        propagator = ThermofieldPhaseless(timestep, mu)
        propagator.build(hamiltonian, trial=trial, walkers=walkers)
        propagator.batched = batched
        propagator.growth_bound = growth_bound
        for t in range(nslices):
            numpy.random.seed(100 + t)
            propagator.propagate_walkers(walkers, hamiltonian, trial)
        runs[(growth_bound, batched)] = (
            propagator.ngrowth_trig,
            walkers.weight.copy(),
            walkers.Qmat.copy(),
            walkers.log_ovlp.copy(),
        )
    off, wide = runs[(0.0, True)], runs[(1e6, True)]
    assert off[0] == 0 and wide[0] == 0
    for a, b in zip(off[1:], wide[1:]):
        numpy.testing.assert_array_equal(a, b)
    tight, tight_scalar = runs[(0.05, True)], runs[(0.05, False)]
    assert tight[0] > 0
    assert tight[0] == tight_scalar[0]
    assert not numpy.array_equal(tight[1], off[1])
    for a, b in zip(tight[1:], tight_scalar[1:]):
        numpy.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-13)


@pytest.mark.unit
def test_force_bias_sign_finite_difference_real_chol():
    # System params.
    nbasis = 3

    # Thermofield AFQMC params.
    mu = 0.0
    beta = 1.3
    timestep = 0.01
    nwalkers = 1
    seed = 31
    numpy.random.seed(seed)

    h1e = numpy.random.standard_normal((nbasis, nbasis))
    h1e = 0.5 * (h1e + h1e.T)
    L = numpy.random.standard_normal((nbasis, nbasis))
    L = 0.5 * (L + L.T)
    chol = L.ravel()[:, None]
    hamiltonian = GenericRealChol(numpy.array([h1e, h1e]), chol)
    trial = ThermofieldThermalTrial(hamiltonian, beta=beta, mu=mu)
    walkers = ThermofieldWalkers(trial, nbasis, nwalkers)
    # Random complex walker.
    shape = (2, nbasis, nbasis)
    Delta = numpy.random.standard_normal(shape) + 1j * numpy.random.standard_normal(shape)
    walkers.set_delta(0, Delta, trial=trial)

    propagator = ThermofieldPhaseless(timestep, mu)
    propagator.build(hamiltonian, trial=trial, walkers=walkers)
    sqrt_dt = timestep**0.5

    # The force bias is the derivative of the guide overlap,
    # d/dx log S_T(exp(i sqrt(dt) x L) Delta) at x = 0 equals i sqrt(dt) sum_s Tr(L G_T,s).
    def log_ovlp(x):
        BV = scipy.linalg.expm(1j * sqrt_dt * x * L)
        Delta_x = numpy.array([BV @ Delta[s] for s in range(2)])
        return trial.calc_log_overlap(Delta_x)

    dx = 1e-4
    deriv = (log_ovlp(dx) - log_ovlp(-dx)) / (2 * dx)

    vbias = construct_force_bias(hamiltonian, walkers)[0, 0]
    G = trial.calc_greens_function(Delta)
    trace = numpy.trace(L @ G[0]) + numpy.trace(L @ G[1])
    assert vbias == pytest.approx(trace, abs=1e-11)
    assert deriv == pytest.approx(1j * sqrt_dt * vbias, abs=1e-6)

    # The shifted fields of the propagator satisfy
    # xbar = xi - xshifted = -sqrt(dt) (i vbias - mf_shift), i.e. minus the
    # overlap derivative plus the mean-field shift correction. The force-bias
    # bound does not rescale anything for this small system.
    numpy.random.seed(7)
    _, _, xshifted, _ = propagator.construct_two_body_propagator(
        walkers, hamiltonian, trial, debug=True
    )
    xbar = propagator.xi[0] - xshifted[:, 0]
    expected = -sqrt_dt * (1j * vbias - propagator.mf_shift)
    numpy.testing.assert_allclose(xbar, expected, atol=1e-10)
    numpy.testing.assert_allclose(xbar, -deriv + sqrt_dt * propagator.mf_shift, atol=1e-6)


@pytest.mark.unit
def test_force_bias_sign_finite_difference_complex_chol():
    # System params.
    nbasis = 2

    # Thermofield AFQMC params.
    mu = 0.0
    beta = 0.9
    timestep = 0.01
    nwalkers = 1
    seed = 41
    numpy.random.seed(seed)

    h1e = numpy.random.standard_normal((nbasis, nbasis))
    h1e = 0.5 * (h1e + h1e.T)
    shape = (nbasis, nbasis)
    Lc = 0.3 * (numpy.random.standard_normal(shape) + 1j * numpy.random.standard_normal(shape))
    # GenericComplexChol builds A = L + L^dagger and B = i (L - L^dagger).
    chol = Lc.ravel()[:, None]
    hamiltonian = GenericComplexChol(numpy.array([h1e, h1e], dtype=numpy.complex128), chol)
    trial = ThermofieldThermalTrial(hamiltonian, beta=beta, mu=mu)
    walkers = ThermofieldWalkers(trial, nbasis, nwalkers)
    shape = (2, nbasis, nbasis)
    Delta = numpy.random.standard_normal(shape) + 1j * numpy.random.standard_normal(shape)
    walkers.set_delta(0, Delta, trial=trial)

    propagator = ThermofieldPhaseless(timestep, mu)
    propagator.build(hamiltonian, trial=trial, walkers=walkers)
    sqrt_dt = timestep**0.5

    # Force-bias sign check for the A and B channels.
    vbias = construct_force_bias(hamiltonian, walkers)[0]
    A = hamiltonian.A[:, 0].reshape(nbasis, nbasis)
    B = hamiltonian.B[:, 0].reshape(nbasis, nbasis)

    for op, vb in ((A, vbias[0]), (B, vbias[1])):

        def log_ovlp(x, op=op):
            BV = scipy.linalg.expm(1j * sqrt_dt * x * op)
            Delta_x = numpy.array([BV @ Delta[s] for s in range(2)])
            return trial.calc_log_overlap(Delta_x)

        dx = 1e-4
        deriv = (log_ovlp(dx) - log_ovlp(-dx)) / (2 * dx)
        assert deriv == pytest.approx(1j * sqrt_dt * vb, abs=1e-6)


if __name__ == "__main__":
    test_noninteracting_propagation_is_exact()
    test_stabilization_preserves_physical_state()
    test_propagation_reuses_overlap_factorization()
    test_batched_propagation_matches_scalar_loop()
    test_force_bias_sign_finite_difference_real_chol()
    test_force_bias_sign_finite_difference_complex_chol()
