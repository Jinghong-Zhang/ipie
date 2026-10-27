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

from itertools import product
from types import SimpleNamespace

import numpy
import pytest
from scipy.linalg import expm

from ipie.addons.thermal.thermofield.estimators.cyclic_average import cyclic_average_hubbard
from ipie.addons.thermal.thermofield.estimators.greens_function import thermofield_greens_function
from ipie.addons.thermal.thermofield.estimators.hubbard import HubbardSpinObservables
from ipie.addons.thermal.thermofield.hamiltonians.hubbard import build_hubbard_hamiltonian
from ipie.addons.thermal.thermofield.propagation.hubbard_discrete import ThermofieldHubbardDiscrete
from ipie.addons.thermal.thermofield.qmc.hubbard_afqmc import (
    propagate_population_cp,
    ThermofieldHubbardAFQMC,
)
from ipie.addons.thermal.thermofield.trial.hubbard import rhf_k_trial, ThermofieldCPTrial
from ipie.addons.thermal.thermofield.utils.testing import creation_operators
from ipie.addons.thermal.thermofield.walkers.pop_controller import ThermofieldPopController
from ipie.addons.thermal.thermofield.walkers.uhf_walkers import ThermofieldWalkers
from ipie.config import MPI
from ipie.utils.mpi import MPIHandler
from ipie.walkers.pop_controller import get_buffer, set_buffer

comm = MPI.COMM_WORLD


def _build_objects(U=4.0, beta=0.4, nwalkers=4, **kwargs):
    # Pinned 2x2 lattice, open in x, with an RHF guide. Keyword arguments go to the propagator.
    ham, hop, pin = build_hubbard_hamiltonian(
        2, 2, U, 0.5, pbc_x=False, pin_h=0.1, pin_edges=("left",)
    )
    k_trial, _, _ = rhf_k_trial(hop, pin, beta, 3.5)
    trial = ThermofieldCPTrial(ham, beta, k_trial=k_trial, guide_time=beta)
    mpi = MPIHandler()
    walkers = ThermofieldWalkers(trial, 4, nwalkers, mpi)
    options = dict(record_fields=True, guide_mode="fixed", fixed_guide_time=beta, energy_shift=-8.0)
    options.update(kwargs)
    prop = ThermofieldHubbardDiscrete(ham.H1 + U / 2 * numpy.eye(4), U, 0.02, beta, **options)
    return ham, hop, trial, walkers, prop, mpi


def _driver_stub(mpi, pop_control_freq=10000):
    # The driver attributes used by propagate_population_cp. The default frequency skips
    # population control.
    params = SimpleNamespace(num_stblz=5, pop_control_freq=pop_control_freq)
    return SimpleNamespace(mpi_handler=mpi, params=params)


def _slice_propagator(prop, fields):
    # Dense B = K D(x) K of one slice of spin Hirsch fields.
    diagonal = numpy.array(
        [numpy.diag(numpy.exp(prop.gamma * fields)), numpy.diag(numpy.exp(-prop.gamma * fields))]
    )
    return prop.BK2 @ diagonal @ prop.BK2


def _normalized_energy(results):
    # Ratio of means of the endpoint energy with a block jackknife error.
    absolute = numpy.array([r["absolute_weight"] for r in results])
    num = numpy.array([r["numer_endpoint"][0] for r in results]) / absolute
    den = numpy.array([r["denom"] for r in results]) / absolute
    n = len(den)
    energy = (num.sum() / den.sum()).real
    jk = numpy.array([((num.sum() - num[i]) / (den.sum() - den[i])).real for i in range(n)])
    return energy, numpy.sqrt((n - 1) / n * ((jk - jk.mean()) ** 2).sum())


@pytest.mark.unit
@pytest.mark.parametrize("trace_branching", [False, True])
def test_discrete_driver_smoke(trace_branching):
    ham, hop, trial, walkers, prop, mpi = _build_objects(trace_branching=trace_branching)
    driver = ThermofieldHubbardAFQMC(
        ham,
        hop,
        trial,
        walkers,
        prop,
        mpi,
        2,
        2,
        nblocks=3,
        pop_control_freq=5,
        wrap_stride=5,
        measure_one_rdm=True,
        shell_distances_squared=(1, 2),
        measure_magnitude_only=True,
        preserve_population_normalization=True,
    )
    driver.run(verbose=False)
    out = driver.finalise(verbose=False)
    assert out["max_recdev"] < 1e-10
    assert out["max_wrapdev"] < 1e-10
    assert driver.observables.U == 4.0
    assert driver.metadata["hs_type"] == "discrete_spin"
    assert driver.metadata["branching_mode"] == ("trace_magnitude" if trace_branching else "guide")
    assert driver.metadata["force_bias_mode"] == "none"
    assert numpy.isin(walkers.field_hist[walkers.weight != 0], [-1.0, 1.0]).all()
    assert numpy.isfinite(driver.one_rdm_summary()["mean"]).all()
    assert numpy.isfinite(driver.results[0]["numer_endpoint_magnitude"]).all()


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["fixed", "completion"])
@pytest.mark.parametrize("batched", [False, True])
def test_discrete_trace_branching_preserves_paths_and_endpoint_integrands(mode, batched):
    # The same sampled fields give equal physical contributions before resampling.
    paths = []
    for trace_branching in (False, True):
        _, _, trial, walkers, prop, mpi = _build_objects(
            beta=1.0, guide_mode=mode, batched=batched, trace_branching=trace_branching
        )
        initial_trace = trial.calc_log_trace_batched(
            walkers.Qmat, walkers.log_d, walkers.Tmat
        ).copy()
        initial_guide, _ = trial.calc_log_overlap_and_greens_function_scaled_batched(
            walkers.Qmat, walkers.log_d, walkers.Tmat, 1.0
        )
        numpy.random.seed(713)
        propagate_population_cp(
            _driver_stub(mpi), walkers, trial, prop, 50, SimpleNamespace(total_weight=4.0)
        )
        trace, green = trial.calc_log_overlap_and_greens_function_scaled_batched(
            walkers.Qmat, walkers.log_d, walkers.Tmat, 0.0
        )
        assert numpy.all(walkers.weight > 0)
        assert prop.trace_invalid_count == 0
        assert prop.invalid_weight_count == 0
        paths.append((walkers, trace, green, initial_trace, initial_guide))
    a, b = paths
    numpy.testing.assert_array_equal(a[0].field_hist, b[0].field_hist)
    numpy.testing.assert_array_equal(a[0].Qmat, b[0].Qmat)
    numpy.testing.assert_array_equal(a[0].log_ovlp, b[0].log_ovlp)
    logq = a[1] - a[0].log_ovlp
    expected = logq.real + (a[4] - a[3]).real
    numpy.testing.assert_allclose(numpy.log(b[0].weight / a[0].weight), expected, atol=1e-10)
    physical_a = a[0].weight * numpy.exp(logq + a[4])
    physical_b = b[0].weight * numpy.exp(1j * logq.imag + b[3])
    numpy.testing.assert_allclose(physical_b, physical_a, rtol=1e-10, atol=1e-10)
    numpy.testing.assert_allclose(b[0].log_tr, b[1], atol=1e-10)
    numpy.testing.assert_allclose(a[2], b[2], atol=1e-12)


@pytest.mark.unit
def test_discrete_trace_branching_keeps_negative_physical_trace():
    # A positive fixed guide must not impose positivity of the physical trace.
    ham, _, _, walkers, prop, _ = _build_objects(U=0.0, trace_branching=True)
    trial = ThermofieldCPTrial(
        ham, prop.beta, k_trial=numpy.tile(2 * numpy.eye(4), (2, 1, 1)), guide_time=prop.beta
    )
    delta = numpy.tile(numpy.eye(4), (2, 1, 1))
    delta[0, 0, 0] = -2.0
    for i in range(walkers.nwalkers):
        walkers.set_delta(i, delta, trial=trial)
    prop.BK2[:] = numpy.eye(4)
    prop.BK2inv[:] = numpy.eye(4)
    prop.logdet_BK2[:] = 0.0
    walkers.field_hist = numpy.zeros((walkers.nwalkers, 20, 4))
    prop.init_walker_cache(walkers, trial)
    assert numpy.all(numpy.cos(walkers.log_ovlp.imag) > 0)
    assert numpy.all(numpy.cos(walkers.log_tr.imag) < 0)
    numpy.random.seed(713)
    prop.propagate_walkers(walkers, trial, 0)
    assert numpy.all(walkers.weight > 0)
    assert numpy.all(numpy.cos(walkers.log_tr.imag) < 0)
    assert prop.trace_invalid_count == 0
    assert prop.kill_count == 0


@pytest.mark.unit
@pytest.mark.parametrize("U,dt", [(0.0, 0.02), (4.0, 0.02), (8.0, 0.3)])
def test_hirsch_identity_all_local_fock_states(U, dt):
    prop = ThermofieldHubbardDiscrete(numpy.zeros((2, 1, 1)), U, dt, 1.0)
    for up, dn in product([0, 1], repeat=2):
        hs = numpy.mean(prop.auxf[:, 0] ** up * prop.auxf[:, 1] ** dn)
        expected = numpy.exp(-dt * U * (up * dn - (up + dn) / 2))
        assert hs == pytest.approx(expected, rel=2e-14)


@pytest.mark.unit
def test_site_ratios_and_updates_against_noncommuting_dense_determinants():
    # System params.
    nsites = 4
    U = 4.0
    dt = 0.06
    beta = 2.0
    numpy.random.seed(81)

    h = numpy.random.normal(size=(2, nsites, nsites))
    h = (h + h.transpose(0, 2, 1)) * 0.2
    prop = ThermofieldHubbardDiscrete(h, U, dt, beta)
    lam = expm(-h * 0.7)
    delta = expm(numpy.random.normal(size=(2, nsites, nsites)) * 0.2)
    g = numpy.array([thermofield_greens_function(lam[s], delta[s]) for s in range(2)])[None]
    # Thermal, not Slater-idempotent.
    assert numpy.max(abs(g[0] @ g[0] - g[0])) > 0.01
    eye = numpy.eye(nsites)
    for site, xi in product(range(nsites), range(2)):
        expected_delta = delta.copy()
        expected_delta[:, site, :] *= prop.auxf[xi, :, None]
        overlap = numpy.prod([numpy.linalg.det(eye + delta[s] @ lam[s]) for s in range(2)])
        new_overlap = numpy.prod(
            [numpy.linalg.det(eye + expected_delta[s] @ lam[s]) for s in range(2)]
        )
        assert 2 * prop._site_probabilities(g, site)[0, xi] == pytest.approx(
            new_overlap / overlap, abs=2e-13
        )
        updated = g.copy()
        prop._site_green_update(updated, site, prop.delta[[xi]])
        exact = numpy.array(
            [thermofield_greens_function(lam[s], expected_delta[s]) for s in range(2)]
        )
        numpy.testing.assert_allclose(updated[0], exact, atol=2e-13)


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["completion", "fixed"])
@pytest.mark.parametrize("cache", [False, True])
def test_scalar_batched_paths_and_caches(mode, cache):
    _, _, trial, a, pa, mpi = _build_objects(guide_mode=mode, batched=False)
    _, _, _, b, pb, _ = _build_objects(guide_mode=mode, batched=True, cache_qinv=cache)
    driver = _driver_stub(mpi)
    pcontrol = SimpleNamespace(total_weight=4.0)
    numpy.random.seed(713)
    propagate_population_cp(driver, a, trial, pa, 20, pcontrol)
    numpy.random.seed(713)
    propagate_population_cp(driver, b, trial, pb, 20, pcontrol)
    numpy.testing.assert_array_equal(a.field_hist, b.field_hist)
    numpy.testing.assert_allclose(a.weight, b.weight, rtol=2e-11)
    numpy.testing.assert_allclose(a.Ga, b.Ga, atol=2e-11)
    for iw in range(4):
        numpy.testing.assert_allclose(a.get_delta(iw), b.get_delta(iw), atol=2e-11)
    lo, g = trial.calc_log_overlap_and_greens_function_scaled_batched(
        b.Qmat, b.log_d, b.Tmat, pb.guide_scale(19)
    )
    numpy.testing.assert_allclose(lo, b.log_ovlp, atol=2e-11)
    numpy.testing.assert_allclose(g[:, 0], numpy.eye(4) - b.Ga, atol=2e-11)
    if cache:
        numpy.testing.assert_allclose(
            b.Qinv @ b.Qmat, numpy.eye(4)[None, None] + numpy.zeros_like(b.Qmat), atol=2e-11
        )


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["completion", "fixed"])
def test_slice_weights_against_independent_dense_conditional_sampler(mode):
    _, _, trial, walkers, prop, _ = _build_objects(guide_mode=mode, nwalkers=1)
    prop.energy_shift = 0.0
    walkers.field_hist = numpy.zeros((1, 20, 4))
    prop.init_walker_cache(walkers, trial)
    # The propagator draws one uniform per site and slice; replay the same numbers.
    numpy.random.seed(713)
    uniforms = numpy.random.random((3, 4))
    numpy.random.seed(713)
    eye = numpy.eye(4)
    D = walkers.get_delta(0)
    for step in range(3):
        lam = expm(-prop.guide_scale(step) * trial.k_trial)
        old_scale = prop.beta if step == 0 else prop.guide_scale(step - 1)
        old_lam = expm(-old_scale * trial.k_trial)
        S0 = numpy.prod(numpy.linalg.det(eye + D @ old_lam))
        X = prop.BK2 @ D
        S = numpy.prod(numpy.linalg.det(eye + prop.BK2 @ X @ lam))
        weight = S / S0
        chosen = []
        for site in range(4):
            candidates = []
            for xi in range(2):
                new = X.copy()
                new[:, site, :] *= prop.auxf[xi, :, None]
                candidates.append(numpy.prod(numpy.linalg.det(eye + prop.BK2 @ new @ lam)))
            probs = numpy.maximum(numpy.real(numpy.asarray(candidates) / S) / 2, 0)
            pick = int(uniforms[step, site] >= probs[0] / sum(probs))
            weight *= sum(probs)
            S = candidates[pick]
            X[:, site, :] *= prop.auxf[pick, :, None]
            chosen.append(1 - 2 * pick)
        old_weight = walkers.weight[0]
        prop.propagate_walkers(walkers, trial, step)
        D = prop.BK2 @ X
        numpy.testing.assert_allclose(walkers.get_delta(0), D, atol=2e-12)
        numpy.testing.assert_array_equal(walkers.field_hist[0, step], chosen)
        assert walkers.weight[0] / old_weight == pytest.approx(weight.real, rel=2e-12)
        assert walkers.log_ovlp[0].real == pytest.approx(numpy.log(S).real, abs=2e-12)


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["completion", "fixed"])
@pytest.mark.parametrize("trace_branching", [False, True])
def test_noninteracting_exact_trace(mode, trace_branching):
    ham, hop, trial, walkers, prop, mpi = _build_objects(
        U=0.0, guide_mode=mode, trace_branching=trace_branching
    )
    driver = ThermofieldHubbardAFQMC(
        ham,
        hop,
        trial,
        walkers,
        prop,
        mpi,
        2,
        2,
        nblocks=2,
        wrap_stride=5,
        shell_distances_squared=(1, 2),
    )
    driver.run(verbose=False)
    G = numpy.eye(4) - numpy.linalg.inv(numpy.eye(4) + expm(-prop.beta * ham.H1))
    expected = driver.observables.measure(G)
    for r in driver.results:
        numpy.testing.assert_allclose(r["numer"] / r["denom"], expected, atol=1e-12)


@pytest.mark.unit
def test_history_buffer_and_wrap_reconstruct_actual_discrete_path():
    _, _, trial, walkers, prop, mpi = _build_objects()
    pcontrol = ThermofieldPopController(4, 20, mpi, "pair_branch", verbose=False)
    numpy.random.seed(713)
    propagate_population_cp(
        _driver_stub(mpi, pop_control_freq=5), walkers, trial, prop, 20, pcontrol
    )
    # The MPI walker buffer carries the field history.
    stored = get_buffer(walkers, 0)
    set_buffer(walkers, 1, stored)
    numpy.testing.assert_array_equal(walkers.field_hist[0], walkers.field_hist[1])
    actual = []
    dev, _ = cyclic_average_hubbard(
        walkers, trial, prop, 20, 3, lambda iw, g, cut: actual.append((iw, cut, g.copy()))
    )
    assert dev < 1e-10
    for iw, cut, g in actual:
        B = [_slice_propagator(prop, f) for f in walkers.field_hist[iw]]
        D = numpy.tile(numpy.eye(4), (2, 1, 1))
        for l in list(range(cut, 20)) + list(range(cut)):
            D = B[l] @ D
        numpy.testing.assert_allclose(
            g, numpy.eye(4) - numpy.linalg.inv(numpy.eye(4) + D), atol=2e-11
        )


@pytest.mark.skipif(comm.size < 2, reason="Requires at least two MPI ranks.")
@pytest.mark.mpi
@pytest.mark.parametrize("trace_branching", [False, True])
def test_discrete_mpi_population_history_and_reduction(trace_branching):
    ham, hop, trial, walkers, prop, mpi = _build_objects(trace_branching=trace_branching)
    driver = ThermofieldHubbardAFQMC(
        ham,
        hop,
        trial,
        walkers,
        prop,
        mpi,
        2,
        2,
        nblocks=2,
        wrap_stride=3,
        pop_control_freq=1,
        shell_distances_squared=(1, 2),
        measure_magnitude_only=True,
        preserve_population_normalization=True,
    )
    driver.run(verbose=False)
    all_rows = comm.allgather(driver.results[0]["numer"])
    for row in all_rows:
        numpy.testing.assert_array_equal(row, all_rows[0])
    all_magnitude = comm.allgather(driver.results[0]["numer_endpoint_magnitude"])
    for row in all_magnitude:
        numpy.testing.assert_array_equal(row, all_magnitude[0])
    assert driver.finalise(verbose=False)["max_recdev"] < 1e-10


@pytest.mark.unit
def test_two_slice_partition_and_energy_against_fock_space():
    # Sum over every discrete field, independently of Monte Carlo or a guide.
    # System params.
    nsites = 2
    U = 4.0
    dt = 0.1
    hop = numpy.array([[0.0, -1.0], [-1.0, 0.0]])
    potential = numpy.array([[0.4, 0.2], [0.2, 0.4]])

    kinetic = numpy.array([hop + numpy.diag(v) for v in potential])
    prop = ThermofieldHubbardDiscrete(kinetic, U, dt, 2 * dt)
    cre = creation_operators(2 * nsites)
    ann = [c.T for c in cre]
    number = [cre[i] @ ann[i] for i in range(2 * nsites)]
    hop_fock = sum(
        hop[i, j] * (cre[s * nsites + i] @ ann[s * nsites + j])
        for s in range(2)
        for i in range(nsites)
        for j in range(nsites)
    )
    kin_fock = hop_fock + sum(
        potential[s, i] * number[s * nsites + i] for s in range(2) for i in range(nsites)
    )
    interaction = U * sum(
        number[i] @ number[nsites + i] - 0.5 * (number[i] + number[nsites + i])
        for i in range(nsites)
    )
    K = expm(-dt * 0.5 * kin_fock)
    transfer = K @ expm(-dt * interaction) @ K
    rho = transfer @ transfer
    energy_op = hop_fock + U * sum(number[i] @ number[nsites + i] for i in range(nsites))
    exact_z = numpy.trace(rho)
    exact_e = numpy.trace(energy_op @ rho) / exact_z / nsites
    obs = HubbardSpinObservables(hop, U, nsites, 1, (1,))
    eye = numpy.eye(nsites)
    z = en = 0.0
    for fields in product([-1.0, 1.0], repeat=2 * nsites):
        D = numpy.tile(eye, (2, 1, 1))
        for f in numpy.reshape(fields, (2, nsites)):
            D = _slice_propagator(prop, f) @ D
        trace = numpy.prod(numpy.linalg.det(eye + D)) / 2 ** (2 * nsites)
        G = eye - numpy.linalg.inv(eye + D)
        z += trace
        en += trace * obs.measure(G)[0]
    assert z == pytest.approx(exact_z, rel=2e-14)
    assert en / z == pytest.approx(exact_e, abs=2e-14)


@pytest.mark.unit
def test_beta80_prescribed_field_wrap_against_spectral_reference():
    # An 8000-slice prescribed path; no Monte Carlo sampling or constraining trial.
    # System params.
    U = 4.0
    dt = 0.01
    beta = 80.0
    nslices = 8000
    ham, _, _ = build_hubbard_hamiltonian(4, 4, U, 0.5, pbc_x=False, pin_h=0.1)

    prop = ThermofieldHubbardDiscrete(ham.H1 + U / 2 * numpy.eye(16), U, dt, beta)
    fields = (-1.0) ** (numpy.arange(16) % 4 + numpy.arange(16) // 4)
    walkers = SimpleNamespace(
        nwalkers=1, weight=numpy.ones(1), field_hist=numpy.tile(fields, (1, nslices, 1))
    )
    walkers.field_hist[:, 1::2] *= -1
    B = _slice_propagator(prop, fields)
    opposite = _slice_propagator(prop, -fields)
    eig, vec = numpy.linalg.eig(opposite @ B)
    assert numpy.max(abs(eig.imag)) < 1e-12
    powers = nslices // 2 * numpy.log(eig.real)
    occupation = 0.5 * (1 + numpy.tanh(powers / 2))
    expected_g = (vec * occupation[:, None, :]) @ numpy.linalg.inv(vec)
    expected_log = numpy.logaddexp(0.0, powers).sum()
    measured = []
    dev, log0 = cyclic_average_hubbard(
        walkers,
        SimpleNamespace(nbasis=16),
        prop,
        nslices,
        1000,
        lambda iw, g, cut: measured.append(g.copy()),
    )
    assert len(measured) == 8
    assert dev < 1e-8
    assert log0[0].real == pytest.approx(expected_log, abs=1e-8)
    for g in measured:
        numpy.testing.assert_allclose(g, expected_g, atol=2e-10)


@pytest.mark.unit
def test_extreme_coherent_path_refuses_incomplete_wrap():
    # Repeating the same staggered field 8000 times gives log-scale spans beyond 3000. The
    # finite-precision product engine can lose cuts; its failure must be explicit, never an
    # average over the surviving cuts.
    # System params.
    U = 4.0
    dt = 0.01
    beta = 80.0
    nslices = 8000
    ham, _, _ = build_hubbard_hamiltonian(4, 4, U, 0.5, pbc_x=False, pin_h=0.1)

    prop = ThermofieldHubbardDiscrete(ham.H1 + U / 2 * numpy.eye(16), U, dt, beta)
    fields = (-1.0) ** (numpy.arange(16) % 4 + numpy.arange(16) // 4)
    walkers = SimpleNamespace(
        nwalkers=1, weight=numpy.ones(1), field_hist=numpy.tile(fields, (1, nslices, 1))
    )
    measured = []
    try:
        cyclic_average_hubbard(
            walkers,
            SimpleNamespace(nbasis=16),
            prop,
            nslices,
            1000,
            lambda iw, green, cut: measured.append(green.copy()),
        )
    except FloatingPointError as error:
        assert "Incomplete discrete-field" in str(error)
        assert len(measured) != 8
    else:
        # A future kernel may represent this extreme path. Accept that only with all cuts and
        # the correct spectral answer.
        eig, vec = numpy.linalg.eigh(_slice_propagator(prop, fields))
        occupation = 0.5 * (1 + numpy.tanh(nslices // 2 * numpy.log(eig)))
        expected = (vec * occupation[:, None, :]) @ vec.swapaxes(-1, -2)
        assert len(measured) == 8
        for green in measured:
            numpy.testing.assert_allclose(green, expected, atol=2e-10)


@pytest.mark.unit
def test_discrete_driver_metadata_uses_physical_interaction():
    # Object construction only: no path is sampled.
    ham, hop, trial, walkers, prop, mpi = _build_objects()
    driver = ThermofieldHubbardAFQMC(
        ham,
        hop,
        trial,
        walkers,
        prop,
        mpi,
        2,
        2,
        shell_distances_squared=(1, 2),
        metadata={"U": 123.0},
    )
    assert not numpy.isclose(prop.gamma**2 / prop.dt, 4.0)
    assert driver.observables.U == 4.0
    assert driver.metadata["U"] == 4.0
    assert driver.metadata["hs_type"] == "discrete_spin"
    assert driver.metadata["hs_gamma"] == prop.gamma


# ---------------------------------------------------------------------------
# Fixed-guide constraint with the completion guide as the importance function.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_completion_importance_option_validation():
    ham, _, _ = build_hubbard_hamiltonian(
        2, 2, 4.0, 0.5, pbc_x=False, pin_h=0.1, pin_edges=("left",)
    )
    h = ham.H1 + 2.0 * numpy.eye(4)
    with pytest.raises(ValueError):
        ThermofieldHubbardDiscrete(
            h, 4.0, 0.02, 1.0, guide_mode="completion", importance_guide="completion"
        )
    with pytest.raises(ValueError):
        ThermofieldHubbardDiscrete(
            h,
            4.0,
            0.02,
            1.0,
            guide_mode="fixed",
            fixed_guide_time=1.0,
            importance_guide="completion",
            trace_branching=True,
        )
    with pytest.raises(ValueError):
        ThermofieldHubbardDiscrete(
            h, 4.0, 0.02, 1.0, guide_mode="fixed", fixed_guide_time=1.0, importance_guide="trace"
        )


@pytest.mark.unit
def test_default_importance_guide_leaves_sampler_unchanged():
    # Without importance_guide the sampler is that of importance_guide="same" bit for bit,
    # while the completion importance function samples differently from the same numbers.
    runs = {}
    for label, kwargs in (
        ("default", {}),
        ("same", {"importance_guide": "same"}),
        ("completion", {"importance_guide": "completion"}),
    ):
        _, _, trial, walkers, prop, mpi = _build_objects(beta=1.0, **kwargs)
        numpy.random.seed(713)
        propagate_population_cp(
            _driver_stub(mpi), walkers, trial, prop, 20, SimpleNamespace(total_weight=4.0)
        )
        runs[label] = (prop, walkers.field_hist.copy(), walkers.weight.copy())
    assert runs["default"][0].importance_guide == "same"
    assert runs["default"][0].field_proposal == runs["same"][0].field_proposal
    numpy.testing.assert_array_equal(runs["default"][1], runs["same"][1])
    numpy.testing.assert_array_equal(runs["default"][2], runs["same"][2])
    assert not numpy.array_equal(runs["default"][2], runs["completion"][2])


@pytest.mark.unit
@pytest.mark.parametrize("batched", [False, True])
def test_completion_importance_noninteracting_exact(batched):
    ham, hop, trial, walkers, prop, mpi = _build_objects(
        U=0.0, beta=1.0, batched=batched, importance_guide="completion"
    )
    driver = ThermofieldHubbardAFQMC(
        ham,
        hop,
        trial,
        walkers,
        prop,
        mpi,
        2,
        2,
        nblocks=2,
        wrap_stride=None,
        shell_distances_squared=(1, 2),
    )
    driver.run(verbose=False)
    assert driver.metadata["importance_guide"] == "completion"
    G = numpy.eye(4) - numpy.linalg.inv(numpy.eye(4) + expm(-prop.beta * ham.H1))
    expected = driver.observables.measure(G)
    for r in driver.results:
        numpy.testing.assert_allclose(r["numer_endpoint"] / r["denom"], expected, atol=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize("batched", [False, True])
def test_completion_importance_endpoint_correction_is_a_sign(batched):
    # The cached endpoint overlap is |det(1 + Delta)|, so Q = det(1 + Delta) / |det(1 + Delta)|
    # is +-1, and the constraint cache is the (positive) fixed-guide overlap.
    _, _, trial, walkers, prop, mpi = _build_objects(
        beta=1.0, nwalkers=48, batched=batched, importance_guide="completion"
    )
    pcontrol = ThermofieldPopController(48, 50, mpi, "pair_branch", verbose=False)
    numpy.random.seed(713)
    propagate_population_cp(
        _driver_stub(mpi, pop_control_freq=5), walkers, trial, prop, 50, pcontrol
    )
    alive = walkers.weight > 0
    assert alive.sum() > 40
    trace = trial.calc_log_trace_batched(walkers.Qmat, walkers.log_d, walkers.Tmat)
    logq = trace[alive] - walkers.log_ovlp[alive]
    numpy.testing.assert_allclose(logq.real, 0.0, atol=1e-8)
    numpy.testing.assert_allclose(numpy.abs(numpy.cos(logq.imag)), 1.0, atol=1e-8)
    lo_f, _ = trial.calc_log_overlap_and_greens_function_scaled_batched(
        walkers.Qmat, walkers.log_d, walkers.Tmat, prop.fixed_guide_time
    )
    numpy.testing.assert_allclose(
        numpy.exp(walkers.log_ovlp_constraint[alive] - lo_f[alive]), 1.0, atol=1e-8
    )
    assert numpy.all(numpy.cos(walkers.log_ovlp_constraint[alive].imag) > 0)
    assert prop.invalid_weight_count == 0
    assert "log_ovlp_constraint" in walkers.buff_names


@pytest.mark.driver
def test_completion_importance_matches_guide_branching_statistically():
    # Same constraint, different importance function: the endpoint energies agree within
    # statistics, and the endpoint effective walker count is larger.
    out = {}
    for importance in ("same", "completion"):
        ham, hop, trial, walkers, prop, mpi = _build_objects(
            beta=1.0, nwalkers=256, importance_guide=importance
        )
        driver = ThermofieldHubbardAFQMC(
            ham,
            hop,
            trial,
            walkers,
            prop,
            mpi,
            2,
            2,
            nblocks=24,
            wrap_stride=None,
            shell_distances_squared=(1, 2),
            seed=17,
        )
        driver.run(verbose=False)
        ess = numpy.median(
            [r["absolute_weight"] ** 2 / r["squared_weight"] for r in driver.results]
        )
        out[importance] = (*_normalized_energy(driver.results), ess)
    (e1, s1, ess1), (e2, s2, ess2) = out["same"], out["completion"]
    assert abs(e1 - e2) < 4.0 * numpy.hypot(s1, s2), (e1, s1, e2, s2)
    assert ess2 > ess1


@pytest.mark.skipif(comm.size != 2, reason="Requires two MPI ranks.")
@pytest.mark.mpi
def test_completion_importance_caches_survive_mpi_population_control():
    # After every population-control event both caches (the importance |S_c| and the fixed
    # constraint overlap) equal a fresh recomputation on every rank, including for walkers
    # cloned from the other rank.
    # Thermofield AFQMC params.
    nwalkers = 6
    nslices = 50

    _, _, trial, walkers, prop, mpi = _build_objects(
        beta=1.0, nwalkers=nwalkers, importance_guide="completion"
    )
    pcontrol = ThermofieldPopController(nwalkers, nslices, mpi, "pair_branch", verbose=False)
    # No driver loop here, so no field history buffer.
    prop.record_fields = False
    # Population control copies registered arrays, so these labels follow the walkers.
    walkers.add_walker_array("label", numpy.zeros(nwalkers))
    prop.init_walker_cache(walkers, trial)
    prop.seed_qinv_cache(walkers)
    numpy.random.seed(3 + 11 * comm.rank)
    cross_rank_clones = 0
    for l in range(nslices):
        prop.propagate_walkers(walkers, trial, l)
        if l == 0:
            continue
        walkers.label[:] = comm.rank * nwalkers + numpy.arange(nwalkers)
        if l % 10 == 0 and comm.rank == 1:
            # Light walkers on rank 1 are replaced by clones of walkers from rank 0.
            walkers.weight *= 0.01
        pcontrol.pop_control(walkers, comm)
        prop.seed_qinv_cache(walkers)
        cross_rank_clones += int(numpy.sum(walkers.label // nwalkers != comm.rank))
        alive = walkers.weight > 0
        lo_f, _ = trial.calc_log_overlap_and_greens_function_scaled_batched(
            walkers.Qmat, walkers.log_d, walkers.Tmat, prop.fixed_guide_time
        )
        lo_c, _ = trial.calc_log_overlap_and_greens_function_scaled_batched(
            walkers.Qmat, walkers.log_d, walkers.Tmat, prop._completion_scale(l)
        )
        numpy.testing.assert_allclose(
            numpy.exp(walkers.log_ovlp_constraint[alive] - lo_f[alive]), 1.0, atol=1e-8
        )
        numpy.testing.assert_allclose(walkers.log_ovlp[alive].real, lo_c[alive].real, atol=1e-8)
    assert comm.allreduce(cross_rank_clones) > 0
    assert comm.allreduce(int(numpy.all(walkers.weight > 0))) >= 1


# ---------------------------------------------------------------------------
# Tempered trace branching: importance S^(1 - eps) |T|^eps with the fixed constraint.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_tempered_exponent_validation():
    with pytest.raises(ValueError):
        _build_objects(beta=1.0, trace_branching=True, trace_exponent=1.5)
    with pytest.raises(ValueError):
        _build_objects(beta=1.0, trace_branching=True, trace_exponent=-0.1)


@pytest.mark.unit
@pytest.mark.parametrize("batched", [False, True])
def test_tempered_exponent_zero_is_guide_branching_bitwise(batched):
    # Exponent 0 with trace branching reproduces plain guide branching: same fields, weights
    # and endpoint numerators and denominators.
    outs = []
    for trace_branching, eps in ((False, 1.0), (True, 0.0)):
        ham, hop, trial, walkers, prop, mpi = _build_objects(
            beta=1.0,
            nwalkers=8,
            batched=batched,
            trace_branching=trace_branching,
            trace_exponent=eps,
        )
        driver = ThermofieldHubbardAFQMC(
            ham,
            hop,
            trial,
            walkers,
            prop,
            mpi,
            2,
            2,
            nblocks=2,
            wrap_stride=None,
            shell_distances_squared=(),
            seed=5,
        )
        driver.run(verbose=False)
        outs.append(
            (
                walkers.Qmat.copy(),
                walkers.weight.copy(),
                numpy.array([r["numer_endpoint"] for r in driver.results]),
                numpy.array([r["denom"] for r in driver.results]),
            )
        )
    # The same sampled fields give the same final Q factors.
    numpy.testing.assert_array_equal(outs[0][0], outs[1][0])
    numpy.testing.assert_allclose(outs[0][1], outs[1][1], rtol=1e-12, atol=1e-14)
    numpy.testing.assert_allclose(outs[0][2], outs[1][2], rtol=1e-10, atol=1e-12)
    numpy.testing.assert_allclose(outs[0][3], outs[1][3], rtol=1e-10, atol=1e-12)


@pytest.mark.unit
def test_tempered_exponent_one_is_trace_branching_default():
    ham, hop, trial, walkers, prop, mpi = _build_objects(beta=1.0, trace_branching=True)
    assert prop.trace_exponent == 1.0 and prop.trace_branching
    driver = ThermofieldHubbardAFQMC(
        ham,
        hop,
        trial,
        walkers,
        prop,
        mpi,
        2,
        2,
        nblocks=1,
        wrap_stride=None,
        shell_distances_squared=(),
    )
    assert driver.metadata["trace_exponent"] == 1.0
    assert driver.metadata["branching_mode"] == "trace_magnitude"


@pytest.mark.unit
@pytest.mark.parametrize("eps", [0.3, 0.5])
def test_tempered_noninteracting_exact(eps):
    ham, hop, trial, walkers, prop, mpi = _build_objects(
        U=0.0, beta=1.0, trace_branching=True, trace_exponent=eps
    )
    driver = ThermofieldHubbardAFQMC(
        ham,
        hop,
        trial,
        walkers,
        prop,
        mpi,
        2,
        2,
        nblocks=2,
        wrap_stride=None,
        shell_distances_squared=(1, 2),
    )
    driver.run(verbose=False)
    G = numpy.eye(4) - numpy.linalg.inv(numpy.eye(4) + expm(-prop.beta * ham.H1))
    expected = driver.observables.measure(G)
    for r in driver.results:
        numpy.testing.assert_allclose(r["numer_endpoint"] / r["denom"], expected, atol=1e-12)


@pytest.mark.unit
def test_tempered_endpoint_weight_identity_without_resampling():
    # Without population control, w sign(T) |T/S|^(1 - eps) is the same physical path weight
    # for every eps up to the common constant f(Delta_0) of the importance function, so the
    # relative weights across walkers agree.
    phys, fields = [], []
    for eps in (0.0, 0.5, 1.0):
        _, _, trial, walkers, prop, mpi = _build_objects(
            beta=1.0, nwalkers=6, trace_branching=True, trace_exponent=eps
        )
        # The same field stream for every eps.
        numpy.random.seed(99)
        propagate_population_cp(
            _driver_stub(mpi), walkers, trial, prop, 50, SimpleNamespace(total_weight=6.0)
        )
        trace = trial.calc_log_trace_batched(walkers.Qmat, walkers.log_d, walkers.Tmat)
        logq = trace - walkers.log_ovlp
        phys.append(walkers.weight * numpy.exp((1.0 - eps) * logq.real + 1j * logq.imag))
        fields.append(walkers.field_hist.copy())
    for other in fields[1:]:
        numpy.testing.assert_array_equal(other, fields[0])
    ref = phys[0] / phys[0].sum()
    numpy.testing.assert_allclose(phys[1] / phys[1].sum(), ref, rtol=1e-9, atol=1e-12)
    numpy.testing.assert_allclose(phys[2] / phys[2].sum(), ref, rtol=1e-9, atol=1e-12)


@pytest.mark.driver
def test_tempered_matches_guide_branching_statistically():
    out = {}
    for eps in (0.0, 0.5):
        ham, hop, trial, walkers, prop, mpi = _build_objects(
            beta=1.0, nwalkers=256, trace_branching=True, trace_exponent=eps
        )
        driver = ThermofieldHubbardAFQMC(
            ham,
            hop,
            trial,
            walkers,
            prop,
            mpi,
            2,
            2,
            nblocks=24,
            wrap_stride=None,
            shell_distances_squared=(),
            seed=17,
        )
        driver.run(verbose=False)
        out[eps] = _normalized_energy(driver.results)
    (e0, s0), (e1, s1) = out[0.0], out[0.5]
    assert abs(e0 - e1) < 4.0 * numpy.hypot(s0, s1), out


# ---------------------------------------------------------------------------
# Mixed (guide-bra) estimator: Tr[Lambda H B_w] / Tr[Lambda B_w], plain weights, no Q.
# ---------------------------------------------------------------------------


def _mixed_objects(U, walk_beta, guide_beta, k_trial=None, nwalkers=8):
    # Pinned 2x2 lattice walked to walk_beta under a fixed guide of time guide_beta.
    ham, hop, pin = build_hubbard_hamiltonian(
        2, 2, U, 0.5, pbc_x=False, pin_h=0.1, pin_edges=("left",)
    )
    if k_trial is None:
        k_trial, _, _ = rhf_k_trial(hop, pin, walk_beta + guide_beta, 3.5)
    trial = ThermofieldCPTrial(ham, walk_beta, k_trial=k_trial, guide_time=guide_beta)
    mpi = MPIHandler()
    walkers = ThermofieldWalkers(trial, 4, nwalkers, mpi)
    prop = ThermofieldHubbardDiscrete(
        ham.H1 + U / 2 * numpy.eye(4),
        U,
        0.02,
        walk_beta,
        record_fields=False,
        guide_mode="fixed",
        fixed_guide_time=guide_beta,
        batched=True,
        energy_shift=-8.0,
    )
    return ham, hop, trial, walkers, prop, mpi


@pytest.mark.unit
def test_mixed_estimator_exact_for_noninteracting_exact_trial():
    # U = 0 with k_T equal to the exact one-body Hamiltonian: the guide bra completes the walk
    # exactly, so the mixed estimator after a walk of beta/2 with guide beta/2 is the exact
    # grand-canonical energy at beta, and the physical endpoint estimator is exact at beta/2.
    # System params.
    walk = 0.4
    guide = 0.4
    ham, _, _ = build_hubbard_hamiltonian(
        2, 2, 0.0, 0.5, pbc_x=False, pin_h=0.1, pin_edges=("left",)
    )

    ham, hop, trial, walkers, prop, mpi = _mixed_objects(
        0.0, walk, guide, k_trial=numpy.array([ham.H1[0], ham.H1[1]])
    )
    driver = ThermofieldHubbardAFQMC(
        ham,
        hop,
        trial,
        walkers,
        prop,
        mpi,
        2,
        2,
        nblocks=2,
        wrap_stride=None,
        shell_distances_squared=(1, 2),
        seed=3,
    )
    driver.run(verbose=False)
    G_full = numpy.eye(4) - numpy.linalg.inv(numpy.eye(4) + expm(-(walk + guide) * ham.H1))
    G_half = numpy.eye(4) - numpy.linalg.inv(numpy.eye(4) + expm(-walk * ham.H1))
    for r in driver.results:
        numpy.testing.assert_allclose(
            r["numer_mixed"] / r["denom_mixed"], driver.observables.measure(G_full), atol=1e-10
        )
        numpy.testing.assert_allclose(
            r["numer_endpoint"] / r["denom"], driver.observables.measure(G_half), atol=1e-10
        )
        assert r["mixed_overlap_dev"] < 1e-8
    summary = driver.finalise(verbose=False)
    assert "mixed" in summary and numpy.isfinite(summary["max_mixed_overlap_dev"])


@pytest.mark.unit
def test_measure_mixed_false_keeps_endpoint_and_overlap_check():
    # With measure_mixed=False the endpoint sums are identical for the same seed, the mixed
    # observables are absent and the overlap check is kept.
    out = {}
    for flag in (True, False):
        ham, hop, trial, walkers, prop, mpi = _mixed_objects(4.0, 0.4, 0.4, nwalkers=16)
        driver = ThermofieldHubbardAFQMC(
            ham,
            hop,
            trial,
            walkers,
            prop,
            mpi,
            2,
            2,
            nblocks=2,
            wrap_stride=None,
            shell_distances_squared=(1, 2),
            seed=9,
            measure_mixed=flag,
        )
        driver.run(verbose=False)
        out[flag] = (driver.results, driver.finalise(verbose=False))
    (rows_t, sum_t), (rows_f, sum_f) = out[True], out[False]
    for a, b in zip(rows_t, rows_f):
        numpy.testing.assert_array_equal(a["numer_endpoint"], b["numer_endpoint"])
        assert a["denom"] == b["denom"] and a["mixed_overlap_dev"] == b["mixed_overlap_dev"] < 1e-8
        assert "numer_mixed" in a and "numer_mixed" not in b
    assert "mixed" in sum_t and "mixed" not in sum_f
    assert sum_f["max_mixed_overlap_dev"] == sum_t["max_mixed_overlap_dev"]


@pytest.mark.unit
def test_mixed_estimator_reduces_to_physical_with_identity_guide():
    # A fixed guide time of zero makes Lambda the identity: the guide bra is the physical
    # trace, Q = 1, and the mixed and endpoint accumulations coincide.
    ham, hop, trial, walkers, prop, mpi = _mixed_objects(4.0, 0.4, 0.0, nwalkers=16)
    driver = ThermofieldHubbardAFQMC(
        ham,
        hop,
        trial,
        walkers,
        prop,
        mpi,
        2,
        2,
        nblocks=2,
        wrap_stride=None,
        shell_distances_squared=(1, 2),
        seed=5,
    )
    driver.run(verbose=False)
    for r in driver.results:
        # Same walkers and weights up to the common endpoint shift exp(-shift): the ratios
        # agree exactly and the denominators differ by that constant only.
        numpy.testing.assert_allclose(
            r["numer_mixed"] / r["denom_mixed"],
            r["numer_endpoint"] / r["denom"],
            rtol=1e-10,
            atol=1e-12,
        )
        shift = r["log_shift"] - (
            r["population_log_normalization"] if driver.preserve_population_normalization else 0.0
        )
        numpy.testing.assert_allclose(
            r["denom_mixed"], (r["denom"] * numpy.exp(shift)).real, rtol=1e-10, atol=1e-12
        )
        assert abs(r["denom"].imag) < 1e-12 * abs(r["denom"])


@pytest.mark.unit
def test_mixed_estimator_guide_overlap_consistent_and_positive():
    # Interacting walk under a real fixed guide: the guide bra reproduces the cached overlap of
    # every live walker, and the mixed denominator is the positive total weight.
    ham, hop, trial, walkers, prop, mpi = _mixed_objects(4.0, 1.0, 1.0, nwalkers=32)
    driver = ThermofieldHubbardAFQMC(
        ham,
        hop,
        trial,
        walkers,
        prop,
        mpi,
        2,
        2,
        nblocks=3,
        wrap_stride=None,
        shell_distances_squared=(1, 2),
        seed=11,
    )
    driver.run(verbose=False)
    for r in driver.results:
        assert r["mixed_overlap_dev"] < 1e-8
        assert r["denom_mixed"] > 0
        e_mixed = (r["numer_mixed"][0] / r["denom_mixed"]).real
        assert numpy.isfinite(e_mixed) and -3.0 < e_mixed < 0.0
    # The mixed denominator of the last block is the total live weight of the final population.
    last = driver.results[-1]
    numpy.testing.assert_allclose(
        last["denom_mixed"], walkers.weight[walkers.weight > 0].sum(), rtol=1e-10
    )


if __name__ == "__main__":
    test_discrete_driver_smoke(False)
    test_discrete_trace_branching_preserves_paths_and_endpoint_integrands("fixed", True)
    test_discrete_trace_branching_keeps_negative_physical_trace()
    test_hirsch_identity_all_local_fock_states(4.0, 0.02)
    test_site_ratios_and_updates_against_noncommuting_dense_determinants()
    test_scalar_batched_paths_and_caches("fixed", True)
    test_slice_weights_against_independent_dense_conditional_sampler("fixed")
    test_noninteracting_exact_trace("fixed", False)
    test_history_buffer_and_wrap_reconstruct_actual_discrete_path()
    test_two_slice_partition_and_energy_against_fock_space()
    test_beta80_prescribed_field_wrap_against_spectral_reference()
    test_extreme_coherent_path_refuses_incomplete_wrap()
    test_discrete_driver_metadata_uses_physical_interaction()
    test_completion_importance_option_validation()
    test_default_importance_guide_leaves_sampler_unchanged()
    test_completion_importance_noninteracting_exact(True)
    test_completion_importance_endpoint_correction_is_a_sign(True)
    test_completion_importance_matches_guide_branching_statistically()
    test_tempered_exponent_validation()
    test_tempered_exponent_zero_is_guide_branching_bitwise(True)
    test_tempered_exponent_one_is_trace_branching_default()
    test_tempered_noninteracting_exact(0.5)
    test_tempered_endpoint_weight_identity_without_resampling()
    test_tempered_matches_guide_branching_statistically()
    test_mixed_estimator_exact_for_noninteracting_exact_trial()
    test_measure_mixed_false_keeps_endpoint_and_overlap_check()
    test_mixed_estimator_reduces_to_physical_with_identity_guide()
    test_mixed_estimator_guide_overlap_consistent_and_positive()
    if comm.size > 1:
        test_discrete_mpi_population_history_and_reduction(False)
    if comm.size == 2:
        test_completion_importance_caches_survive_mpi_population_control()
