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
def test_discrete_driver_smoke():
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
    assert numpy.isin(walkers.field_hist[walkers.weight != 0], [-1.0, 1.0]).all()
    assert numpy.isfinite(driver.one_rdm_summary()["mean"]).all()
    assert numpy.isfinite(driver.results[0]["numer_endpoint_magnitude"]).all()


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
def test_noninteracting_exact_trace(mode):
    ham, hop, trial, walkers, prop, mpi = _build_objects(U=0.0, guide_mode=mode)
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
def test_discrete_mpi_population_history_and_reduction():
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
    test_discrete_driver_smoke()
    test_hirsch_identity_all_local_fock_states(4.0, 0.02)
    test_site_ratios_and_updates_against_noncommuting_dense_determinants()
    test_scalar_batched_paths_and_caches("fixed", True)
    test_slice_weights_against_independent_dense_conditional_sampler("fixed")
    test_noninteracting_exact_trace("fixed")
    test_history_buffer_and_wrap_reconstruct_actual_discrete_path()
    test_two_slice_partition_and_energy_against_fock_space()
    test_beta80_prescribed_field_wrap_against_spectral_reference()
    test_extreme_coherent_path_refuses_incomplete_wrap()
    test_discrete_driver_metadata_uses_physical_interaction()
    test_mixed_estimator_exact_for_noninteracting_exact_trial()
    test_measure_mixed_false_keeps_endpoint_and_overlap_check()
    test_mixed_estimator_reduces_to_physical_with_identity_guide()
    test_mixed_estimator_guide_overlap_consistent_and_positive()
    if comm.size > 1:
        test_discrete_mpi_population_history_and_reduction()
