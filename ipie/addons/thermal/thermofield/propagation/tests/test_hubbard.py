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

from types import SimpleNamespace

import numpy
import pytest
from scipy.linalg import eigh, expm

from ipie.addons.thermal.thermofield.estimators.cyclic_average import cyclic_average_hubbard
from ipie.addons.thermal.thermofield.estimators.hubbard import HubbardSpinObservables
from ipie.addons.thermal.thermofield.hamiltonians.hubbard import (
    build_hubbard_hamiltonian,
    hopping_matrix,
)
from ipie.addons.thermal.thermofield.propagation.hubbard import (
    ThermofieldHubbardCP,
    update_trace_weight,
)
from ipie.addons.thermal.thermofield.qmc.hubbard_afqmc import propagate_population_cp
from ipie.addons.thermal.thermofield.trial.hubbard import (
    fit_hubbard_density_trial,
    fit_hubbard_trial,
    rhf_k_trial,
    thermal_ip_scf,
    thermal_uhf_scf,
    ThermofieldCPTrial,
)
from ipie.addons.thermal.thermofield.utils.testing import creation_operators
from ipie.addons.thermal.thermofield.walkers.pop_controller import ThermofieldPopController
from ipie.addons.thermal.thermofield.walkers.uhf_walkers import ThermofieldWalkers
from ipie.utils.mpi import MPIHandler
from ipie.walkers.pop_controller import get_buffer, set_buffer


def build_test_case(
    U=4.0,
    beta=0.4,
    dt=0.02,
    batched=True,
    cache=True,
    mode="fixed",
    trace_branching=False,
    energy_shift=0.0,
):
    # Periodic 2x2 lattice at mu = 0.3 with a noninteracting guide at 3.6 electrons.
    ham, hop, v = build_hubbard_hamiltonian(2, 2, U, 0.3)
    k, _, _ = rhf_k_trial(hop, v, beta, 3.6)
    trial = ThermofieldCPTrial(ham, beta, k_trial=k, guide_time=beta)
    handler = MPIHandler()
    walkers = ThermofieldWalkers(trial, 4, 4, mpi_handler=handler)
    prop = ThermofieldHubbardCP(
        ham.H1.real + 0.5 * U * numpy.eye(4),
        U,
        dt,
        beta,
        record_fields=True,
        guide_mode=mode,
        fixed_guide_time=beta,
        batched=batched,
        cache_qinv=cache,
        trace_branching=trace_branching,
        energy_shift=energy_shift,
    )
    afqmc = SimpleNamespace(
        mpi_handler=handler, params=SimpleNamespace(num_stblz=5, pop_control_freq=10)
    )
    pc = ThermofieldPopController(4, round(beta / dt), handler, "pair_branch", verbose=False)
    return trial, walkers, prop, afqmc, pc


@pytest.mark.unit
def test_hubbard_cpu_smoke():
    trial, walkers, prop, afqmc, pc = build_test_case()
    numpy.random.seed(73)
    propagate_population_cp(afqmc, walkers, trial, prop, 20, pc)
    assert numpy.isfinite(walkers.Qmat).all()
    assert numpy.isfinite(walkers.weight).all()
    assert walkers.weight.sum() > 0
    assert "field_hist" in walkers.buff_names
    observed = []
    dev, ld0 = cyclic_average_hubbard(
        walkers,
        trial,
        prop,
        20,
        5,
        lambda iw, G, cut: observed.append((iw, cut, G.copy())),
    )
    assert len(observed) == 16
    assert dev < 1e-10
    assert numpy.isfinite(ld0).all()
    assert all(numpy.isfinite(item[2]).all() for item in observed)


@pytest.mark.unit
def test_all_dead_population_fails_before_controller():
    trial, walkers, prop, afqmc, pc = build_test_case()
    prop.propagate_walkers = lambda W, T, step: W.weight.fill(0.0)
    called = []
    pc.pop_control = lambda *args: called.append(True)
    with pytest.raises(RuntimeError, match="all dead"):
        propagate_population_cp(afqmc, walkers, trial, prop, 20, pc)
    assert not called


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["fixed", "completion"])
@pytest.mark.parametrize("cache", [False, True])
def test_cp_scalar_batched_same_random_fields(mode, cache):
    trial, scalar, sp, afqmc, _ = build_test_case(batched=False, mode=mode)
    _, batched, bp, _, _ = build_test_case(batched=True, cache=cache, mode=mode)
    # Disable branching here to isolate propagation/cache parity.
    afqmc.params.pop_control_freq = 1000
    dummy_pc = SimpleNamespace(total_weight=4.0)
    numpy.random.seed(73)
    propagate_population_cp(afqmc, scalar, trial, sp, 20, dummy_pc)
    numpy.random.seed(73)
    propagate_population_cp(afqmc, batched, trial, bp, 20, dummy_pc)
    numpy.testing.assert_allclose(batched.field_hist, scalar.field_hist, atol=2e-13, rtol=1e-13)
    for name in ("weight", "Qmat", "log_d", "Tmat", "Ga", "Gb"):
        numpy.testing.assert_allclose(
            getattr(batched, name), getattr(scalar, name), atol=3e-11, rtol=3e-11
        )
    assert bp.kill_count == sp.kill_count


@pytest.mark.unit
def test_noninteracting_trace_matches_fermi_density():
    trial, walkers, prop, afqmc, pc = build_test_case(U=0.0)
    numpy.random.seed(73)
    propagate_population_cp(afqmc, walkers, trial, prop, 20, pc)
    _, particle_G = trial.calc_log_overlap_and_greens_function_scaled_batched(
        walkers.Qmat, walkers.log_d, walkers.Tmat, 0.0
    )
    hop = hopping_matrix(2, 2) + 0.3 * numpy.eye(4)
    exact = numpy.eye(4) - numpy.linalg.inv(numpy.eye(4) + expm(-0.4 * hop))
    numpy.testing.assert_allclose(
        particle_G, numpy.broadcast_to(exact, particle_G.shape), atol=1e-12
    )


@pytest.mark.unit
def test_history_clone_and_fullpath_against_dense_cyclic_product():
    trial, walkers, prop, afqmc, pc = build_test_case()
    numpy.random.seed(73)
    propagate_population_cp(afqmc, walkers, trial, prop, 20, pc)
    # The same buffer API is used by cross-rank population control.
    walkers.field_hist[0, 0] += 0.04
    set_buffer(walkers, 1, get_buffer(walkers, 0).copy())
    numpy.testing.assert_array_equal(walkers.field_hist[0], walkers.field_hist[1])
    prop.seed_qinv_cache(walkers)
    numpy.testing.assert_allclose(
        walkers.Qinv[1] @ walkers.Qmat[1],
        numpy.broadcast_to(numpy.eye(4), (2, 4, 4)),
        atol=1e-12,
    )
    before = walkers.Qmat.copy()

    def compare(iw, G, cut):
        fields = walkers.field_hist[iw]
        matrices = []
        for x in fields:
            d = numpy.exp(prop.gamma * x)
            matrices.append(
                numpy.array(
                    [
                        prop.BK2[0] @ (d[:, None] * prop.BK2[0]),
                        prop.BK2[1] @ ((1 / d)[:, None] * prop.BK2[1]),
                    ]
                )
            )
        prefix = numpy.broadcast_to(numpy.eye(4), (2, 4, 4)).copy()
        suffix = prefix.copy()
        for B in matrices[:cut]:
            prefix = B @ prefix
        for B in matrices[cut:]:
            suffix = B @ suffix
        expected = numpy.eye(4) - numpy.linalg.inv(numpy.eye(4) + prefix @ suffix)
        numpy.testing.assert_allclose(G, expected, atol=1e-11)

    dev, _ = cyclic_average_hubbard(walkers, trial, prop, 20, 3, compare)
    assert dev < 1e-10
    numpy.testing.assert_array_equal(walkers.Qmat, before)


@pytest.mark.unit
def test_lattice_and_trial_filling():
    hop = hopping_matrix(8, 8)
    assert numpy.count_nonzero(hop) == 64 * 4
    assert numpy.all(hop.sum(axis=1) == -4)
    assert hop[0, 7] == hop[0, 56] == -1
    assert numpy.count_nonzero(hopping_matrix(8, 8, False, False)) == 4 * 8 * 7
    v = numpy.zeros((2, 64))
    k, dens, mu = rhf_k_trial(hop, v, 1 / 0.23, 0.85 * 64)
    assert abs(dens.sum() / 64 - 0.85) < 1e-12
    assert mu > 0
    numpy.testing.assert_allclose(k[0], k[1])
    hop2 = hopping_matrix(2, 2)
    _, dens2, _ = thermal_ip_scf(hop2, numpy.zeros((2, 4)), 0.5, 3.5, 2.0)
    assert abs(dens2.sum() - 3.5) < 1e-12


@pytest.mark.unit
@pytest.mark.parametrize(
    "argument,value", [("U", -1), ("U", numpy.nan), ("timestep", 0), ("beta", numpy.inf)]
)
def test_invalid_propagation_parameters(argument, value):
    kwargs = dict(h_kin=numpy.zeros((2, 2, 2)), U=4, timestep=0.01, beta=2)
    kwargs[argument] = value
    with pytest.raises(ValueError):
        ThermofieldHubbardCP(**kwargs)


@pytest.mark.unit
def test_trial_fit_recovers_noninteracting_temperature_and_preserves_operator():
    hop = hopping_matrix(4, 4)
    v = numpy.zeros((2, 16))
    beta_eff, physical_beta = 0.8, 2.0
    k, density, _ = rhf_k_trial(hop, v, beta_eff, 13.6)
    exact_G = numpy.eye(16) - numpy.linalg.inv(numpy.eye(16) + expm(-beta_eff * k))
    obs = HubbardSpinObservables(hop, 0.0, 4, 4)
    target = obs.finalise(obs.measure(exact_G))["czz"]
    fit = fit_hubbard_trial(hop, v, physical_beta, 13.6, 4, 4, target, [0.0], [0.4, 0.8, 1.6])
    assert fit["beta_eff"] == beta_eff
    assert fit["u_eff"] == 0.0
    assert fit["objective"] < 1e-25
    numpy.testing.assert_allclose(
        expm(-physical_beta * fit["k_trial"]), expm(-beta_eff * k), atol=1e-12
    )
    numpy.testing.assert_allclose(fit["densities"], density)
    assert len(fit["candidates"]) == 3


@pytest.mark.unit
def test_trial_fit_records_scf_failures_and_rejects_all_failed():
    hop = hopping_matrix(4, 4)
    v = numpy.zeros((2, 16))
    args = (hop, v, 2.0, 13.6, 4, 4, {1: -0.1})
    # RHF starts at its uniform fixed point; UHF's seeded state needs >1 iteration.
    fit = fit_hubbard_trial(*args, [0.0, 8.0], [1.0], scf_options={"max_iter": 1})
    assert fit["candidates"][0]["status"] == "ok"
    assert fit["candidates"][1]["status"] == "failed"
    with pytest.raises(RuntimeError, match="All thermal-HF"):
        fit_hubbard_trial(*args, [8.0], [1.0], scf_options={"max_iter": 1})


@pytest.mark.unit
@pytest.mark.parametrize("batched", [False, True])
def test_trace_branching_unbranched_telescoping(batched):
    # The same proposed paths have equal signed physical contributions. Direct
    # per-slice calls omit branching and the population driver's broad
    # numerical weight guard, so the actual telescoping identity is tested.
    beta, dt, nslices = 2.0, 0.02, 100
    paths = []
    for trace_branching in (False, True):
        trial, W, prop, _, _ = build_test_case(beta=beta, dt=dt, batched=batched)
        prop.trace_branching = trace_branching
        W.field_hist = numpy.zeros((W.nwalkers, nslices, prop.nfields))
        prop.init_walker_cache(W, trial)
        initial_guide = W.log_ovlp.copy()
        initial_trace = trial.calc_log_trace_batched(W.Qmat, W.log_d, W.Tmat)
        if batched:
            prop.seed_qinv_cache(W)
        numpy.random.seed(73)
        for step in range(nslices):
            prop.propagate_walkers(W, trial, step)
            if (step + 1) % 5 == 0:
                W.stabilize()
                if batched:
                    prop.refresh_qinv_after_stabilize(W)
                else:
                    prop.refresh_cache_after_stabilize(W, trial, step)
        trace_log, trace_G = trial.calc_log_overlap_and_greens_function_scaled_batched(
            W.Qmat, W.log_d, W.Tmat, 0.0
        )
        assert numpy.all(W.weight > 0)
        assert prop.trace_invalid_count == 0
        assert prop.trace_log_weight_bound_count == 0
        paths.append((W, initial_guide, initial_trace, trace_log, trace_G))
    old, new = paths[0][0], paths[1][0]
    numpy.testing.assert_array_equal(old.field_hist, new.field_hist)
    numpy.testing.assert_array_equal(old.Qmat, new.Qmat)
    numpy.testing.assert_array_equal(old.Ga, new.Ga)
    numpy.testing.assert_array_equal(old.Gb, new.Gb)
    logQ = paths[0][3] - old.log_ovlp
    # Here w_trace / w_guide = |T_L / S_L| S_0 / T_0.
    expected = logQ.real + (paths[0][1] - paths[0][2]).real
    numpy.testing.assert_allclose(
        numpy.log(new.weight) - numpy.log(old.weight), expected, atol=2e-10, rtol=1e-11
    )
    old_logs = numpy.log(old.weight) + logQ.real + paths[0][1].real
    new_logs = numpy.log(new.weight) + paths[1][2].real
    shift = max(old_logs.max(), new_logs.max())
    signs = numpy.exp(1j * logQ.imag)
    old_physical = signs * numpy.exp(old_logs - shift)
    new_physical = signs * numpy.exp(new_logs - shift)
    numpy.testing.assert_allclose(old_physical, new_physical, atol=2e-11)
    obs = HubbardSpinObservables(hopping_matrix(2, 2), 4.0, 2, 2, (1, 2))
    values = numpy.asarray([obs.measure(G) for G in paths[0][4]])
    # The physical paths must differ.
    assert numpy.std(values[:, 1].real) > 1e-6
    numpy.testing.assert_allclose(
        old_physical @ values / old_physical.sum(),
        new_physical @ values / new_physical.sum(),
        atol=2e-11,
    )
    numpy.testing.assert_allclose(new.log_tr, paths[1][3], atol=2e-11)


@pytest.mark.unit
@pytest.mark.parametrize("batched", [False, True])
def test_trace_branching_allows_negative_trace_with_positive_guide(batched):
    # A trace sign crossing must not introduce a second CP boundary. The
    # positive-determinant real Gaussian starts with up-block eigenvalues
    # (-1/2, -1/2, 1, 1). Multiplying by the positive B = diag(4, 1, 1, 1)
    # changes the sign of the trace determinant while the guide Lambda = 0.1 I
    # remains positive, which isolates the boundary decision exactly.
    ham, _, _ = build_hubbard_hamiltonian(2, 2, 0.0, 0.0)
    beta, dt = 1.0, 0.02
    trial = ThermofieldCPTrial(
        ham, beta, k_trial=numpy.tile(numpy.log(10.0) * numpy.eye(4), (2, 1, 1)), guide_time=beta
    )
    W = ThermofieldWalkers(trial, 4, 1)
    W.Qmat[0, 0] = numpy.diag([-0.5, -0.5, 1.0, 1.0])
    kinetic = numpy.zeros((2, 4, 4))
    kinetic[0, 0, 0] = -numpy.log(4.0) / dt
    prop = ThermofieldHubbardCP(
        kinetic,
        0.0,
        dt,
        beta,
        guide_mode="fixed",
        fixed_guide_time=beta,
        batched=batched,
        trace_branching=True,
    )
    prop.init_walker_cache(W, trial)
    assert numpy.cos(W.log_tr[0].imag) > 0
    if batched:
        prop.seed_qinv_cache(W)
    numpy.random.seed(1)
    prop.propagate_walkers(W, trial, 0)
    assert prop.kill_count == 0
    assert prop.trace_invalid_count == 0
    assert numpy.cos(W.log_ovlp[0].imag) > 0
    assert numpy.cos(W.log_tr[0].imag) < 0
    numpy.testing.assert_allclose(W.weight, [2.0], atol=1e-12)
    assert numpy.exp(1j * (W.log_tr[0] - W.log_ovlp[0]).imag).real < 0


@pytest.mark.unit
def test_trace_cache_reset_and_population_buffer_clone():
    trial, W, prop, afqmc, pc = build_test_case()
    prop.trace_branching = True
    W.log_tr[:] = 123.0 + 2.0j
    W.reset(trial, compute_cache=False)
    prop.init_walker_cache(W, trial)
    numpy.testing.assert_allclose(W.log_tr, 2 * W.nbasis * numpy.log(2.0), atol=1e-12)
    numpy.random.seed(73)
    propagate_population_cp(afqmc, W, trial, prop, 20, pc)
    fresh = trial.calc_log_trace_batched(W.Qmat, W.log_d, W.Tmat)
    numpy.testing.assert_allclose(W.log_tr, fresh, atol=1e-11)
    assert "log_tr" in W.buff_names
    W.log_tr[1] = -999.0 + 1.0j
    set_buffer(W, 1, get_buffer(W, 0).copy())
    assert W.log_tr[1] == W.log_tr[0]
    numpy.testing.assert_array_equal(W.field_hist[1], W.field_hist[0])
    numpy.testing.assert_allclose(
        W.log_tr[1],
        trial.calc_log_trace_batched(W.Qmat[1:2], W.log_d[1:2], W.Tmat[1:2])[0],
        atol=1e-11,
    )


@pytest.mark.unit
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("bad_trace", [-numpy.inf, numpy.nan])
def test_trace_branching_zero_or_invalid_trace_kills_numerical_path(batched, bad_trace):
    trial, W, prop, _, _ = build_test_case(batched=batched)
    prop.trace_branching = True
    prop.record_fields = False
    prop.init_walker_cache(W, trial)
    trial.calc_log_trace_batched = lambda Q, d, T: numpy.full(len(Q), bad_trace, dtype=complex)
    if batched:
        prop.seed_qinv_cache(W)
    numpy.random.seed(73)
    prop.propagate_walkers(W, trial, 0)
    numpy.testing.assert_array_equal(W.weight, numpy.zeros(W.nwalkers))
    assert prop.trace_invalid_count == W.nwalkers
    assert prop.kill_count == 0


@pytest.mark.unit
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("trace_branching", [False, True])
def test_common_energy_reference_preserves_paths_and_normalized_weights(batched, trace_branching):
    paths = []
    for reference in (0.0, -5.0):
        trial, W, prop, afqmc, pc = build_test_case(
            beta=0.04,
            dt=0.02,
            batched=batched,
            trace_branching=trace_branching,
            energy_shift=reference,
        )
        numpy.random.seed(73)
        propagate_population_cp(afqmc, W, trial, prop, 2, pc)
        assert prop.log_weight_bound_count == prop.population_weight_bound_count == 0
        paths.append(W)
    numpy.testing.assert_array_equal(paths[0].Qmat, paths[1].Qmat)
    numpy.testing.assert_array_equal(paths[0].field_hist, paths[1].field_hist)
    numpy.testing.assert_allclose(
        paths[1].weight, paths[0].weight * numpy.exp(-5.0 * 0.04), atol=1e-12, rtol=1e-12
    )
    numpy.testing.assert_allclose(
        paths[1].weight / paths[1].weight.sum(),
        paths[0].weight / paths[0].weight.sum(),
        atol=1e-13,
    )


@pytest.mark.unit
@pytest.mark.parametrize("batched", [False, True])
def test_energy_reference_prevents_extensive_population_guard_hits(batched):
    # A soluble occupied-level model exposes the broad cap before branching.
    # For eight spin orbitals at energy -100 the trace grows as
    # (1 + exp(100 theta))^8; a reference of -800 removes that deterministic growth.
    ham, _, _ = build_hubbard_hamiltonian(2, 2, 0.0, -100.0, t=0.0)
    beta, dt = 0.1, 0.01
    trial = ThermofieldCPTrial(ham, beta, k_trial=numpy.zeros((2, 4, 4)), guide_time=beta)
    handler = MPIHandler()
    afqmc = SimpleNamespace(
        mpi_handler=handler, params=SimpleNamespace(num_stblz=5, pop_control_freq=1000)
    )
    hits = []
    for reference in (0.0, -800.0):
        W = ThermofieldWalkers(trial, 4, 4, mpi_handler=handler)
        prop = ThermofieldHubbardCP(
            ham.H1,
            0.0,
            dt,
            beta,
            guide_mode="fixed",
            fixed_guide_time=beta,
            batched=batched,
            energy_shift=reference,
        )
        numpy.random.seed(1)
        propagate_population_cp(
            afqmc, W, trial, prop, 10, SimpleNamespace(total_weight=4.0), weight_cap=0.0
        )
        hits.append(prop.population_weight_bound_count)
        assert prop.log_weight_bound_count == 0
        if reference != 0.0:
            exact = numpy.exp(8 * numpy.log1p(numpy.exp(-100 * beta)) - 8 * numpy.log(2.0))
            numpy.testing.assert_allclose(W.weight, exact, atol=1e-12)
    assert hits[0] > 0
    assert hits[1] == 0


@pytest.mark.unit
def test_trace_weight_update_compensates_extreme_ratios_and_preserves_signs():
    old = numpy.array(
        [numpy.exp(-80.0), numpy.exp(80.0), -numpy.exp(-700.0), -numpy.exp(700.0), 0.0]
    )
    increment = numpy.array([80.0, -80.0, 740.0, -740.0, numpy.inf])
    signs = numpy.array([1.0, -1.0, 1.0, -1.0, 1.0])
    actual, failures = update_trace_weight(old, increment, signs)
    numpy.testing.assert_allclose(
        actual, [1.0, -1.0, -numpy.exp(40.0), numpy.exp(-40.0), 0.0], rtol=2e-14, atol=0.0
    )
    assert failures == 0


@pytest.mark.unit
def test_trace_weight_update_ordinary_arithmetic_and_true_representation_failures():
    old = numpy.array([0.3, -0.7, 1e-100, 1e100])
    increment = numpy.array([-60.0, 60.0, 20.0, -20.0])
    signs = numpy.array([1.0, -1.0, -1.0, 1.0])
    actual, failures = update_trace_weight(old, increment, signs)
    numpy.testing.assert_array_equal(actual, old * (signs * numpy.exp(increment)))
    assert failures == 0
    actual, failures = update_trace_weight(
        [1.0, 1.0, 1.0, 0.0], [800.0, -800.0, numpy.nan, numpy.inf]
    )
    numpy.testing.assert_array_equal(actual, numpy.zeros(4))
    assert failures == 3
    # A small increment can also exceed the representable total weight: the
    # first product overflows, the second is subnormal but representable.
    actual, failures = update_trace_weight([1e300, 1e-300], [50.0, -50.0])
    assert failures == 1
    assert actual[0] == 0.0 and actual[1] > 0.0


@pytest.mark.unit
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("increment", [-80.0, 80.0])
def test_trace_propagation_large_compensated_increment_has_no_limiter(batched, increment):
    trial, W, prop, _, _ = build_test_case(U=0.0, batched=batched, trace_branching=True)
    prop.record_fields = False
    prop.init_walker_cache(W, trial)
    old_trace = W.log_tr[0]
    W.weight[:] = numpy.exp(-increment)
    W.weight[-1] = 0.0
    dead_q = W.Qmat[-1].copy()
    # A finite determinant ratio is imposed to isolate its weight arithmetic;
    # U = 0 makes the field proposal correction identically zero.
    trial.calc_log_trace_batched = lambda Q, d, T: numpy.full(len(Q), old_trace + increment)
    if batched:
        prop.seed_qinv_cache(W)
    numpy.random.seed(73)
    prop.propagate_walkers(W, trial, 0)
    numpy.testing.assert_allclose(W.weight, [1.0, 1.0, 1.0, 0.0], atol=2e-14)
    numpy.testing.assert_array_equal(W.Qmat[-1], dead_q)
    assert prop.log_weight_bound_count == 0
    assert prop.trace_log_weight_bound_count == 0
    assert prop.trace_invalid_count == 0


@pytest.mark.unit
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("increment", [-800.0, 800.0])
def test_trace_propagation_true_unrepresentable_total_is_counted(batched, increment):
    trial, W, prop, _, _ = build_test_case(U=0.0, batched=batched, trace_branching=True)
    prop.record_fields = False
    prop.init_walker_cache(W, trial)
    old_trace = W.log_tr[0]
    W.weight[-1] = 0.0
    trial.calc_log_trace_batched = lambda Q, d, T: numpy.full(len(Q), old_trace + increment)
    if batched:
        prop.seed_qinv_cache(W)
    numpy.random.seed(73)
    prop.propagate_walkers(W, trial, 0)
    numpy.testing.assert_array_equal(W.weight, numpy.zeros(4))
    assert prop.log_weight_bound_count == 3
    assert prop.trace_log_weight_bound_count == 3
    assert prop.trace_invalid_count == 0


@pytest.mark.unit
@pytest.mark.parametrize("cache", [False, True])
def test_trace_force_bias_matches_closed_trace_derivative(cache):
    trial, walkers, prop, afqmc, pc = build_test_case(trace_branching=True, cache=cache)
    numpy.random.seed(73)
    propagate_population_cp(afqmc, walkers, trial, prop, 20, pc)
    prop.force_bias_mode = "trace"
    bias = prop.trace_force_bias(walkers, trial, numpy.arange(walkers.nwalkers))
    step = 1e-5
    expected = numpy.empty_like(bias)
    for iw in range(walkers.nwalkers):
        for site in range(walkers.nbasis):
            logs = []
            for sign in (-1, 1):
                q = walkers.Qmat[iw].copy()
                q[0, site] *= numpy.exp(sign * step * prop.gamma)
                q[1, site] *= numpy.exp(-sign * step * prop.gamma)
                logs.append(
                    trial.calc_log_trace_batched(
                        q[None], walkers.log_d[iw : iw + 1], walkers.Tmat[iw : iw + 1]
                    )[0].real
                )
            expected[iw, site] = (logs[1] - logs[0]) / (2 * step)
    numpy.testing.assert_allclose(bias, numpy.clip(expected, -1, 1), atol=2e-9, rtol=2e-9)


@pytest.mark.unit
@pytest.mark.parametrize("cache", [False, True])
def test_trace_force_bias_scalar_batched_parity(cache):
    trial, scalar, sp, afqmc, _ = build_test_case(batched=False, cache=False, trace_branching=True)
    _, batched, bp, _, _ = build_test_case(batched=True, cache=cache, trace_branching=True)
    sp.force_bias_mode = bp.force_bias_mode = "trace"
    afqmc.params.pop_control_freq = 1000
    dummy_pc = SimpleNamespace(total_weight=4.0)
    numpy.random.seed(73)
    propagate_population_cp(afqmc, scalar, trial, sp, 20, dummy_pc)
    numpy.random.seed(73)
    propagate_population_cp(afqmc, batched, trial, bp, 20, dummy_pc)
    for name in ("weight", "Qmat", "log_d", "Tmat", "Ga", "Gb", "field_hist"):
        numpy.testing.assert_allclose(
            getattr(batched, name), getattr(scalar, name), atol=5e-11, rtol=5e-11
        )


@pytest.mark.unit
def test_trace_force_bias_leaves_noninteracting_propagation_unchanged():
    outcomes = []
    for mode in ("guide", "trace"):
        trial, walkers, prop, afqmc, pc = build_test_case(U=0.0, trace_branching=True)
        prop.force_bias_mode = mode
        numpy.random.seed(73)
        propagate_population_cp(afqmc, walkers, trial, prop, 20, pc)
        outcomes.append((walkers.Qmat.copy(), walkers.weight.copy()))
    for first, second in zip(*outcomes):
        numpy.testing.assert_array_equal(first, second)


@pytest.mark.driver
@pytest.mark.parametrize(
    "population,blocks,retain_normalization,force_mode",
    [(64, 96, False, "trace"), (4, 384, True, "trace"), (4, 384, True, "guide")],
)
def test_trace_force_bias_sampling_against_interacting_fock_space(
    population, blocks, retain_normalization, force_mode
):
    trial, _, prop, afqmc, _ = build_test_case(U=4.0, beta=1.0, dt=0.025, trace_branching=True)
    prop.force_bias_mode = force_mode
    prop.energy_shift = -8.0
    handler = afqmc.mpi_handler
    walkers = ThermofieldWalkers(trial, 4, population, mpi_handler=handler)
    pc = ThermofieldPopController(population, 40, handler, "pair_branch")
    afqmc.params.pop_control_freq = 1
    hop = hopping_matrix(2, 2)
    obs = HubbardSpinObservables(hop, 4.0, 2, 2, (1, 2))
    cre = creation_operators(8)
    number = [c @ c.T for c in cre]
    kinetic = sum(
        hop[i, j] * (cre[4 * s + i] @ cre[4 * s + j].T)
        for s in range(2)
        for i in range(4)
        for j in range(4)
    )
    interaction = 4 * sum(number[i] @ number[4 + i] for i in range(4))
    energy_op = kinetic + interaction
    hamiltonian = energy_op + (0.3 - 2) * sum(number)
    energies, vectors = eigh(hamiltonian)
    probability = numpy.exp(-(energies - energies.min()))
    probability /= probability.sum()
    rho = (vectors * probability) @ vectors.T
    exact_energy = numpy.trace(rho @ energy_op) / 4
    spin = [number[i] - number[4 + i] for i in range(4)]
    ii, jj = obs.pairs[1]
    exact_spin = sum(numpy.trace(rho @ (spin[i] @ spin[j])) for i, j in zip(ii, jj)) / len(ii)
    numerator, denominator, log_shifts = [], [], []
    numpy.random.seed(1031 + handler.comm.rank)
    for _ in range(blocks):
        propagate_population_cp(afqmc, walkers, trial, prop, 40, pc, weight_cap=0.0)
        logtrace, green = trial.calc_log_overlap_and_greens_function_scaled_batched(
            walkers.Qmat, walkers.log_d, walkers.Tmat, 0.0
        )
        weights = walkers.weight * numpy.cos((logtrace - walkers.log_ovlp).imag)
        raw = numpy.array([obs.measure(g) for g in green])
        numerator.append(handler.comm.allreduce(weights @ raw))
        denominator.append(handler.comm.allreduce(weights.sum()))
        log_shifts.append(afqmc.population_log_normalization if retain_normalization else 0.0)
    result = obs.summary(numerator, denominator, log_shifts)
    for name, exact in (("energy_per_site", exact_energy),):
        assert abs(result["mean"][name] - exact) < 5 * result["error"][name] + 0.002
    assert abs(result["mean"]["czz"]["1"] - exact_spin) < 5 * result["error"]["czz"]["1"] + 0.002
    assert prop.log_weight_bound_count == prop.population_weight_bound_count == 0


@pytest.mark.unit
def test_multistart_thermal_uhf_density_and_stationarity():
    hop = hopping_matrix(4, 4)
    v = numpy.zeros((2, 16))
    k, density, mu, info = thermal_uhf_scf(
        hop, v, 2.0, 16, 4.0, 4, 4, scf_options=dict(mix=0.3, max_iter=1500)
    )
    eps, coeff = numpy.linalg.eigh(k)
    occ = 0.5 * (1 - numpy.tanh(eps))
    expected = numpy.einsum("sik,sk,sik->si", coeff, occ, coeff.conj()).real
    numpy.testing.assert_allclose(density, expected, atol=1e-11)
    numpy.testing.assert_allclose(density.sum(), 16, atol=1e-11)
    for spin in range(2):
        numpy.testing.assert_allclose(
            k[spin], hop + numpy.diag(4 * (density[1 - spin] - 0.5) + mu), atol=1e-9
        )
    assert info["max_local_magnetization"] > 0.1
    assert info["free_energy"] == min(
        c["free_energy"] for c in info["candidates"] if c["status"] == "ok"
    )
    assert len(info["candidates"]) == 4


@pytest.mark.unit
def test_uhf_only_trials_reject_free_electron_candidate():
    hop = hopping_matrix(4, 4)
    v = numpy.zeros((2, 16))
    with pytest.raises(ValueError, match="positive U_eff"):
        thermal_uhf_scf(hop, v, 2, 16, 0, 4, 4)
    with pytest.raises(ValueError, match="strictly positive"):
        fit_hubbard_trial(hop, v, 2, 16, 4, 4, {1: -0.2}, [0, 4], [2], uhf_options={})


@pytest.mark.unit
def test_uhf_feedback_fit_recovers_thermal_density_operator():
    hop = hopping_matrix(4, 4)
    v = numpy.zeros((2, 16))
    options = dict(mix=0.3, max_iter=1500)
    k, density, mu, info = thermal_uhf_scf(
        hop, v, 2, 16, 4, 4, 4, random_starts=0, scf_options=options
    )
    eps, coeff = numpy.linalg.eigh(k)
    occ = 0.5 * (1 - numpy.tanh(eps))
    green = numpy.einsum("sik,sk,sjk->sij", coeff, occ, coeff.conj())
    obs = HubbardSpinObservables(hop, 4, 4, 4)
    target = obs.finalise(0.5 * (obs.measure(green) + obs.measure(green[::-1])))["czz"]
    fit = fit_hubbard_trial(
        hop,
        v,
        3,
        16,
        4,
        4,
        target,
        [2, 4],
        [0.5, 2],
        spin_flip_average=True,
        scf_options=options,
        uhf_options=dict(random_starts=0),
    )
    assert fit["u_eff"] == 4
    assert fit["beta_eff"] == 2
    assert fit["objective"] < 1e-20
    numpy.testing.assert_allclose(expm(-3 * fit["k_trial"]), expm(-2 * k), rtol=1e-10, atol=1e-10)


@pytest.mark.unit
def test_figure5_physical_geometry_has_two_periodic_axes_and_zero_pinning():
    ham, hop, pin = build_hubbard_hamiltonian(12, 12, 8, 0, pbc_x=True, pbc_y=True, pin_h=0)
    numpy.testing.assert_array_equal(pin, numpy.zeros((2, 144)))
    numpy.testing.assert_array_equal(numpy.sum(hop != 0, axis=1), numpy.full(144, 4))
    assert hop[0, 11] == hop[0, 132] == -1
    numpy.testing.assert_array_equal(ham.H1[0], hop - 4 * numpy.eye(144))
    numpy.testing.assert_array_equal(ham.H1[0], ham.H1[1])
    obs = HubbardSpinObservables(hop, 8, 12, 12)
    assert obs.metadata["shell_pair_counts"] == {"1": 576, "2": 576, "4": 576, "5": 1152}


@pytest.mark.unit
def test_density_feedback_recovers_known_thermal_uhf_fixed_point():
    hop = hopping_matrix(4, 4)
    pinning = numpy.zeros((2, 16))
    reference, densities, _, _ = thermal_uhf_scf(hop, pinning, 2.0, 16, 4.0, 4, 4, random_starts=0)
    fit = fit_hubbard_density_trial(hop, pinning, 2.0, 16, densities, [2, 3, 4, 5, 6])
    assert fit["u_eff"] == 4
    assert fit["chi"] < 2e-9
    numpy.testing.assert_allclose(fit["k_trial"], reference, atol=1e-8)
    assert fit["metadata"]["inner_uhf_scf"] is False
    assert fit["metadata"]["objective_spread"] > 0.01


@pytest.mark.unit
def test_density_feedback_uses_frozen_qmc_input_not_an_inner_scf():
    hop = hopping_matrix(4, 4)
    numpy.random.seed(421)
    density = numpy.random.uniform(0.25, 0.7, (2, 16))
    pinning = numpy.random.uniform(-0.1, 0.1, (2, 16))
    before = density.copy()
    fit = fit_hubbard_density_trial(hop, pinning, 1.3, 13.0, density, [3.0])
    potential = pinning + 3 * (density[::-1] - 0.5) + fit["mu_eff"]
    for spin in range(2):
        numpy.testing.assert_allclose(
            fit["k_trial"][spin] - hop, numpy.diag(potential[spin]), atol=1e-14
        )
    assert abs(fit["densities"].sum() - 13) < 1e-12
    # The deliberately nonstationary input was not iterated away.
    assert fit["chi"] > 0.01
    numpy.testing.assert_array_equal(density, before)


@pytest.mark.unit
def test_density_feedback_reports_uniform_input_cannot_identify_ueff():
    hop = hopping_matrix(4, 4)
    fit = fit_hubbard_density_trial(
        hop, numpy.zeros((2, 16)), 4, 16, numpy.full((2, 16), 0.5), [1, 4, 8]
    )
    assert fit["metadata"]["objective_spread"] < 1e-13
    for candidate in fit["candidates"]:
        assert candidate["chi"] < 1e-13


@pytest.mark.unit
def test_density_feedback_rejects_invalid_input_and_free_trial_grid():
    hop = hopping_matrix(4, 4)
    args = [hop, numpy.zeros((2, 16)), 2, 16, numpy.full((2, 16), 0.5)]
    with pytest.raises(ValueError, match="strictly positive"):
        fit_hubbard_density_trial(*args, [0, 4])
    args[4] = numpy.full((2, 16), numpy.nan)
    with pytest.raises(ValueError, match="densities"):
        fit_hubbard_density_trial(*args, [4])


if __name__ == "__main__":
    test_hubbard_cpu_smoke()
    test_all_dead_population_fails_before_controller()
    test_cp_scalar_batched_same_random_fields("fixed", True)
    test_noninteracting_trace_matches_fermi_density()
    test_history_clone_and_fullpath_against_dense_cyclic_product()
    test_lattice_and_trial_filling()
    test_invalid_propagation_parameters("U", -1)
    test_trial_fit_recovers_noninteracting_temperature_and_preserves_operator()
    test_trial_fit_records_scf_failures_and_rejects_all_failed()
    test_trace_branching_unbranched_telescoping(True)
    test_trace_branching_allows_negative_trace_with_positive_guide(True)
    test_trace_cache_reset_and_population_buffer_clone()
    test_trace_branching_zero_or_invalid_trace_kills_numerical_path(True, numpy.nan)
    test_common_energy_reference_preserves_paths_and_normalized_weights(True, True)
    test_energy_reference_prevents_extensive_population_guard_hits(True)
    test_trace_weight_update_compensates_extreme_ratios_and_preserves_signs()
    test_trace_weight_update_ordinary_arithmetic_and_true_representation_failures()
    test_trace_propagation_large_compensated_increment_has_no_limiter(True, 80.0)
    test_trace_propagation_true_unrepresentable_total_is_counted(True, 800.0)
    test_trace_force_bias_matches_closed_trace_derivative(True)
    test_trace_force_bias_scalar_batched_parity(True)
    test_trace_force_bias_leaves_noninteracting_propagation_unchanged()
    test_trace_force_bias_sampling_against_interacting_fock_space(64, 96, False, "trace")
    test_multistart_thermal_uhf_density_and_stationarity()
    test_uhf_only_trials_reject_free_electron_candidate()
    test_uhf_feedback_fit_recovers_thermal_density_operator()
    test_figure5_physical_geometry_has_two_periodic_axes_and_zero_pinning()
    test_density_feedback_recovers_known_thermal_uhf_fixed_point()
    test_density_feedback_uses_frozen_qmc_input_not_an_inner_scf()
    test_density_feedback_reports_uniform_input_cannot_identify_ueff()
    test_density_feedback_rejects_invalid_input_and_free_trial_grid()
