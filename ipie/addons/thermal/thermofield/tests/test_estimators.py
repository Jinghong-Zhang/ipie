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
"""Estimator tests: beta = 0 identity limit (Test 3), noninteracting
exactness of mixed and two-replica estimators (Tests 2 and 8)."""

import numpy
import pytest

from ipie.addons.thermal.thermofield.estimators import (
    local_energy_thermofield_mixed,
    particle_number_thermofield_mixed,
    replica_energy_estimate,
    replica_pair_quantities,
    ThermofieldEnergyEstimator,
    ThermofieldNumberEstimator,
)
from ipie.addons.thermal.thermofield.propagation import ThermofieldPhaseless
from ipie.addons.thermal.thermofield.tests.ed_utils import (
    build_hubbard_atom_hamiltonian,
    build_noninteracting_hamiltonian,
    hubbard_atom_exact,
    noninteracting_exact,
)
from ipie.addons.thermal.thermofield.trial import ThermofieldThermalTrial
from ipie.addons.thermal.thermofield.walkers import ThermofieldWalkers


@pytest.mark.unit
def test_beta_zero_noninteracting():
    """Test 3a: at beta = 0, N = M per spin pair and E1 = Tr(h) (f = I/2)."""
    M = 4
    rng = numpy.random.default_rng(19)
    h = rng.standard_normal((M, M))
    h = 0.5 * (h + h.T)
    hamiltonian = build_noninteracting_hamiltonian(h)
    trial = ThermofieldThermalTrial(hamiltonian, beta=0.0, mu=0.0)
    walkers = ThermofieldWalkers(trial, M, 2)

    energies = local_energy_thermofield_mixed(hamiltonian, walkers)
    navs = particle_number_thermofield_mixed(walkers)
    # Two spin species, f = I/2 each: N = M, E = 2 * 0.5 * Tr(h) = Tr(h).
    assert navs[0] == pytest.approx(M, abs=1e-10)
    assert energies[0, 0] == pytest.approx(numpy.trace(h), abs=1e-10)


@pytest.mark.unit
def test_beta_zero_hubbard_atom():
    """Test 3b: Hubbard atom at beta = 0: N = 1, E = eps + U / 4."""
    eps, U = 0.3, 2.0
    hamiltonian = build_hubbard_atom_hamiltonian(eps, U)
    trial = ThermofieldThermalTrial(hamiltonian, beta=0.0, mu=0.0)
    walkers = ThermofieldWalkers(trial, 1, 2)

    energies = local_energy_thermofield_mixed(hamiltonian, walkers)
    navs = particle_number_thermofield_mixed(walkers)
    assert navs[0] == pytest.approx(1.0, abs=1e-10)
    assert energies[0, 0] == pytest.approx(eps + 0.25 * U, abs=1e-10)

    exact = hubbard_atom_exact(eps, U, mu=0.0, beta=0.0)
    assert exact["nav"] == pytest.approx(1.0, abs=1e-14)
    assert exact["energy"] == pytest.approx(eps + 0.25 * U, abs=1e-14)

    # Replica estimator at Delta = I on both sides gives the same.
    walkers_left = ThermofieldWalkers(trial, 1, 2)
    rep = replica_energy_estimate(hamiltonian, walkers_left, walkers, pairing="all_pairs")
    assert rep["etotal"] == pytest.approx(eps + 0.25 * U, abs=1e-10)
    assert rep["nav"] == pytest.approx(1.0, abs=1e-10)


def propagate_population(hamiltonian, trial, walkers, propagator, nslices):
    for _ in range(nslices):
        propagator.propagate_walkers(walkers, hamiltonian, trial)


@pytest.mark.unit
@pytest.mark.parametrize("beta", [0.5, 2.0])
def test_noninteracting_mixed_and_replica_estimators(beta):
    """Tests 2 and 8: mixed and replica estimators are exact without
    interactions, for all pairings and measurement modes."""
    M = 4
    mu = -0.1
    timestep = 0.05
    rng = numpy.random.default_rng(29)
    h = rng.standard_normal((M, M))
    h = 0.5 * (h + h.T)
    hamiltonian = build_noninteracting_hamiltonian(h)
    trial = ThermofieldThermalTrial(hamiltonian, beta=beta, mu=mu)

    nwalkers = 3
    walkers_left = ThermofieldWalkers(trial, M, nwalkers)
    walkers_right = ThermofieldWalkers(trial, M, nwalkers)
    propagator = ThermofieldPhaseless(timestep, mu)
    propagator.build(hamiltonian, trial=trial, walkers=walkers_right)

    nslices = int(numpy.rint(0.5 * beta / timestep))
    numpy.random.seed(7)
    propagate_population(hamiltonian, trial, walkers_left, propagator, nslices)
    propagate_population(hamiltonian, trial, walkers_right, propagator, nslices)

    f, nav_exact, e_exact = noninteracting_exact(h, mu, beta)
    nav_exact *= 2  # Two spin species.
    e_exact *= 2

    energies = local_energy_thermofield_mixed(hamiltonian, walkers_right)
    navs = particle_number_thermofield_mixed(walkers_right)
    weights = walkers_right.weight
    e_mix = numpy.sum(weights * energies[:, 0]).real / numpy.sum(weights)
    nav_mix = numpy.sum(weights * navs).real / numpy.sum(weights)
    assert e_mix == pytest.approx(e_exact, abs=1e-10)
    assert nav_mix == pytest.approx(nav_exact, abs=1e-10)

    rng_pairs = numpy.random.default_rng(3)
    for pairing in ("all_pairs", "random_permutation", "k_random_partners"):
        for mode in ("complex", "phaseless"):
            rep = replica_energy_estimate(
                hamiltonian,
                walkers_left,
                walkers_right,
                pairing=pairing,
                mode=mode,
                rng=rng_pairs,
                num_partners=2,
            )
            assert rep["etotal"] == pytest.approx(e_exact, abs=1e-10), (pairing, mode)
            assert rep["nav"] == pytest.approx(nav_exact, abs=1e-10), (pairing, mode)

    # Q_ij bookkeeping: for the deterministic noninteracting walk,
    # Q_ij = S_ij / (S_T(Delta_L)^* S_T(Delta_R)) directly.
    log_S, _, _ = replica_pair_quantities(hamiltonian, walkers_left, walkers_right, 0, 1)
    log_q = log_S - numpy.conj(walkers_left.log_ovlp[0]) - walkers_right.log_ovlp[1]
    from ipie.addons.thermal.thermofield.gaussian import thermofield_log_overlap

    Delta_L = walkers_left.get_delta(0)
    Delta_R = walkers_right.get_delta(1)
    manual = 0.0j
    for s in range(2):
        manual += thermofield_log_overlap(Delta_L[s].conj().T, Delta_R[s])
    manual -= numpy.conj(trial.calc_log_overlap(Delta_L))
    manual -= trial.calc_log_overlap(Delta_R)
    assert log_q == pytest.approx(manual, abs=1e-10)


@pytest.mark.unit
def test_estimator_classes_api():
    """EstimatorBase plumbing of the mixed energy / number estimators."""
    M = 2
    rng = numpy.random.default_rng(37)
    h = rng.standard_normal((M, M))
    h = 0.5 * (h + h.T)
    hamiltonian = build_noninteracting_hamiltonian(h)
    trial = ThermofieldThermalTrial(hamiltonian, beta=1.0, mu=0.0)
    walkers = ThermofieldWalkers(trial, M, 4)

    estim = ThermofieldEnergyEstimator(hamiltonian=hamiltonian, trial=trial)
    assert len(estim.names) == 5
    data = estim.compute_estimator(walkers=walkers, hamiltonian=hamiltonian, trial=trial)
    assert estim.print_to_stdout
    assert estim.shape == (5,)
    tmp = data.copy()
    estim.post_reduce_hook(tmp)
    e_ref = numpy.sum(
        walkers.weight * local_energy_thermofield_mixed(hamiltonian, walkers)[:, 0].real
    ) / numpy.sum(walkers.weight)
    assert tmp[estim.get_index("ETotal")].real == pytest.approx(e_ref, abs=1e-12)

    nestim = ThermofieldNumberEstimator(hamiltonian=hamiltonian, trial=trial)
    assert len(nestim.names) == 3
    data = nestim.compute_estimator(walkers=walkers, hamiltonian=hamiltonian, trial=trial)
    tmp = data.copy()
    nestim.post_reduce_hook(tmp)
    nav_ref = numpy.sum(
        walkers.weight * particle_number_thermofield_mixed(walkers).real
    ) / numpy.sum(walkers.weight)
    assert tmp[nestim.get_index("Nav")].real == pytest.approx(nav_ref, abs=1e-12)


@pytest.mark.unit
def test_estimator_classes_include_free_projection_phase():
    """Mixed estimator classes retain complex free-projection reweighting."""
    hamiltonian = build_noninteracting_hamiltonian(numpy.array([[1.0]]))
    trial = ThermofieldThermalTrial(hamiltonian, beta=0.0)
    walkers = ThermofieldWalkers(trial, 1, 2)
    walkers.set_delta(0, numpy.full((2, 1, 1), 0.2), trial)
    walkers.set_delta(1, numpy.full((2, 1, 1), 2.0), trial)
    walkers.weight = numpy.array([1.0, 2.0])
    walkers.phase = numpy.array([1.0, 1.0j])
    weights = walkers.weight * walkers.phase

    energy = local_energy_thermofield_mixed(hamiltonian, walkers)
    estimator = ThermofieldEnergyEstimator(hamiltonian=hamiltonian, trial=trial)
    data = estimator.compute_estimator(walkers=walkers, hamiltonian=hamiltonian)
    estimator.post_reduce_hook(data)
    expected_energy = numpy.sum(weights * energy[:, 0]) / numpy.sum(weights)
    assert data[estimator.get_index("ETotal")] == pytest.approx(expected_energy, abs=1e-12)

    nav = particle_number_thermofield_mixed(walkers)
    estimator = ThermofieldNumberEstimator(hamiltonian=hamiltonian, trial=trial)
    data = estimator.compute_estimator(walkers=walkers)
    estimator.post_reduce_hook(data)
    expected_nav = numpy.sum(weights * nav) / numpy.sum(weights)
    assert data[estimator.get_index("Nav")] == pytest.approx(expected_nav, abs=1e-12)
