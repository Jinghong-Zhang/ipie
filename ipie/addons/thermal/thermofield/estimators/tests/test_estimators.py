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

from ipie.addons.thermal.thermofield.estimators.energy import (
    local_energy_thermofield_mixed,
    ThermofieldEnergyEstimator,
)
from ipie.addons.thermal.thermofield.estimators.particle_number import (
    particle_number_thermofield_mixed,
    ThermofieldNumberEstimator,
)
from ipie.addons.thermal.thermofield.propagation.phaseless_thermofield import ThermofieldPhaseless
from ipie.addons.thermal.thermofield.trial.thermofield import ThermofieldThermalTrial
from ipie.addons.thermal.thermofield.utils.testing import (
    build_hubbard_atom_hamiltonian,
    build_noninteracting_hamiltonian,
    hubbard_atom_exact,
    noninteracting_exact,
)
from ipie.addons.thermal.thermofield.walkers.uhf_walkers import ThermofieldWalkers


@pytest.mark.unit
def test_beta_zero_noninteracting():
    # System params.
    nbasis = 4

    # Thermofield AFQMC params.
    nwalkers = 2
    seed = 19
    numpy.random.seed(seed)

    h1e = numpy.random.standard_normal((nbasis, nbasis))
    h1e = 0.5 * (h1e + h1e.T)
    hamiltonian = build_noninteracting_hamiltonian(h1e)
    trial = ThermofieldThermalTrial(hamiltonian, beta=0.0, mu=0.0)
    walkers = ThermofieldWalkers(trial, nbasis, nwalkers)

    # At beta = 0, f = I / 2 for each spin: N = nbasis and E = Tr(h).
    energies = local_energy_thermofield_mixed(hamiltonian, walkers)
    navs = particle_number_thermofield_mixed(walkers)
    assert navs[0] == pytest.approx(nbasis, abs=1e-10)
    assert energies[0, 0] == pytest.approx(numpy.trace(h1e), abs=1e-10)


@pytest.mark.unit
def test_beta_zero_hubbard_atom():
    # System params.
    eps = 0.3
    U = 2.0

    # Thermofield AFQMC params.
    nwalkers = 2

    hamiltonian = build_hubbard_atom_hamiltonian(eps, U)
    trial = ThermofieldThermalTrial(hamiltonian, beta=0.0, mu=0.0)
    walkers = ThermofieldWalkers(trial, 1, nwalkers)

    # At beta = 0 the atom has N = 1 and E = eps + U / 4.
    energies = local_energy_thermofield_mixed(hamiltonian, walkers)
    navs = particle_number_thermofield_mixed(walkers)
    assert navs[0] == pytest.approx(1.0, abs=1e-10)
    assert energies[0, 0] == pytest.approx(eps + 0.25 * U, abs=1e-10)

    exact = hubbard_atom_exact(eps, U, mu=0.0, beta=0.0)
    assert exact["nav"] == pytest.approx(1.0, abs=1e-14)
    assert exact["energy"] == pytest.approx(eps + 0.25 * U, abs=1e-14)


@pytest.mark.unit
def test_noninteracting_mixed_estimator():
    # System params.
    nbasis = 4

    # Thermofield AFQMC params.
    mu = -0.1
    timestep = 0.05
    nwalkers = 3
    seed = 29

    # Without interactions the mixed estimator is exact.
    for beta in (0.5, 2.0):
        numpy.random.seed(seed)
        h1e = numpy.random.standard_normal((nbasis, nbasis))
        h1e = 0.5 * (h1e + h1e.T)
        hamiltonian = build_noninteracting_hamiltonian(h1e)
        trial = ThermofieldThermalTrial(hamiltonian, beta=beta, mu=mu)
        walkers = ThermofieldWalkers(trial, nbasis, nwalkers)
        propagator = ThermofieldPhaseless(timestep, mu)
        propagator.build(hamiltonian, trial=trial, walkers=walkers)

        nslices = int(numpy.rint(0.5 * beta / timestep))
        numpy.random.seed(7)
        for _ in range(nslices):
            propagator.propagate_walkers(walkers, hamiltonian, trial)

        _, nav_exact, e_exact = noninteracting_exact(h1e, mu, beta)
        energies = local_energy_thermofield_mixed(hamiltonian, walkers)
        navs = particle_number_thermofield_mixed(walkers)
        weights = walkers.weight
        e_mix = numpy.sum(weights * energies[:, 0]).real / numpy.sum(weights)
        nav_mix = numpy.sum(weights * navs).real / numpy.sum(weights)
        # Two spin species.
        assert e_mix == pytest.approx(2 * e_exact, abs=1e-10)
        assert nav_mix == pytest.approx(2 * nav_exact, abs=1e-10)


@pytest.mark.unit
def test_estimator_classes_api():
    # System params.
    nbasis = 2

    # Thermofield AFQMC params.
    mu = 0.0
    beta = 1.0
    nwalkers = 4
    seed = 37
    numpy.random.seed(seed)

    h1e = numpy.random.standard_normal((nbasis, nbasis))
    h1e = 0.5 * (h1e + h1e.T)
    hamiltonian = build_noninteracting_hamiltonian(h1e)
    trial = ThermofieldThermalTrial(hamiltonian, beta=beta, mu=mu)
    walkers = ThermofieldWalkers(trial, nbasis, nwalkers)

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
    # The estimator classes keep the complex free-projection reweighting.
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


if __name__ == "__main__":
    test_beta_zero_noninteracting()
    test_beta_zero_hubbard_atom()
    test_noninteracting_mixed_estimator()
    test_estimator_classes_api()
    test_estimator_classes_include_free_projection_phase()
