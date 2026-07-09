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
"""Walker container tests: population-control buffer round trip."""

import numpy
import pytest

from ipie.addons.thermal.thermofield.tests.ed_utils import build_noninteracting_hamiltonian
from ipie.addons.thermal.thermofield.trial import ThermofieldThermalTrial
from ipie.addons.thermal.thermofield.walkers import ThermofieldWalkers
from ipie.utils.mpi import MPIHandler
from ipie.walkers.pop_controller import PopController


@pytest.mark.unit
def test_pop_control_clones_full_walker_state():
    """pair_branch cloning must copy the full QDT state through the MPI
    buffer (Qmat, log_d, Tmat, caches), not just the weight."""
    M = 3
    rng = numpy.random.default_rng(47)
    h = rng.standard_normal((M, M))
    h = 0.5 * (h + h.T)
    hamiltonian = build_noninteracting_hamiltonian(h)
    trial = ThermofieldThermalTrial(hamiltonian, beta=1.5, mu=0.1)
    walkers = ThermofieldWalkers(trial, M, 2)

    # Give walker 0 a nontrivial factored state (stabilize populates log_d
    # and Tmat) and force a branch: after rescaling, walker 1 dies.
    Delta = rng.standard_normal((2, M, M)) + 1j * rng.standard_normal((2, M, M))
    walkers.set_delta(0, 2.0 * Delta, trial=trial)
    walkers.stabilize()
    walkers.weight = numpy.array([8.0, 0.05])

    mpi_handler = MPIHandler()
    pcontrol = PopController(2, 1, mpi_handler, "pair_branch")
    numpy.random.seed(7)
    pcontrol.pop_control(walkers, mpi_handler.comm)

    # Walker 1 is a clone of walker 0 with the averaged weight.
    assert walkers.weight[0] == pytest.approx(walkers.weight[1])
    numpy.testing.assert_allclose(walkers.Qmat[1], walkers.Qmat[0], atol=1e-14)
    numpy.testing.assert_allclose(walkers.log_d[1], walkers.log_d[0], atol=1e-14)
    numpy.testing.assert_allclose(walkers.Tmat[1], walkers.Tmat[0], atol=1e-14)
    numpy.testing.assert_allclose(walkers.Ga[1], walkers.Ga[0], atol=1e-14)
    numpy.testing.assert_allclose(walkers.Gb[1], walkers.Gb[0], atol=1e-14)
    assert walkers.log_ovlp[1] == pytest.approx(walkers.log_ovlp[0], abs=1e-14)
