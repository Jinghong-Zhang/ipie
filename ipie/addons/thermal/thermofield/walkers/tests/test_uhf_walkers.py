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

from ipie.addons.thermal.thermofield.trial.thermofield import ThermofieldThermalTrial
from ipie.addons.thermal.thermofield.utils.testing import build_noninteracting_hamiltonian
from ipie.addons.thermal.thermofield.walkers.pop_controller import ThermofieldPopController
from ipie.addons.thermal.thermofield.walkers.uhf_walkers import ThermofieldWalkers
from ipie.utils.mpi import MPIHandler
from ipie.walkers.pop_controller import PopController


@pytest.mark.unit
def test_pop_control_clones_full_walker_state():
    # System params.
    nbasis = 3

    # Thermofield AFQMC params.
    mu = 0.1
    beta = 1.5
    nwalkers = 2
    seed = 47
    numpy.random.seed(seed)

    h1e = numpy.random.standard_normal((nbasis, nbasis))
    hamiltonian = build_noninteracting_hamiltonian(0.5 * (h1e + h1e.T))
    trial = ThermofieldThermalTrial(hamiltonian, beta=beta, mu=mu)
    shape = (2, nbasis, nbasis)
    Delta = numpy.random.standard_normal(shape) + 1j * numpy.random.standard_normal(shape)

    # The core controller (used with several MPI processes) copies the walker
    # buffers; the thermofield controller reorders the walkers in place. Both
    # must clone the full factored state, the caches and the registered arrays.
    for controller in (PopController, ThermofieldPopController):
        walkers = ThermofieldWalkers(trial, nbasis, nwalkers)
        # Walker 0 gets a nontrivial factored state (stabilize sets log_d and
        # Tmat); after rescaling, walker 1 dies and is replaced by a clone.
        walkers.set_delta(0, 2.0 * Delta, trial=trial)
        walkers.stabilize()
        walkers.weight = numpy.array([8.0, 0.05])

        mpi_handler = MPIHandler()
        pcontrol = controller(nwalkers, 1, mpi_handler, "pair_branch")
        numpy.random.seed(7)
        pcontrol.pop_control(walkers, mpi_handler.comm)

        assert walkers.weight[0] == pytest.approx(walkers.weight[1])
        numpy.testing.assert_allclose(walkers.Qmat[1], walkers.Qmat[0], atol=1e-14)
        numpy.testing.assert_allclose(walkers.log_d[1], walkers.log_d[0], atol=1e-14)
        numpy.testing.assert_allclose(walkers.Tmat[1], walkers.Tmat[0], atol=1e-14)
        numpy.testing.assert_allclose(walkers.Ga[1], walkers.Ga[0], atol=1e-14)
        numpy.testing.assert_allclose(walkers.Gb[1], walkers.Gb[0], atol=1e-14)
        assert walkers.log_ovlp[1] == pytest.approx(walkers.log_ovlp[0], abs=1e-14)
        assert numpy.abs(walkers.log_d[0]).max() > 0.0


if __name__ == "__main__":
    test_pop_control_clones_full_walker_state()
