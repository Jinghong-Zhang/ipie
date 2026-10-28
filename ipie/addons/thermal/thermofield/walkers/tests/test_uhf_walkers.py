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
        walkers.add_walker_array("field_hist", numpy.arange(nwalkers * 5.0).reshape(nwalkers, 5))
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
        numpy.testing.assert_allclose(walkers.field_hist[1], walkers.field_hist[0], atol=1e-14)
        assert walkers.log_ovlp[1] == pytest.approx(walkers.log_ovlp[0], abs=1e-14)
        assert numpy.abs(walkers.log_d[0]).max() > 0.0


@pytest.mark.unit
@pytest.mark.parametrize(
    "method, low, high",
    [("pair_branch", -5.0, 4.0), ("stochastic_reconfiguration", -5.0, 4.0), ("comb", -0.6, 0.6)],
)
def test_population_control_matches_core_controller(method, low, high):
    nbasis, nwalkers = 3, 40
    numpy.random.seed(3)
    h1e = numpy.random.standard_normal((nbasis, nbasis))
    hamiltonian = build_noninteracting_hamiltonian(0.5 * (h1e + h1e.T))
    trial = ThermofieldThermalTrial(hamiltonian, beta=1.0, mu=0.1)
    mpi_handler = MPIHandler()
    comm = mpi_handler.comm
    # Distinct walkers on every process. Pair branching and reconfiguration get
    # weights over four decades, so that many walkers move, also across
    # processes; comb gets a narrow spread, where the core controller copies a
    # walker at most twice.
    rng = numpy.random.RandomState(100 + comm.rank)
    weight = numpy.exp(rng.uniform(low, high, nwalkers))
    phase = numpy.exp(1j * rng.uniform(0, 2 * numpy.pi, nwalkers))
    Q = rng.standard_normal((nwalkers, 2, nbasis, nbasis))
    states = []
    # The reordering controller and the core controller, which copies the
    # walker buffers through MPI, must give the same walkers everywhere.
    for controller in (ThermofieldPopController, PopController):
        walkers = ThermofieldWalkers(trial, nbasis, nwalkers)
        walkers.weight = weight.copy()
        walkers.phase = phase.copy()
        walkers.Qmat[:] = Q
        walkers.add_walker_array(
            "field_hist", numpy.arange(nwalkers * 5.0).reshape(nwalkers, 5) + 1000 * comm.rank
        )
        pcontrol = controller(nwalkers, 1, mpi_handler, method)
        numpy.random.seed(11)
        pcontrol.pop_control(walkers, comm)
        states.append(walkers)
    new, core = states
    assert not numpy.array_equal(core.Qmat, Q)
    for name in core.buff_names:
        numpy.testing.assert_array_equal(getattr(new, name), getattr(core, name))


@pytest.mark.unit
def test_comb_population_copies_every_selected_walker():
    nbasis, nwalkers = 2, 30
    numpy.random.seed(5)
    h1e = numpy.random.standard_normal((nbasis, nbasis))
    hamiltonian = build_noninteracting_hamiltonian(0.5 * (h1e + h1e.T))
    trial = ThermofieldThermalTrial(hamiltonian, beta=1.0, mu=0.1)
    mpi_handler = MPIHandler()
    comm = mpi_handler.comm
    rng = numpy.random.RandomState(200 + comm.rank)
    weight = numpy.exp(rng.uniform(-5.0, 4.0, nwalkers))
    Q = rng.standard_normal((nwalkers, 2, nbasis, nbasis))
    walkers = ThermofieldWalkers(trial, nbasis, nwalkers)
    walkers.weight = weight.copy()
    walkers.Qmat[:] = Q
    # Global walker ids.
    walkers.add_walker_array("walker_id", numpy.arange(nwalkers) + nwalkers * comm.rank + 0.0)
    pcontrol = ThermofieldPopController(nwalkers, 1, mpi_handler, "comb")
    numpy.random.seed(11)
    pcontrol.pop_control(walkers, comm)
    all_weights = numpy.concatenate(comm.allgather(weight))
    source = numpy.concatenate(comm.allgather(walkers.walker_id)).astype(int)
    # A comb with one tooth per walker gives floor(n w / W) or ceil(n w / W) copies.
    copies = numpy.bincount(source, minlength=len(all_weights))
    expected = len(all_weights) * all_weights / all_weights.sum()
    assert copies.max() > 2
    assert numpy.all(numpy.floor(expected - 1e-9) <= copies)
    assert numpy.all(copies <= numpy.ceil(expected + 1e-9))
    mine = source[comm.rank * nwalkers : (comm.rank + 1) * nwalkers]
    numpy.testing.assert_array_equal(walkers.Qmat, numpy.concatenate(comm.allgather(Q))[mine])
    numpy.testing.assert_array_equal(walkers.weight, 1.0)


if __name__ == "__main__":
    test_pop_control_clones_full_walker_state()
    test_population_control_matches_core_controller("pair_branch", -5.0, 4.0)
    test_comb_population_copies_every_selected_walker()
