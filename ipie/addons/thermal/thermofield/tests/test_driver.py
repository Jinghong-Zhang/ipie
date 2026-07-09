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
"""Driver tests: noninteracting exactness end-to-end, the Hubbard-atom
grand-canonical analytic benchmark (Test 4), a random two-orbital ED
regression (Test 10), and a stochastic canonical dimer check (Test 5)."""

import numpy
import pytest

from ipie.addons.thermal.thermofield.canonical import canonical_replica_local_energy
from ipie.addons.thermal.thermofield.estimators import replica_energy_estimate
from ipie.addons.thermal.thermofield.propagation import ThermofieldPhaseless
from ipie.addons.thermal.thermofield.qmc import (
    _allgather_thermofield_walkers,
    ThermofieldAFQMC,
)
from ipie.addons.thermal.thermofield.tests.ed_utils import (
    build_hubbard_atom_hamiltonian,
    build_hubbard_dimer_hamiltonian,
    build_noninteracting_hamiltonian,
    build_random_real_hamiltonian,
    canonical_ed,
    grand_canonical_ed,
    hubbard_atom_exact,
    noninteracting_exact,
)
from ipie.addons.thermal.thermofield.trial import ThermofieldThermalTrial
from ipie.addons.thermal.thermofield.walkers import ThermofieldWalkers
from ipie.config import MPI


def stochastic_tolerance(blocks, floor):
    """Loose tolerance: 4 sigma of the block mean, floored."""
    sem = numpy.std(blocks) / max(1.0, numpy.sqrt(len(blocks)))
    return max(4.0 * sem, floor)


@pytest.mark.driver
def test_driver_noninteracting_end_to_end():
    """The full driver (with population control) is exact without fields."""
    M = 2
    beta, mu, timestep = 2.0, -0.1, 0.05
    rng = numpy.random.default_rng(61)
    h = rng.standard_normal((M, M))
    h = 0.5 * (h + h.T)
    hamiltonian = build_noninteracting_hamiltonian(h)

    afqmc = ThermofieldAFQMC.build(
        mu,
        beta,
        hamiltonian,
        nwalkers=4,
        nblocks=2,
        timestep=timestep,
        seed=7,
        pairing="all_pairs",
    )
    results = afqmc.run(verbose=0)

    _, nav_exact, e_exact = noninteracting_exact(h, mu, beta)
    numpy.testing.assert_allclose(results["e_rep"], 2 * e_exact, atol=1e-8)
    numpy.testing.assert_allclose(results["e_mix"], 2 * e_exact, atol=1e-8)
    numpy.testing.assert_allclose(results["nav_rep"], 2 * nav_exact, atol=1e-8)


@pytest.mark.driver
@pytest.mark.parametrize("beta", [0.5, 2.0])
def test_driver_hubbard_atom_free_projection(beta):
    """Test 4: Hubbard atom vs the analytic grand-canonical answer, in
    free-projection (phaseless=False) mode with complex replica weights."""
    eps, U, mu = 0.1, 2.0, 0.5
    timestep = 0.05
    hamiltonian = build_hubbard_atom_hamiltonian(eps, U)
    exact = hubbard_atom_exact(eps, U, mu, beta)

    afqmc = ThermofieldAFQMC.build(
        mu,
        beta,
        hamiltonian,
        nwalkers=128,
        nblocks=10,
        timestep=timestep,
        seed=7,
        phaseless=False,
        pairing="random_permutation",
        measure_mode="complex",
    )
    results = afqmc.run(verbose=0)

    e_rep = numpy.mean(results["e_rep"])
    nav_rep = numpy.mean(results["nav_rep"])
    assert e_rep == pytest.approx(exact["energy"], abs=stochastic_tolerance(results["e_rep"], 0.05))
    assert nav_rep == pytest.approx(
        exact["nav"], abs=stochastic_tolerance(results["nav_rep"], 0.05)
    )


@pytest.mark.driver
def test_driver_hubbard_atom_phaseless():
    """Test 4 (phaseless): finite, stable, and close for weak coupling."""
    eps, U, mu, beta = 0.1, 2.0, 0.5, 2.0
    hamiltonian = build_hubbard_atom_hamiltonian(eps, U)
    exact = hubbard_atom_exact(eps, U, mu, beta)

    afqmc = ThermofieldAFQMC.build(
        mu,
        beta,
        hamiltonian,
        nwalkers=64,
        nblocks=10,
        timestep=0.05,
        seed=7,
        phaseless=True,
        measure_mode="phaseless",
    )
    results = afqmc.run(verbose=0)

    assert numpy.all(numpy.isfinite(results["e_rep"]))
    assert numpy.all(numpy.isfinite(results["e_mix"]))
    # Loose: phaseless + cosine-projected replica reweighting are both biased.
    assert numpy.mean(results["e_rep"]) == pytest.approx(exact["energy"], abs=0.3)


@pytest.mark.driver
def test_driver_random_two_orbital_vs_ed():
    """Test 10: random two-orbital ab initio Hamiltonian vs Fock-space ED,
    free projection."""
    beta, mu, timestep = 1.0, 0.1, 0.05
    hamiltonian = build_random_real_hamiltonian(2, 2, scale=0.3, seed=71)
    e_exact, nav_exact, _ = grand_canonical_ed(hamiltonian, mu, beta)

    afqmc = ThermofieldAFQMC.build(
        mu,
        beta,
        hamiltonian,
        nwalkers=128,
        nblocks=10,
        timestep=timestep,
        seed=11,
        phaseless=False,
        measure_mode="complex",
    )
    results = afqmc.run(verbose=0)

    assert numpy.mean(results["e_rep"]) == pytest.approx(
        e_exact, abs=stochastic_tolerance(results["e_rep"], 0.08)
    )
    assert numpy.mean(results["nav_rep"]) == pytest.approx(
        nav_exact, abs=stochastic_tolerance(results["nav_rep"], 0.08)
    )


@pytest.mark.driver
def test_canonical_dimer_stochastic():
    """Test 5 (stochastic): canonical-projected replica estimator on the
    interacting Hubbard dimer at N_up = N_down = 1 vs canonical ED.

    Walkers are sampled grand canonically with the fixed target-beta guide;
    the canonical projection acts at measurement time (the chemical potential
    cancels in the fixed-N ratio).
    """
    t, U, beta, mu = 1.0, 2.0, 1.0, 0.2
    timestep = 0.05
    nwalkers = 24
    nblocks = 8
    hamiltonian = build_hubbard_dimer_hamiltonian(t, U)
    e_exact = canonical_ed(hamiltonian, (1, 1), beta)

    trial = ThermofieldThermalTrial(hamiltonian, beta, mu=mu)
    propagator = ThermofieldPhaseless(timestep, mu, phaseless=False)
    walkers_left = ThermofieldWalkers(trial, 2, nwalkers)
    walkers_right = ThermofieldWalkers(trial, 2, nwalkers)
    propagator.build(hamiltonian, trial=trial, walkers=walkers_right)

    nslices = int(numpy.rint(0.5 * beta / timestep))
    numpy.random.seed(7)
    estimates = []
    for _ in range(nblocks):
        for walkers in (walkers_left, walkers_right):
            walkers.reset(trial)
            for _ in range(nslices):
                propagator.propagate_walkers(walkers, hamiltonian, trial)
        numer = 0.0j
        denom = 0.0j
        for i in range(nwalkers):
            for j in range(nwalkers):
                energy_N, ovlp_N = canonical_replica_local_energy(
                    hamiltonian,
                    walkers_left.get_delta(i),
                    walkers_right.get_delta(j),
                    (1, 1),
                )
                w = (
                    walkers_left.weight[i]
                    * numpy.conj(walkers_left.phase[i])
                    * walkers_right.weight[j]
                    * walkers_right.phase[j]
                )
                q = ovlp_N * numpy.exp(
                    -numpy.conj(walkers_left.log_ovlp[i]) - walkers_right.log_ovlp[j]
                )
                numer += w * q * energy_N
                denom += w * q
        estimates.append((numer / denom).real)

    estimates = numpy.array(estimates)
    assert numpy.mean(estimates) == pytest.approx(e_exact, abs=stochastic_tolerance(estimates, 0.1))


@pytest.mark.mpi
@pytest.mark.skipif(MPI.COMM_WORLD.size != 2, reason="Requires exactly two MPI ranks.")
def test_all_pairs_includes_cross_rank_walkers():
    """The MPI reduction must reproduce all pairs of the global populations."""
    comm = MPI.COMM_WORLD
    hamiltonian = build_noninteracting_hamiltonian(numpy.array([[1.0]]))
    trial = ThermofieldThermalTrial(hamiltonian, beta=0.0)

    left_values = (0.2, 2.0)
    right_values = (0.3, 3.0)
    walkers_left = ThermofieldWalkers(trial, 1, 1)
    walkers_right = ThermofieldWalkers(trial, 1, 1)
    walkers_left.set_delta(
        0, numpy.full((2, 1, 1), left_values[comm.rank], dtype=numpy.complex128), trial
    )
    walkers_right.set_delta(
        0, numpy.full((2, 1, 1), right_values[comm.rank], dtype=numpy.complex128), trial
    )

    global_right = _allgather_thermofield_walkers(walkers_right, comm)
    local = replica_energy_estimate(
        hamiltonian, walkers_left, global_right, pairing="all_pairs"
    )
    shift_max = comm.allreduce(local["log_shift"], op=MPI.MAX)
    scale = numpy.exp(local["log_shift"] - shift_max)
    numer = comm.allreduce(scale * local["numer"], op=MPI.SUM)
    denom = comm.allreduce(scale * local["denom"], op=MPI.SUM)

    global_left = _allgather_thermofield_walkers(walkers_left, comm)
    reference = replica_energy_estimate(
        hamiltonian, global_left, global_right, pairing="all_pairs"
    )
    assert (numer / denom).real == pytest.approx(reference["etotal"], abs=1e-12)
