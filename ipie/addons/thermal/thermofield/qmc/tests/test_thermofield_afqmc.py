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

from ipie.addons.thermal.thermofield.analysis.thermofield_analysis import (
    extrapolate_eta,
    pooled_ratio,
    summarize,
)
from ipie.addons.thermal.thermofield.estimators.energy import local_energy_thermofield_mixed
from ipie.addons.thermal.thermofield.qmc.thermofield_afqmc import ThermofieldAFQMC
from ipie.addons.thermal.thermofield.trial.thermofield import ThermofieldThermalTrial
from ipie.addons.thermal.thermofield.utils.testing import (
    build_hubbard_atom_hamiltonian,
    build_noninteracting_hamiltonian,
    build_random_real_hamiltonian,
    grand_canonical_ed,
    hubbard_atom_exact,
    hubbard_atom_member_exact,
    noninteracting_exact,
)


def tolerance(err, floor):
    # Loose stochastic tolerance: four jackknife errors, with a floor.
    return max(4.0 * float(err), floor)


@pytest.mark.unit
def test_extrapolate_eta_polynomials():
    eta = numpy.array([1.0, 0.5, 0.25, 0.125])
    for degree in range(4):
        coef = numpy.arange(1.0, degree + 2.0)
        values = numpy.polyval(coef, eta)
        assert extrapolate_eta(eta, values, degree) == pytest.approx(coef[-1], abs=1e-12)
    # The default degree interpolates through all members: for three halving
    # members this is the Richardson combination (8 E(b/8) - 6 E(b/4) + E(b/2)) / 3.
    values = numpy.array([0.3, -1.2, 2.5])
    richardson = (8.0 * values[2] - 6.0 * values[1] + values[0]) / 3.0
    assert extrapolate_eta(eta[:3], values) == pytest.approx(richardson, abs=1e-12)
    # The trailing axis holds the members; leading axes are kept.
    stacked = numpy.stack([values, 2.0 * values])
    numpy.testing.assert_allclose(extrapolate_eta(eta[:3], stacked), [richardson, 2.0 * richardson])
    with pytest.raises(ValueError):
        extrapolate_eta(eta[:2], values[:2], degree=2)


@pytest.mark.unit
def test_pooled_ratio_restores_block_scales():
    # Thermofield AFQMC params.
    nblocks = 6
    seed = 3
    numpy.random.seed(seed)

    numer = numpy.random.normal(size=nblocks) + 1j * numpy.random.normal(size=nblocks)
    denom = 1.0 + numpy.random.uniform(size=nblocks) + 0.1j * numpy.random.normal(size=nblocks)
    scale = numpy.random.uniform(-30.0, 30.0, size=nblocks)
    value, error = pooled_ratio(numer, denom, scale)
    direct = (numer * numpy.exp(scale)).sum() / (denom * numpy.exp(scale)).sum()
    assert value == pytest.approx(direct.real, rel=1e-12)
    N, D = numer * numpy.exp(scale), denom * numpy.exp(scale)
    jack = numpy.array(
        [(numpy.delete(N, b).sum() / numpy.delete(D, b).sum()).real for b in range(nblocks)]
    )
    jack_error = numpy.sqrt((nblocks - 1) * ((jack - jack.mean()) ** 2).mean())
    assert error == pytest.approx(jack_error, rel=1e-10)


@pytest.mark.driver
def test_driver_noninteracting_exact():
    # System params.
    nbasis = 2

    # Thermofield AFQMC params.
    mu = -0.1
    beta = 2.0
    timestep = 0.05
    nwalkers = 4
    nblocks = 2
    seed = 61

    # Without fields every member, the extrapolation and the trace are exact.
    # With the zero-split force bias the members must still be measured with
    # the frontier Green's function (here the only difference between splits).
    for split in ("frontier", "zero"):
        numpy.random.seed(seed)
        h1e = numpy.random.standard_normal((nbasis, nbasis))
        h1e = 0.5 * (h1e + h1e.T)
        hamiltonian = build_noninteracting_hamiltonian(h1e)
        trial = ThermofieldThermalTrial(hamiltonian, beta, mu=mu)
        trial.force_bias_split = split

        afqmc = ThermofieldAFQMC.build(
            mu,
            beta,
            hamiltonian,
            trial=trial,
            nwalkers=nwalkers,
            nblocks=nblocks,
            timestep=timestep,
            seed=7,
        )
        results = afqmc.run(verbose=0)

        _, nav_exact, e_exact = noninteracting_exact(h1e, mu, beta)
        numpy.testing.assert_allclose(results["e_mix"], 2 * e_exact, atol=1e-8)
        numpy.testing.assert_allclose(results["nav_mix"], 2 * nav_exact, atol=1e-8)
        numpy.testing.assert_allclose(results["e_trace"], 2 * e_exact, atol=1e-8)
        numpy.testing.assert_allclose(results["nav_trace"], 2 * nav_exact, atol=1e-8)
        summary = summarize(results, guide_time=trial.guide_time)
        assert summary["eta_extrapolated"]["energy"][0] == pytest.approx(2 * e_exact, abs=1e-8)
        assert summary["mixed"]["energy"][0] == pytest.approx(2 * e_exact, abs=1e-8)
        assert summary["trace"]["nav"][0] == pytest.approx(2 * nav_exact, abs=1e-8)


@pytest.mark.driver
def test_members_only_walk_matches_full_walk():
    # System params.
    eps = 0.1
    U = 2.0

    # Thermofield AFQMC params.
    mu = 0.5
    beta = 2.0
    timestep = 0.05
    nwalkers = 16
    seed = 5

    # With measure_trace=False the walk stops at the last member, and the
    # members are unchanged. One block only: the full walk draws random numbers
    # for its extra slices, so later blocks would start from a different state.
    hamiltonian = build_hubbard_atom_hamiltonian(eps, U)
    runs = []
    for measure_trace in (True, False):
        afqmc = ThermofieldAFQMC.build(
            mu,
            beta,
            hamiltonian,
            nwalkers=nwalkers,
            nblocks=1,
            timestep=timestep,
            seed=seed,
            measure_trace=measure_trace,
        )
        assert afqmc.nwalk == (40 if measure_trace else 35)
        runs.append(afqmc.run(verbose=0))
    full, short = runs
    for key in ("numer_mix", "denom_mix", "navnum_mix", "log_scale_mix"):
        numpy.testing.assert_array_equal(full[key], short[key])
    assert "e_trace" in full and "e_trace" not in short


@pytest.mark.driver
def test_guide_member_is_the_mixed_estimator():
    # System params.
    eps = 0.1
    U = 2.0

    # Thermofield AFQMC params.
    mu = 0.5
    beta = 1.0
    timestep = 0.05
    nwalkers = 16
    seed = 9

    # The member eta = tau_g has F = 1, so it equals the cached mixed estimator.
    hamiltonian = build_hubbard_atom_hamiltonian(eps, U)
    afqmc = ThermofieldAFQMC.build(
        mu,
        beta,
        hamiltonian,
        nwalkers=nwalkers,
        nblocks=1,
        timestep=timestep,
        seed=seed,
        phaseless=False,
        eta_members=(0.5 * beta,),
        measure_trace=False,
    )
    results = afqmc.run(verbose=0)
    # Free projection has no population control, so the walkers are still
    # those of the member slice.
    walkers = afqmc.walkers
    weights = walkers.weight * walkers.phase
    energies = local_energy_thermofield_mixed(hamiltonian, walkers)[:, 0]
    e_cached = numpy.sum(weights * energies) / numpy.sum(weights)
    assert results["e_mix"][0, 0] == pytest.approx(e_cached.real, abs=1e-10)


@pytest.mark.driver
def test_driver_hubbard_atom_free_projection():
    # System params.
    eps = 0.1

    # Thermofield AFQMC params.
    mu = 0.5
    nwalkers = 128
    nblocks = 10
    seed = 7

    # The members, their extrapolation and the trace against the analytic atom.
    # Free projection is exact up to the Trotter error, but the reweighting
    # factors become heavy tailed as U beta grows (at U = 2, beta = 2 a few
    # blocks carry the trace and the block errors are unreliable), so the
    # checks use couplings where the block statistics are well behaved.
    for U, beta in ((0.5, 2.0), (2.0, 0.5)):
        hamiltonian = build_hubbard_atom_hamiltonian(eps, U)
        exact = hubbard_atom_exact(eps, U, mu, beta)
        afqmc = ThermofieldAFQMC.build(
            mu,
            beta,
            hamiltonian,
            nwalkers=nwalkers,
            nblocks=nblocks,
            timestep=beta / 40,
            seed=seed,
            phaseless=False,
        )
        summary = summarize(afqmc.run(verbose=0))

        members = summary["members"]
        for k, eta in enumerate(members["eta"]):
            e_ref, n_ref = hubbard_atom_member_exact(eps, U, mu, beta, eta)
            e, de = members["energy"][0][k], members["energy"][1][k]
            n, dn = members["nav"][0][k], members["nav"][1][k]
            assert e == pytest.approx(e_ref, abs=tolerance(de, 0.01))
            assert n == pytest.approx(n_ref, abs=tolerance(dn, 0.01))
        e, de = summary["trace"]["energy"]
        n, dn = summary["trace"]["nav"]
        assert e == pytest.approx(exact["energy"], abs=tolerance(de, 0.02))
        assert n == pytest.approx(exact["nav"], abs=tolerance(dn, 0.02))
        e, de = summary["eta_extrapolated"]["energy"]
        assert e == pytest.approx(exact["energy"], abs=tolerance(de, 0.02))


@pytest.mark.driver
def test_driver_hubbard_atom_phaseless():
    # System params.
    eps = 0.1
    U = 2.0

    # Thermofield AFQMC params.
    mu = 0.5
    beta = 2.0
    timestep = 0.05
    nwalkers = 64
    nblocks = 10
    seed = 7

    # The phaseless constraint is biased: the estimates must be finite and close
    # at weak coupling.
    hamiltonian = build_hubbard_atom_hamiltonian(eps, U)
    exact = hubbard_atom_exact(eps, U, mu, beta)
    afqmc = ThermofieldAFQMC.build(
        mu, beta, hamiltonian, nwalkers=nwalkers, nblocks=nblocks, timestep=timestep, seed=seed
    )
    results = afqmc.run(verbose=0)
    assert numpy.all(numpy.isfinite(results["e_mix"]))
    assert numpy.all(numpy.isfinite(results["e_trace"]))
    summary = summarize(results)
    assert summary["trace"]["energy"][0] == pytest.approx(exact["energy"], abs=0.3)
    assert summary["eta_extrapolated"]["energy"][0] == pytest.approx(exact["energy"], abs=0.3)


@pytest.mark.driver
def test_driver_random_two_orbital_vs_ed():
    # System params.
    nbasis = 2
    nchol = 2

    # Thermofield AFQMC params.
    mu = 0.1
    beta = 1.0
    timestep = 0.025
    nwalkers = 128
    nblocks = 10
    seed = 11

    # Free-projection trace estimator against Fock-space exact diagonalization.
    hamiltonian = build_random_real_hamiltonian(nbasis, nchol, scale=0.3, seed=71)
    e_exact, nav_exact, _ = grand_canonical_ed(hamiltonian, mu, beta)
    afqmc = ThermofieldAFQMC.build(
        mu,
        beta,
        hamiltonian,
        nwalkers=nwalkers,
        nblocks=nblocks,
        timestep=timestep,
        seed=seed,
        phaseless=False,
    )
    summary = summarize(afqmc.run(verbose=0))
    e, de = summary["trace"]["energy"]
    n, dn = summary["trace"]["nav"]
    assert e == pytest.approx(e_exact, abs=tolerance(de, 0.08))
    assert n == pytest.approx(nav_exact, abs=tolerance(dn, 0.08))


@pytest.mark.unit
def test_eta_members_must_be_whole_slices():
    # System params.
    eps = 0.1
    U = 2.0

    # Thermofield AFQMC params.
    mu = 0.5
    beta = 1.0
    timestep = 0.05
    nwalkers = 2

    hamiltonian = build_hubbard_atom_hamiltonian(eps, U)
    for eta_members in ((0.33,), (1.0,)):
        with pytest.raises(ValueError):
            ThermofieldAFQMC.build(
                mu,
                beta,
                hamiltonian,
                nwalkers=nwalkers,
                timestep=timestep,
                eta_members=eta_members,
            )


if __name__ == "__main__":
    test_extrapolate_eta_polynomials()
    test_pooled_ratio_restores_block_scales()
    test_driver_noninteracting_exact()
    test_members_only_walk_matches_full_walk()
    test_guide_member_is_the_mixed_estimator()
    test_driver_hubbard_atom_free_projection()
    test_driver_hubbard_atom_phaseless()
    test_driver_random_two_orbital_vs_ed()
    test_eta_members_must_be_whole_slices()
