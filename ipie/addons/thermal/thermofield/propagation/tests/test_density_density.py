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

from ipie.addons.thermal.thermofield.estimators.density_density import DensityDensityObservables
from ipie.addons.thermal.thermofield.estimators.greens_function import thermofield_greens_function
from ipie.addons.thermal.thermofield.hamiltonians.density_density import (
    ising_pairs,
    slater_d_interaction,
)
from ipie.addons.thermal.thermofield.hamiltonians.hubbard import build_hubbard_hamiltonian
from ipie.addons.thermal.thermofield.propagation.density_density import (
    ThermofieldDensityDensityDiscrete,
)
from ipie.addons.thermal.thermofield.propagation.hubbard_discrete import ThermofieldHubbardDiscrete
from ipie.addons.thermal.thermofield.qmc.hubbard_afqmc import (
    propagate_population_cp,
    ThermofieldHubbardAFQMC,
)
from ipie.addons.thermal.thermofield.trial.hubbard import rhf_k_trial, ThermofieldCPTrial
from ipie.addons.thermal.thermofield.utils.testing import creation_operators
from ipie.addons.thermal.thermofield.walkers.pop_controller import ThermofieldPopController
from ipie.addons.thermal.thermofield.walkers.uhf_walkers import ThermofieldWalkers
from ipie.hamiltonians.generic import Generic
from ipie.utils.mpi import MPIHandler


def _two_orbital_model(nsites=1, t=0.4):
    # Two orbitals per site with one negative same-spin pair; random hopping from the global
    # stream. h1 has shape (2, M, M).
    m = 2 * nsites
    h = numpy.random.normal(scale=t, size=(m, m))
    h = h + h.T
    h1 = numpy.array([h + numpy.diag([0.1, -0.2] * nsites), h + numpy.diag([-0.05, 0.15] * nsites)])
    direct = numpy.array([[2.0, 1.1], [1.1, 1.6]])
    # U - J < 0 for the same-spin pair.
    exchange = numpy.array([[2.0, 1.4], [1.4, 1.6]])
    atoms = numpy.arange(m).reshape(nsites, 2)
    return h1, ising_pairs(direct, exchange, atoms), atoms


def _fock_model(h1, pairs):
    m = h1.shape[-1]
    cre = creation_operators(2 * m)
    ann = [c.T for c in cre]
    num = [cre[k] @ ann[k] for k in range(2 * m)]
    one = sum(
        h1[s, i, j] * (cre[s * m + i] @ ann[s * m + j])
        for s in range(2)
        for i in range(m)
        for j in range(m)
    )
    inter = sum(
        V * num[int(sa) * m + int(i)] @ num[int(sb) * m + int(j)] for sa, i, sb, j, V in pairs
    )
    mz = sum(num[k] for k in range(m)) - sum(num[m + k] for k in range(m))
    return one, inter, mz, num


def _hubbard_objects(prop_cls, mode, importance_guide="same"):
    # Pinned 2x2 lattice, open in x, with on-site spin Hirsch fields or the equivalent pairs.
    U = 4.0
    beta = 0.4
    dt = 0.02
    nwalkers = 6
    ham, hop, pin = build_hubbard_hamiltonian(
        2, 2, U, 0.5, pbc_x=False, pin_h=0.1, pin_edges=("left",)
    )
    k_trial, _, _ = rhf_k_trial(hop, pin, beta, 3.5)
    trial = ThermofieldCPTrial(ham, beta, k_trial=k_trial, guide_time=beta)
    mpi = MPIHandler()
    walkers = ThermofieldWalkers(trial, 4, nwalkers, mpi)
    kwargs = dict(
        record_fields=True,
        guide_mode=mode,
        fixed_guide_time=beta,
        energy_shift=-8.0,
        importance_guide=importance_guide,
    )
    if prop_cls is ThermofieldHubbardDiscrete:
        prop = prop_cls(ham.H1 + U / 2 * numpy.eye(4), U, dt, beta, **kwargs)
    else:
        pairs = [(0, i, 1, i, U) for i in range(4)]
        prop = prop_cls(ham.H1, pairs, dt, beta, grouped=False, **kwargs)
    driver = SimpleNamespace(
        mpi_handler=mpi, params=SimpleNamespace(num_stblz=5, pop_control_freq=5)
    )
    pcontrol = ThermofieldPopController(nwalkers, round(beta / dt), mpi, "pair_branch")
    return trial, walkers, prop, driver, pcontrol


@pytest.mark.unit
def test_slater_d_matrices_sum_rules_and_t2g_kanamori():
    U, J = 2.3, 0.9
    direct, exchange = slater_d_interaction(U, J)
    f2 = 14 * J / 1.625
    f4 = 0.625 * f2
    numpy.testing.assert_allclose(direct, direct.T, atol=1e-12)
    numpy.testing.assert_allclose(exchange, exchange.T, atol=1e-12)
    numpy.testing.assert_allclose(numpy.diag(exchange), numpy.diag(direct), atol=1e-12)
    assert direct.mean() == pytest.approx(U, abs=1e-12)
    off = ~numpy.eye(5, dtype=bool)
    assert (direct - exchange)[off].mean() == pytest.approx(U - J, abs=1e-12)
    # Intra-orbital U and the t2g Kanamori J (dxz, dyz, dxy) from Slater integrals.
    numpy.testing.assert_allclose(numpy.diag(direct), U + 4 / 49 * f2 + 36 / 441 * f4, atol=1e-12)
    t2g = [1, 2, 4]
    j_t2g = 3 / 49 * f2 + 20 / 441 * f4
    for a, b in product(t2g, t2g):
        if a != b:
            assert exchange[a, b] == pytest.approx(j_t2g, abs=1e-12)
            assert direct[a, b] == pytest.approx(direct[a, a] - 2 * j_t2g, abs=1e-12)


@pytest.mark.unit
def test_ising_pair_count_and_orbital_rows():
    direct, exchange = slater_d_interaction(2.3, 0.9)
    atoms = numpy.arange(10).reshape(2, 5)
    pairs = ising_pairs(direct, exchange, atoms)
    assert pairs.shape == (2 * 45, 5)
    first = pairs[:45]
    assert numpy.all(numpy.isin(first[:, [1, 3]], atoms[0]))
    opposite = first[first[:, 0] != first[:, 2]]
    numpy.testing.assert_allclose(opposite[:, 4], direct.ravel())


@pytest.mark.unit
@pytest.mark.parametrize("V,dt", [(2.0, 0.02), (-1.5, 0.05), (6.0, 0.3), (-4.0, 0.2)])
@pytest.mark.parametrize("same_spin", [False, True])
def test_pair_hirsch_identity_all_local_states(V, dt, same_spin):
    pair = [(0, 0, 0, 1, V)] if same_spin else [(0, 0, 1, 1, V)]
    prop = ThermofieldDensityDensityDiscrete(numpy.zeros((2, 2, 2)), pair, dt, 1.0)
    for na, nb in product([0, 1], repeat=2):
        hs = 0.5 * numpy.sum(
            numpy.exp(prop.field_log_scalar[0])
            * prop.auxf[0, :, 0] ** na
            * prop.auxf[0, :, 1] ** nb
        )
        assert hs == pytest.approx(numpy.exp(-dt * V * (na * nb - (na + nb) / 2)), rel=2e-14)
    shift = numpy.zeros((2, 2))
    shift[0, 0] += V / 2
    shift[0 if same_spin else 1, 1] += V / 2
    numpy.testing.assert_allclose(
        prop.BK2, numpy.array([expm(-0.5 * dt * numpy.diag(s)) for s in shift]), atol=1e-15
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "mode,importance_guide",
    [("fixed", "same"), ("completion", "same"), ("fixed", "completion")],
)
def test_onsite_pairs_reproduce_spin_hirsch_bitwise(mode, importance_guide):
    runs = []
    for cls in (ThermofieldHubbardDiscrete, ThermofieldDensityDensityDiscrete):
        trial, walkers, prop, driver, pcontrol = _hubbard_objects(cls, mode, importance_guide)
        # The same numbers for the fields and population control of both runs.
        numpy.random.seed(713)
        propagate_population_cp(driver, walkers, trial, prop, 20, pcontrol)
        runs.append((walkers, prop))
    (a, pa), (b, pb) = runs
    numpy.testing.assert_array_equal(pa.BK2, pb.BK2)
    numpy.testing.assert_array_equal(a.field_hist, b.field_hist)
    numpy.testing.assert_array_equal(a.weight, b.weight)
    numpy.testing.assert_array_equal(a.Qmat, b.Qmat)
    numpy.testing.assert_array_equal(a.log_ovlp, b.log_ovlp)
    numpy.testing.assert_array_equal(a.Ga, b.Ga)
    assert pb.field_proposal.startswith("pair_")


@pytest.mark.unit
def test_pair_ratios_and_updates_against_dense_determinants():
    # System params.
    nbasis = 4
    dt = 0.08
    beta = 2.0
    pairs = [(0, 1, 1, 2, 3.0), (1, 0, 1, 3, 2.5), (0, 2, 0, 0, -1.7), (1, 2, 0, 2, -0.8)]
    numpy.random.seed(81)

    h = numpy.random.normal(size=(2, nbasis, nbasis))
    h = (h + h.transpose(0, 2, 1)) * 0.2
    prop = ThermofieldDensityDensityDiscrete(h, pairs, dt, beta)
    lam = expm(-h * 0.7)
    delta = expm(numpy.random.normal(size=(2, nbasis, nbasis)) * 0.2)
    eye = numpy.eye(nbasis)
    g = numpy.array([thermofield_greens_function(lam[s], delta[s]) for s in range(2)])[None]
    overlap = numpy.prod([numpy.linalg.det(eye + delta[s] @ lam[s]) for s in range(2)])
    for field, xi in product(range(len(pairs)), range(2)):
        expected = delta.copy()
        for leg in range(2):
            s, i = prop.pair_spin[field, leg], prop.pair_orb[field, leg]
            expected[s, i, :] *= prop.auxf[field, xi, leg]
        new = numpy.prod([numpy.linalg.det(eye + expected[s] @ lam[s]) for s in range(2)])
        prob = prop._field_probabilities(g, field)[0, xi]
        assert 2 * prob / numpy.exp(prop.field_log_scalar[field, xi]) == pytest.approx(
            new / overlap, abs=2e-13
        )
        updated = g.copy()
        prop._field_green_update(updated, field, numpy.array([xi]))
        exact = numpy.array([thermofield_greens_function(lam[s], expected[s]) for s in range(2)])
        numpy.testing.assert_allclose(updated[0], exact, atol=2e-13)
        factors = numpy.ones((1, 2, nbasis))
        prop._field_factors(factors, numpy.array([0]), field, numpy.array([xi]))
        numpy.testing.assert_allclose(factors[0, :, :, None] * delta, expected, atol=1e-15)


@pytest.mark.unit
def test_two_slice_partition_and_observables_against_fock_space():
    # Sum over every pair field of two slices; no sampling or guide.
    # System params.
    dt = 0.15
    numpy.random.seed(5)
    h1, pairs, atoms = _two_orbital_model()

    assert numpy.any(pairs[:, 4] < 0)
    prop = ThermofieldDensityDensityDiscrete(h1, pairs, dt, 2 * dt)
    obs = DensityDensityObservables(h1, pairs, atoms)
    one, inter, mz, num = _fock_model(h1, pairs)
    shift = sum(
        0.5 * V * (num[int(sa) * 2 + int(i)] + num[int(sb) * 2 + int(j)])
        for sa, i, sb, j, V in pairs
    )
    K = expm(-0.5 * dt * (one + shift))
    transfer = K @ expm(-dt * (inter - shift)) @ K
    rho = transfer @ transfer
    z_exact = numpy.trace(rho)
    exact = [numpy.trace(op @ rho) / z_exact for op in (one + inter, mz @ mz, mz @ mz @ mz @ mz)]
    m, nf = 2, len(pairs)
    eye = numpy.eye(m)
    z = 0.0
    acc = numpy.zeros(obs.size, dtype=complex)
    for fields in product([-1.0, 1.0], repeat=2 * nf):
        D = numpy.tile(eye, (2, 1, 1))
        scalar = 1.0
        for f in numpy.reshape(fields, (2, nf)):
            chosen = (f < 0).astype(int)
            scalar *= numpy.prod(numpy.exp(prop.field_log_scalar[numpy.arange(nf), chosen]))
            D = prop.BK2 @ (prop.slice_diagonal(f[None])[0][:, :, None] * (prop.BK2 @ D))
        trace = scalar * numpy.prod(numpy.linalg.det(eye + D)) / 2 ** (2 * nf)
        z += trace
        acc += trace * obs.measure(eye - numpy.linalg.inv(eye + D))
    assert z == pytest.approx(z_exact, rel=1e-12)
    mean = acc / z
    assert mean[0].real == pytest.approx(exact[0], abs=1e-12)
    assert mean[3].real == pytest.approx(exact[1], abs=1e-12)
    assert mean[4].real == pytest.approx(exact[2], abs=1e-11)


@pytest.mark.unit
def test_observable_moments_of_nonhermitian_gaussian():
    # Counting-statistics moments and pair energies against an explicit Fock-space Gaussian.
    numpy.random.seed(3)
    h1, pairs, atoms = _two_orbital_model(nsites=2)
    m = h1.shape[-1]
    # Non-symmetric generator: a non-Hermitian contraction.
    A = numpy.random.normal(scale=0.5, size=(2, m, m))

    cre = creation_operators(2 * m)
    ann = [c.T for c in cre]
    gen = sum(
        A[s, i, j] * (cre[s * m + i] @ ann[s * m + j])
        for s in range(2)
        for i in range(m)
        for j in range(m)
    )
    rho = expm(gen)
    G = numpy.array([numpy.eye(m) - numpy.linalg.inv(numpy.eye(m) + expm(A[s])) for s in range(2)])
    one, inter, mz, num = _fock_model(h1, pairs)
    local_mz = [sum(num[o] for o in orb) - sum(num[m + o] for o in orb) for orb in atoms]
    local = sum(mz_atom @ mz_atom for mz_atom in local_mz) / len(atoms)
    z = numpy.trace(rho)
    obs = DensityDensityObservables(h1, pairs, atoms)
    out = obs.measure(G)
    expect = [
        numpy.trace(op @ rho) / z
        for op in (one + inter, sum(num), mz, mz @ mz, mz @ mz @ mz @ mz, local)
    ]
    numpy.testing.assert_allclose(out[:6], expect, rtol=1e-10, atol=1e-10)


@pytest.mark.unit
def test_driver_noninteracting_exact_and_wrap_reconstruction():
    # System params.
    beta = 0.6
    dt = 0.05
    numpy.random.seed(5)
    h1, pairs, atoms = _two_orbital_model(nsites=2)
    m = h1.shape[-1]

    zero = pairs.copy()
    zero[:, 4] = 0.0
    ham = Generic(h1e=h1, chol=numpy.zeros((m * m, 1)), ecore=0.0, verbose=False)
    trial = ThermofieldCPTrial(ham, beta, k_trial=h1, guide_time=beta)
    mpi = MPIHandler()
    for table, exact in ((zero, True), (pairs, False)):
        walkers = ThermofieldWalkers(trial, m, 5, mpi)
        prop = ThermofieldDensityDensityDiscrete(
            h1, table, dt, beta, guide_mode="fixed", fixed_guide_time=beta, energy_shift=-2.0
        )
        obs = DensityDensityObservables(h1, table, atoms)
        driver = ThermofieldHubbardAFQMC(
            ham,
            h1,
            trial,
            walkers,
            prop,
            mpi,
            m,
            1,
            nblocks=2,
            pop_control_freq=4,
            wrap_stride=3,
            observables=obs,
        )
        driver.run(verbose=False)
        out = driver.finalise(verbose=False)
        assert out["max_wrapdev"] < 1e-10 and out["max_recdev"] < 1e-10
        assert driver.metadata["hs_type"] == "discrete_pair"
        if exact:
            G = numpy.array(
                [
                    numpy.eye(m) - numpy.linalg.inv(numpy.eye(m) + expm(-beta * h1[s]))
                    for s in range(2)
                ]
            )
            for r in driver.results:
                numpy.testing.assert_allclose(r["numer"] / r["denom"], obs.measure(G), atol=1e-11)
        else:
            mean = out["endpoint"]["mean"]
            assert numpy.isfinite([mean["mz2"], mean["binder"], mean["energy"]]).all()
            assert mean["mz2"] > 0.0


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["fixed", "completion"])
def test_grouped_updates_match_sequential(mode):
    # The same fields, weights and caches to rounding, with forbidden fields and kills.
    # System params.
    beta = 1.0
    dt = 0.05
    nwalkers = 24
    numpy.random.seed(5)
    h1, pairs, _ = _two_orbital_model(nsites=3, t=1.5)
    pairs[:, 4] *= 4.0
    m = h1.shape[-1]
    r = numpy.random.normal(scale=3.0, size=(2, m, m))

    ham = Generic(h1e=h1, chol=numpy.zeros((m * m, 1)), ecore=0.0, verbose=False)
    trial = ThermofieldCPTrial(ham, beta, k_trial=h1 + r + r.transpose(0, 2, 1), guide_time=beta)
    mpi = MPIHandler()
    driver = SimpleNamespace(
        mpi_handler=mpi, params=SimpleNamespace(num_stblz=5, pop_control_freq=10000)
    )
    runs = []
    for grouped in (False, True):
        walkers = ThermofieldWalkers(trial, m, nwalkers, mpi)
        prop = ThermofieldDensityDensityDiscrete(
            h1,
            pairs,
            dt,
            beta,
            record_fields=True,
            guide_mode=mode,
            fixed_guide_time=beta,
            energy_shift=-1.0,
            grouped=grouped,
            group_size=4,
        )
        assert len(prop.field_groups) == 3
        numpy.random.seed(31)
        propagate_population_cp(
            driver, walkers, trial, prop, 20, SimpleNamespace(total_weight=float(nwalkers))
        )
        runs.append((walkers, prop))
    (a, pa), (b, pb) = runs
    assert pa.forbidden_field_count == pb.forbidden_field_count > 0
    assert pa.kill_count == pb.kill_count
    numpy.testing.assert_array_equal(a.weight == 0, b.weight == 0)
    numpy.testing.assert_array_equal(a.field_hist, b.field_hist)
    numpy.testing.assert_allclose(a.weight, b.weight, rtol=1e-10, atol=0)
    alive = a.weight != 0
    numpy.testing.assert_allclose(a.Qmat[alive], b.Qmat[alive], rtol=1e-10, atol=1e-10)
    numpy.testing.assert_allclose(a.log_ovlp[alive], b.log_ovlp[alive], atol=1e-9)
    numpy.testing.assert_allclose(a.Ga[alive], b.Ga[alive], atol=1e-9)


@pytest.mark.unit
def test_rejects_bad_pairs():
    h = numpy.zeros((2, 2, 2))
    with pytest.raises(ValueError):
        ThermofieldDensityDensityDiscrete(h, [(0, 0, 0, 0, 1.0)], 0.1, 1.0)
    with pytest.raises(ValueError):
        ThermofieldDensityDensityDiscrete(h, [(0, 0, 1, 2, 1.0)], 0.1, 1.0)
    with pytest.raises(ValueError):
        ThermofieldDensityDensityDiscrete(
            h,
            [(0, 0, 1, 0, 1.0)],
            0.1,
            1.0,
            guide_mode="fixed",
            fixed_guide_time=1.0,
            importance_guide="completion",
        )


if __name__ == "__main__":
    test_slater_d_matrices_sum_rules_and_t2g_kanamori()
    test_ising_pair_count_and_orbital_rows()
    test_pair_hirsch_identity_all_local_states(-1.5, 0.05, True)
    test_onsite_pairs_reproduce_spin_hirsch_bitwise("fixed", "same")
    test_pair_ratios_and_updates_against_dense_determinants()
    test_two_slice_partition_and_observables_against_fock_space()
    test_observable_moments_of_nonhermitian_gaussian()
    test_driver_noninteracting_exact_and_wrap_reconstruction()
    test_grouped_updates_match_sequential("fixed")
    test_rejects_bad_pairs()
