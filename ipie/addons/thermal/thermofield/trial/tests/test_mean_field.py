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
import scipy.linalg

from ipie.addons.thermal.thermofield.hamiltonians.hubbard import hopping_matrix, pinning_field
from ipie.addons.thermal.thermofield.trial.hubbard import thermal_ip_scf
from ipie.addons.thermal.thermofield.trial.mean_field import (
    fermi,
    fermi_divided_difference,
    fermi_entropy,
    neel_gap_roots,
    neel_guides,
    rotate_spin,
    solve_multistart,
    spin_block_diag,
    spin_rotation,
    start_from_site_density,
    ThermalHubbardHF,
)
from ipie.addons.thermal.thermofield.utils.testing import creation_operators

# Checkerboard signs of the 2x2 sites, ordered x + 2 y.
ETA = numpy.array([1.0, -1.0, -1.0, 1.0])


def spin_blocks(lx, ly, pin=0.0):
    hop = hopping_matrix(lx, ly)
    v = pinning_field(lx, ly, pin)
    return hop, numpy.array([hop + numpy.diag(v[0]), hop + numpy.diag(v[1])])


def random_hermitian(n, complex_, scale=1.0):
    a = numpy.random.normal(size=(n, n))
    if complex_:
        a = a + 1j * numpy.random.normal(size=(n, n))
    return scale * 0.5 * (a + a.conj().T)


def matrix_fermi(k, beta):
    e, v = numpy.linalg.eigh(k)
    return (v * fermi(e, beta)) @ v.conj().T


def neel_start(problem, sign=1.0, amplitude=0.3, m=None):
    M = problem.M
    if m is None:
        m = numpy.zeros((M, 3))
        m[:, 2] = sign * amplitude * numpy.resize(ETA, M)
    return start_from_site_density(problem, numpy.ones(M), m)


def many_body_hamiltonian(h_so, U):
    # H = sum_ab h_ab c_a^dagger c_b + sum_i U_i (n_iu n_id - (n_iu + n_id) / 2)
    # with the modes grouped by spin.
    nmodes = h_so.shape[0]
    M = nmodes // 2
    cre = creation_operators(nmodes)
    num = [c @ c.T for c in cre]
    H = sum(h_so[a, b] * (cre[a] @ cre[b].T) for a in range(nmodes) for b in range(nmodes))
    U = numpy.broadcast_to(U, (M,))
    for i in range(M):
        H = H + U[i] * (num[i] @ num[i + M] - 0.5 * (num[i] + num[i + M]))
    return H, sum(num)


def gaussian_state(k_so, beta):
    nmodes = k_so.shape[0]
    cre = creation_operators(nmodes)
    K = sum(k_so[a, b] * (cre[a] @ cre[b].T) for a in range(nmodes) for b in range(nmodes))
    rho = scipy.linalg.expm(-beta * K)
    return rho / numpy.trace(rho)


def curvature(problem, variables, dk, t):
    f = [
        problem.evaluate([v + s * d for v, d in zip(variables, dk)]).free_energy
        for s in (-t, 0.0, t)
    ]
    return (f[0] - 2 * f[1] + f[2]) / t**2


@pytest.mark.unit
def test_fermi_helpers_against_direct_formulas():
    beta = 3.7
    eps = numpy.array([-2.1, -0.4, 0.0, 0.0 + 1e-7, 0.35, 1.9])
    f = 1.0 / (1.0 + numpy.exp(beta * eps))
    numpy.testing.assert_allclose(fermi(eps, beta), f, rtol=1e-14, atol=0)
    s = -numpy.sum(f * numpy.log(f) + (1 - f) * numpy.log(1 - f))
    assert abs(fermi_entropy(eps, beta) - s) < 1e-13
    D = fermi_divided_difference(eps, beta)
    for p in range(len(eps)):
        for q in range(len(eps)):
            if abs(eps[p] - eps[q]) > 1e-3:
                ref = (f[p] - f[q]) / (eps[p] - eps[q])
                assert abs(D[p, q] - ref) < 1e-12
    numpy.testing.assert_allclose(numpy.diagonal(D), -beta * f * (1 - f), rtol=1e-12, atol=0)
    # Continuous across a near-degenerate pair.
    assert abs(D[2, 3] - D[2, 2]) < 1e-6
    D_cold = fermi_divided_difference(numpy.array([-300.0, -1e-3, 2e-3, 400.0]), 50.0)
    assert numpy.isfinite(D_cold).all() and (D_cold <= 0).all()


@pytest.mark.unit
def test_frechet_derivative_of_matrix_fermi_function():
    # System params.
    n = 6
    beta = 2.3
    seed = 3
    numpy.random.seed(seed)

    k = random_hermitian(n, True)
    X = random_hermitian(n, True)
    e, v = numpy.linalg.eigh(k)
    L = v @ (fermi_divided_difference(e, beta) * (v.conj().T @ X @ v)) @ v.conj().T
    t = 1e-5
    fd = (matrix_fermi(k + t * X, beta) - matrix_fermi(k - t * X, beta)) / (2 * t)
    assert numpy.max(numpy.abs(L - fd)) < 1e-9


@pytest.mark.unit
def test_ghf_functional_equals_gaussian_expectation():
    # System params.
    M = 4
    beta = 1.3
    mu = 0.4
    seed = 11

    # E_HF[rho] = Tr(rho_G H), S[rho] = -Tr(rho_G ln rho_G) and
    # rho_ab = <c_b^dagger c_a> in the Fock space.
    cre = creation_operators(2 * M)
    for complex_ in (False, True):
        numpy.random.seed(seed)
        h = random_hermitian(2 * M, complex_)
        U = numpy.random.uniform(1.0, 6.0, M)
        k = random_hermitian(2 * M, complex_)
        problem = ThermalHubbardHF(h, U, beta, "ghf", mu=mu)
        rho = matrix_fermi(k, beta)
        energy, _ = problem.energy_fock([rho])
        H, N = many_body_hamiltonian(h, U)
        rho_g = gaussian_state(k, beta)
        assert abs(energy - numpy.trace(rho_g @ H).real) < 1e-11
        lam = numpy.linalg.eigvalsh(rho_g)
        lam = lam[lam > 1e-300]
        entropy = fermi_entropy(numpy.linalg.eigvalsh(k), beta)
        assert abs(entropy + numpy.sum(lam * numpy.log(lam))) < 1e-10
        one_rdm = numpy.array(
            [[numpy.trace(rho_g @ cre[b] @ cre[a].T) for b in range(2 * M)] for a in range(2 * M)]
        )
        assert numpy.max(numpy.abs(one_rdm - rho)) < 1e-12
        assert abs(numpy.trace(rho_g @ N).real - numpy.trace(rho).real) < 1e-12


@pytest.mark.unit
def test_spin_layout_matches_ghf_embedding():
    # System params.
    M = 4
    beta = 0.9
    U = 3.5
    mu = 0.2
    seed = 5

    for complex_ in (False, True):
        numpy.random.seed(seed)
        h = numpy.array([random_hermitian(M, complex_), random_hermitian(M, complex_)])
        k = [random_hermitian(M, complex_), random_hermitian(M, complex_)]
        spin = ThermalHubbardHF(h, U, beta, "uhf", mu=mu)
        ghf = ThermalHubbardHF(spin_block_diag(h), U, beta, "ghf", mu=mu)
        s1 = spin.evaluate(k)
        s2 = ghf.evaluate([spin_block_diag(k)])
        assert abs(s1.free_energy - s2.free_energy) < 1e-12
        assert numpy.max(numpy.abs(spin_block_diag(s1.fock) - s2.fock[0])) < 1e-12
        H, _ = many_body_hamiltonian(spin_block_diag(h), U)
        rho_g = gaussian_state(spin_block_diag(k), beta)
        assert abs(s1.energy - numpy.trace(rho_g @ H).real) < 1e-11


@pytest.mark.unit
def test_fock_is_derivative_of_energy():
    # System params.
    M = 4
    beta = 1.0
    mu = 0.0
    seed = 8

    for layout in ("uhf", "ghf"):
        numpy.random.seed(seed)
        if layout == "ghf":
            h = random_hermitian(2 * M, True)
        else:
            h = numpy.array([random_hermitian(M, True)] * 2)
        problem = ThermalHubbardHF(h, numpy.random.uniform(1, 5, M), beta, layout, mu=mu)
        n = 2 * M if layout == "ghf" else M
        nb = 1 if layout == "ghf" else 2
        rho = [random_hermitian(n, True, 0.3) + 0.5 * numpy.eye(n) for _ in range(nb)]
        X = [random_hermitian(n, True) for _ in range(nb)]
        t = 1e-3
        e_p, _ = problem.energy_fock([r + t * x for r, x in zip(rho, X)])
        e_m, _ = problem.energy_fock([r - t * x for r, x in zip(rho, X)])
        _, fock = problem.energy_fock(rho)
        assert numpy.allclose(fock, [f.conj().T for f in fock], atol=1e-14)
        deriv = sum(numpy.trace(f @ x).real for f, x in zip(fock, X))
        # E is quadratic, so the central difference is exact.
        assert abs((e_p - e_m) / (2 * t) - deriv) < 1e-10


@pytest.mark.unit
def test_gradient_matches_finite_difference():
    # System params.
    lx = 2
    ly = 2
    U = 6.0
    beta = 2.5
    seed = 21

    cases = [
        ("uhf", dict(mu=0.3), False),
        ("uhf", dict(nelec=3.6), False),
        ("uhf", dict(nelec=4.4), True),
        ("rhf", dict(nelec=4.0), False),
        ("rhf", dict(mu=-0.2), True),
        ("ghf", dict(mu=0.1), False),
        ("ghf", dict(nelec=3.7), True),
        ("uhf_ph", dict(mu=0.0, eta=ETA), False),
        ("uhf_ph", dict(mu=0.0, eta=ETA), True),
    ]
    for layout, kwargs, complex_ in cases:
        numpy.random.seed(seed)
        _, h = spin_blocks(lx, ly, pin=0.1)
        if complex_:
            a = random_hermitian(4, True, 0.2)
            # Keep h_dn = -eta h_up^T eta for uhf_ph.
            a = a - (-ETA[:, None] * a.T * ETA[None, :])
            h = numpy.array([h[0] + 0.5 * a, h[1] + 0.5 * (-ETA[:, None] * a.T * ETA[None, :])])
        if layout == "ghf":
            h = spin_block_diag(h)
            h[:4, 4:] += 0.1 * numpy.eye(4)
            h[4:, :4] += 0.1 * numpy.eye(4)
        problem = ThermalHubbardHF(h, U, beta, layout, **kwargs)
        n = 8 if layout == "ghf" else 4
        nv = 2 if layout == "uhf" else 1
        var = [random_hermitian(n, complex_) for _ in range(nv)]
        X = [random_hermitian(n, complex_) for _ in range(nv)]
        state = problem.evaluate(var)
        t = 1e-6
        fp = problem.evaluate([v + t * x for v, x in zip(var, X)]).free_energy
        fm = problem.evaluate([v - t * x for v, x in zip(var, X)]).free_energy
        fd = (fp - fm) / (2 * t)
        assert abs(fd - problem.dot(state.gradient, X)) < 2e-8 * (1 + abs(fd))
        if "nelec" in kwargs:
            assert abs(state.nelec - kwargs["nelec"]) < 1e-12
        assert problem.dot(state.gradient, state.direction) < 0


@pytest.mark.unit
def test_noninteracting_limit_is_exact():
    # System params.
    lx = 2
    ly = 2
    beta = 1.7
    mu = 0.25

    _, h = spin_blocks(lx, ly, pin=0.1)
    problem = ThermalHubbardHF(h, 0.0, beta, "uhf", mu=mu)
    sol = problem.solve([h[0] + 0.3 * numpy.eye(4), h[1]])
    assert sol.converged
    assert numpy.max(numpy.abs(sol.k - (h - mu * numpy.eye(4)))) < 1e-12
    eps = numpy.concatenate([numpy.linalg.eigvalsh(b) for b in h])
    omega = -numpy.sum(numpy.logaddexp(0.0, -beta * (eps - mu))) / beta
    assert abs(sol.free_energy - omega) < 1e-12
    canonical = ThermalHubbardHF(h, 0.0, beta, "uhf", nelec=3.3).solve(list(h))
    assert canonical.converged and abs(canonical.nelec - 3.3) < 1e-12
    assert abs(numpy.sum(fermi(eps - canonical.mu, beta)) - 3.3) < 1e-12


@pytest.mark.unit
def test_matches_legacy_fixed_point_on_pinned_2x2():
    # System params.
    lx = 2
    ly = 2
    U = 8.0
    beta = 4.0
    nelec = 4.0

    # Free energies from an earlier thermal UHF solver.
    references = {1.0: -16.7807767820, -1.0: -17.1574074470}
    hop, h = spin_blocks(lx, ly, pin=0.1)
    v = pinning_field(lx, ly, 0.1)
    for sign, reference in references.items():
        dens0 = numpy.array([0.5 + 0.15 * sign * ETA, 0.5 - 0.15 * sign * ETA])
        k_old, dens, _ = thermal_ip_scf(
            hop, v, beta, nelec, U, dens0=dens0, mix=0.3, max_iter=20000
        )
        problem = ThermalHubbardHF(h, U, beta, "uhf", nelec=nelec)
        sol = problem.solve(neel_start(problem, sign))
        assert sol.converged and sol.iterations < 40
        densities = numpy.array([numpy.diagonal(r).real for r in sol.rho])
        assert numpy.max(numpy.abs(densities - dens)) < 1e-9
        assert numpy.max(numpy.abs(sol.k - k_old)) < 1e-8
        assert abs(sol.free_energy - reference) < 1e-9


@pytest.mark.unit
def test_matches_legacy_fixed_point_on_doped_4x4():
    # System params.
    lx = 4
    ly = 4
    U = 4.0
    beta = 3.0
    nelec = 14.0

    hop, h = spin_blocks(lx, ly, pin=0.1)
    v = pinning_field(lx, ly, 0.1)
    nsites = lx * ly
    x, y = numpy.arange(nsites) % lx, numpy.arange(nsites) // lx
    tex = 0.15 * (-1.0) ** (x + y)
    dens0 = numpy.array([nelec / 32 + tex, nelec / 32 - tex])
    k_old, dens, _ = thermal_ip_scf(hop, v, beta, nelec, U, dens0=dens0, mix=0.3, max_iter=20000)
    problem = ThermalHubbardHF(h, U, beta, "uhf", nelec=nelec)
    m = numpy.zeros((nsites, 3))
    m[:, 2] = 2 * tex
    sol = problem.solve(start_from_site_density(problem, numpy.full(nsites, nelec / nsites), m))
    assert sol.converged
    densities = numpy.array([numpy.diagonal(r).real for r in sol.rho])
    assert numpy.max(numpy.abs(densities - dens)) < 1e-8
    assert numpy.max(numpy.abs(sol.k - k_old)) < 1e-7
    assert abs(sol.nelec - nelec) < 1e-12


@pytest.mark.unit
def test_canonical_and_grand_canonical_agree():
    # System params.
    lx = 2
    ly = 2
    U = 5.0
    beta = 2.0
    nelec = 3.5

    _, h = spin_blocks(lx, ly, pin=0.1)
    canonical = ThermalHubbardHF(h, U, beta, "uhf", nelec=nelec)
    sol = canonical.solve(neel_start(canonical))
    grand = ThermalHubbardHF(h, U, beta, "uhf", mu=sol.mu)
    sol_g = grand.solve(sol.variables)
    assert sol.converged and sol_g.converged and sol_g.iterations <= 1
    assert numpy.max(numpy.abs(sol.rho - sol_g.rho)) < 1e-9
    assert abs(sol_g.free_energy - (sol.free_energy - sol.mu * sol.nelec)) < 1e-9


@pytest.mark.unit
def test_fixed_n_chemical_potential_is_midgap_in_an_insulator():
    # System params.
    lx = 4
    ly = 4
    U = 8.0
    nelec = 16.0

    # At half filling the Neel state is gapped and N(mu) = 16 to double
    # precision across the gap, yet the fixed-N solution must match the
    # particle-hole symmetric mu = 0 one, including k.
    hop = hopping_matrix(lx, ly)
    h = numpy.array([hop, hop])
    nsites = lx * ly
    m = numpy.zeros((nsites, 3))
    m[:, 2] = 0.3 * (-1.0) ** (numpy.arange(nsites) % lx + numpy.arange(nsites) // lx)
    for beta in (10.0, 40.0, 400.0):
        canonical = ThermalHubbardHF(h, U, beta, "uhf", nelec=nelec)
        sol = canonical.solve(start_from_site_density(canonical, numpy.ones(nsites), m))
        grand = ThermalHubbardHF(h, U, beta, "uhf", mu=0.0)
        ref = grand.solve(start_from_site_density(grand, numpy.ones(nsites), m))
        gap = numpy.min(numpy.abs(numpy.linalg.eigvalsh(sol.k[0])))
        if beta == 400.0:
            # The occupations across the gap underflow; only the log balance works.
            assert fermi(gap, beta) == 0.0
        assert sol.converged and ref.converged and abs(sol.mu) < 1e-9
        assert numpy.max(numpy.abs(sol.k - ref.k)) < 1e-8
        lam_n = canonical.stability(sol.variables)[0]
        lam_mu = grand.stability(ref.variables)[0]
        assert numpy.isfinite(lam_n).all() and lam_n[0] > 0.5 and lam_mu[0] > 0.5


@pytest.mark.unit
def test_saturated_start_with_vanishing_gradient_takes_roothaan_steps():
    # System params.
    U = 8.0
    beta = 40.0

    # Two sites without hopping, site energies -6 and +6, started with the high
    # site doubly occupied. Every matrix is diagonal and every level is 1000
    # from mu, so the gradient is exactly zero although the start is far from
    # self-consistent; one Roothaan step reaches the unique minimum (low site
    # doubly occupied, Omega = -12).
    h = numpy.array([numpy.diag([-6.0, 6.0])] * 2)
    start = [numpy.diag([1000.0, -1000.0])] * 2
    for kwargs in (dict(mu=0.0), dict(nelec=2.0)):
        problem = ThermalHubbardHF(h, U, beta, "uhf", **kwargs)
        state = problem.evaluate(start)
        assert problem.dot(state.gradient, state.gradient) == 0.0
        assert problem.roothaan(state)[2] > 0.99
        sol = problem.solve(start)
        assert sol.converged and abs(sol.free_energy + 12.0) < 1e-12
        occupations = numpy.diagonal(sol.rho, axis1=1, axis2=2)
        assert numpy.max(numpy.abs(occupations - [[1.0, 0.0], [1.0, 0.0]])) < 1e-12


@pytest.mark.unit
def test_mean_field_bounds_exact_grand_potential():
    # System params.
    lx = 2
    ly = 2
    beta = 1.5
    mu = 0.3

    # Gibbs-Bogoliubov: Omega_HF >= -T ln Tr exp(-beta (H - mu N)), with
    # equality at U = 0.
    _, h = spin_blocks(lx, ly, pin=0.1)
    for U in (0.0, 4.0, 8.0):
        problem = ThermalHubbardHF(h, U, beta, "uhf", mu=mu)
        sol = problem.solve(neel_start(problem, -1.0))
        H, N = many_body_hamiltonian(spin_block_diag(h), U)
        w = numpy.linalg.eigvalsh(H - mu * N)
        exact = -numpy.logaddexp.reduce(-beta * w) / beta
        assert sol.converged
        if U == 0.0:
            assert abs(sol.free_energy - exact) < 1e-11
        else:
            assert sol.free_energy > exact + 1e-6


@pytest.mark.unit
def test_paramagnet_unstable_neel_stable_and_hessian_curvature():
    # System params.
    lx = 2
    ly = 2
    U = 8.0
    beta = 4.0

    _, h = spin_blocks(lx, ly)
    for kwargs in (dict(mu=0.0), dict(nelec=4.0)):
        rhf = ThermalHubbardHF(h, U, beta, "rhf", **kwargs)
        para = rhf.solve(neel_start(rhf))
        uhf = ThermalHubbardHF(h, U, beta, "uhf", **kwargs)
        lam, modes = uhf.stability(list(para.k))
        assert para.converged and lam[0] < -1.0
        dk, xnorm = modes[0]
        t = 1e-3 / numpy.max(numpy.abs(dk))
        expected = lam[0] * xnorm
        assert abs(curvature(uhf, list(para.k), dk, t) - expected) < 1e-5 * abs(expected)
        neel = uhf.solve(neel_start(uhf))
        lam, modes = uhf.stability(neel.variables)
        assert neel.converged and neel.free_energy < para.free_energy - 1.0 and lam[0] > 0.5
        for j in (0, len(lam) - 1):
            dk, xnorm = modes[j]
            t = 1e-3 / numpy.max(numpy.abs(dk))
            expected = lam[j] * xnorm
            assert abs(curvature(uhf, neel.variables, dk, t) - expected) < 1e-5 * abs(expected)


@pytest.mark.unit
def test_neel_has_goldstone_modes_in_ghf_space():
    # System params.
    lx = 2
    ly = 2
    U = 8.0
    beta = 4.0

    _, h = spin_blocks(lx, ly)
    uhf = ThermalHubbardHF(h, U, beta, "uhf", mu=0.0)
    neel = uhf.solve(neel_start(uhf))
    ghf = ThermalHubbardHF(spin_block_diag(h), U, beta, "ghf", mu=0.0)
    k = spin_block_diag(neel.k)
    state = ghf.evaluate([k])
    assert ghf.roothaan(state)[2] < 1e-10
    lam_real, _ = ghf.stability([k], complex_perturbations=False)
    lam_cplx, _ = ghf.stability([k], complex_perturbations=True)
    assert lam_cplx[0] > -1e-8
    # Real perturbations rotate about y only; complex ones about x and y.
    assert numpy.sum(numpy.abs(lam_real) < 1e-7) == 1
    assert numpy.sum(numpy.abs(lam_cplx) < 1e-7) == 2


@pytest.mark.unit
def test_pinned_metastable_neel_is_local_minimum():
    # System params.
    lx = 2
    ly = 2
    U = 8.0
    beta = 4.0
    nelec = 4.0

    _, h = spin_blocks(lx, ly, pin=0.1)
    problem = ThermalHubbardHF(h, U, beta, "uhf", nelec=nelec)
    lam, _ = problem.stability(problem.solve(neel_start(problem, 1.0)).variables)
    assert lam[0] > 0.5


@pytest.mark.unit
def test_neel_gap_equation_matches_solvers():
    # System params.
    lx = 2
    ly = 2
    beta = 4.0

    hop, h = spin_blocks(lx, ly)
    for U, expected in ((2.0, 0.505628), (4.0, 0.814189), (8.0, 0.941601)):
        roots = neel_gap_roots(hop, ETA, U, beta)
        assert len(roots) == 2 and roots[0] == 0.0 and abs(roots[1] - expected) < 1e-6
        ph = ThermalHubbardHF(h, U, beta, "uhf_ph", mu=0.0, eta=ETA)
        sol = ph.solve(neel_start(ph))
        assert sol.converged and sol.mu == 0.0 and abs(sol.nelec - 4.0) < 1e-12
        m = numpy.diagonal(sol.rho[0]).real - numpy.diagonal(sol.rho[1]).real
        assert numpy.max(numpy.abs(m - roots[1] * ETA)) < 1e-9
        k_c, _ = neel_guides(hop, ETA, U, roots[1])
        assert numpy.max(numpy.abs(sol.k - k_c)) < 1e-9
        assert numpy.max(numpy.abs(sol.k[1] + ETA[:, None] * sol.k[0].T * ETA[None, :])) == 0.0
        uhf = ThermalHubbardHF(h, U, beta, "uhf", nelec=4.0)
        sol_u = uhf.solve(neel_start(uhf))
        assert abs(sol_u.free_energy - sol.free_energy) < 1e-9


@pytest.mark.unit
def test_high_temperature_has_only_paramagnetic_root():
    # System params.
    lx = 2
    ly = 2
    U = 2.0
    beta = 0.5

    hop, h = spin_blocks(lx, ly)
    assert numpy.array_equal(neel_gap_roots(hop, ETA, U, beta), [0.0])
    ph = ThermalHubbardHF(h, U, beta, "uhf_ph", mu=0.0, eta=ETA)
    sol = ph.solve(neel_start(ph))
    assert sol.converged and numpy.max(numpy.abs(sol.k[0] - hop)) < 1e-9
    assert ThermalHubbardHF(h, U, beta, "uhf", mu=0.0).stability(list(sol.k))[0][0] > 0


@pytest.mark.unit
def test_transverse_guide_is_rotated_stationary_point():
    # System params.
    lx = 2
    ly = 2
    U = 8.0
    beta = 4.0

    hop, h = spin_blocks(lx, ly)
    m_s = neel_gap_roots(hop, ETA, U, beta)[1]
    k_c, k_d = neel_guides(hop, ETA, U, m_s)
    delta = 0.5 * U * m_s
    expected = numpy.block([[hop, -delta * numpy.diag(ETA)], [-delta * numpy.diag(ETA), hop]])
    assert numpy.max(numpy.abs(k_d - expected)) < 1e-14
    numpy.testing.assert_allclose(
        numpy.linalg.eigvalsh(k_d), numpy.linalg.eigvalsh(spin_block_diag(k_c)), atol=1e-13
    )
    ghf = ThermalHubbardHF(spin_block_diag(h), U, beta, "ghf", mu=0.0)
    state_d = ghf.evaluate([k_d])
    state_c = ghf.evaluate([spin_block_diag(k_c)])
    assert ghf.roothaan(state_d)[2] < 1e-9
    assert abs(state_d.free_energy - state_c.free_energy) < 1e-12
    assert abs(state_d.nelec - 4.0) < 1e-12
    r = state_d.rho[0]
    mx = 2 * numpy.array([r[i, i + 4].real for i in range(4)])
    assert (
        numpy.max(numpy.abs(mx + m_s * ETA)) < 1e-9 or numpy.max(numpy.abs(mx - m_s * ETA)) < 1e-9
    )


@pytest.mark.unit
def test_ghf_from_random_spin_texture_finds_rotated_neel():
    # System params.
    lx = 2
    ly = 2
    U = 8.0
    beta = 4.0
    seed = 4
    numpy.random.seed(seed)

    hop, h = spin_blocks(lx, ly)
    m_s = neel_gap_roots(hop, ETA, U, beta)[1]
    ghf = ThermalHubbardHF(spin_block_diag(h), U, beta, "ghf", mu=0.0)
    sol = ghf.solve(neel_start(ghf, m=0.3 * numpy.random.normal(size=(4, 3))))
    uhf = ThermalHubbardHF(h, U, beta, "uhf", mu=0.0)
    assert sol.converged
    assert abs(sol.free_energy - uhf.solve(neel_start(uhf)).free_energy) < 1e-9
    r = sol.rho
    m = numpy.array(
        [
            [2 * r[i, i + 4].real, -2 * r[i, i + 4].imag, (r[i, i] - r[i + 4, i + 4]).real]
            for i in range(4)
        ]
    )
    assert numpy.max(numpy.abs(numpy.linalg.norm(m, axis=1) - m_s)) < 1e-8
    # Neighbours along x are antiparallel.
    assert numpy.max(numpy.abs(m + m[[1, 0, 3, 2]])) < 1e-8
    assert ghf.stability(sol.variables, complex_perturbations=True)[0][0] > -1e-8


@pytest.mark.unit
def test_multistart_escapes_the_paramagnetic_saddle():
    # System params.
    lx = 2
    ly = 2
    U = 8.0
    beta = 4.0

    _, h = spin_blocks(lx, ly)
    neel = ThermalHubbardHF(h, U, beta, "uhf_ph", mu=0.0, eta=ETA)
    best = neel.solve(neel_start(neel)).free_energy
    for layout in ("uhf", "ghf"):
        if layout == "ghf":
            problem = ThermalHubbardHF(spin_block_diag(h), U, beta, "ghf", mu=0.0)
            para = [spin_block_diag(h)]
        else:
            problem = ThermalHubbardHF(h, U, beta, "uhf", mu=0.0)
            para = list(h)
        # The paramagnetic start converges to the saddle point.
        assert problem.stability(problem.solve(para).variables)[0][0] < -1.0
        results = solve_multistart(problem, {"paramagnetic": para, "neel": neel_start(problem)})
        for label, sol, lam in results:
            assert sol.converged and lam > -1e-8 and abs(sol.free_energy - best) < 1e-9, label


@pytest.mark.unit
def test_spin_rotation_matrix():
    R = spin_rotation(0.5 * numpy.pi, [0, 1, 0])
    sz, sx = numpy.diag([1.0, -1.0]), numpy.array([[0.0, 1.0], [1.0, 0.0]])
    assert numpy.max(numpy.abs(R @ sz @ R.conj().T - sx)) < 1e-15
    k = numpy.arange(16.0).reshape(4, 4)
    k = k + k.T
    assert numpy.max(numpy.abs(rotate_spin(k, numpy.eye(2)) - k)) == 0.0


@pytest.mark.unit
def test_invalid_inputs_fail_fast():
    hop, h = spin_blocks(2, 2, pin=0.1)
    with pytest.raises(ValueError):
        ThermalHubbardHF(h, 4.0, 1.0, "rohf", mu=0.0)
    with pytest.raises(ValueError):
        ThermalHubbardHF(h, 4.0, 1.0, "uhf", mu=0.0, nelec=4.0)
    with pytest.raises(ValueError):
        ThermalHubbardHF(h, 4.0, 1.0, "uhf_ph", mu=0.1, eta=ETA)
    h_broken = numpy.array([h[0], h[0] + numpy.diag([0.1, 0, 0, 0])])
    with pytest.raises(ValueError):
        ThermalHubbardHF(h_broken, 4.0, 1.0, "uhf_ph", mu=0.0, eta=ETA)
    with pytest.raises(ValueError):
        neel_gap_roots(hop + numpy.diag([0.2, 0, 0, 0]), ETA, 4.0, 1.0)


if __name__ == "__main__":
    test_fermi_helpers_against_direct_formulas()
    test_frechet_derivative_of_matrix_fermi_function()
    test_ghf_functional_equals_gaussian_expectation()
    test_spin_layout_matches_ghf_embedding()
    test_fock_is_derivative_of_energy()
    test_gradient_matches_finite_difference()
    test_noninteracting_limit_is_exact()
    test_matches_legacy_fixed_point_on_pinned_2x2()
    test_matches_legacy_fixed_point_on_doped_4x4()
    test_canonical_and_grand_canonical_agree()
    test_fixed_n_chemical_potential_is_midgap_in_an_insulator()
    test_saturated_start_with_vanishing_gradient_takes_roothaan_steps()
    test_mean_field_bounds_exact_grand_potential()
    test_paramagnet_unstable_neel_stable_and_hessian_curvature()
    test_neel_has_goldstone_modes_in_ghf_space()
    test_pinned_metastable_neel_is_local_minimum()
    test_neel_gap_equation_matches_solvers()
    test_high_temperature_has_only_paramagnetic_root()
    test_transverse_guide_is_rotated_stationary_point()
    test_ghf_from_random_spin_texture_finds_rotated_neel()
    test_multistart_escapes_the_paramagnetic_saddle()
    test_spin_rotation_matrix()
    test_invalid_inputs_fail_fast()
