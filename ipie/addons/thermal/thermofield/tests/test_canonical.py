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
"""Canonical projection tests: coefficient extraction (Test 9), canonical
Green's function, and the Hubbard dimer canonical benchmark (Test 5)."""

import itertools

import numpy
import pytest
import scipy.linalg

from ipie.addons.thermal.thermofield.canonical import (
    canonical_greens_function,
    canonical_overlap_coefficients,
    canonical_overlap_coefficients_fourier,
    canonical_replica_local_energy,
)
from ipie.addons.thermal.thermofield.tests.ed_utils import (
    build_hubbard_dimer_hamiltonian,
    build_noninteracting_hamiltonian,
    canonical_ed,
    creation_operators,
    hubbard_dimer_canonical_exact,
    thermofield_state_vector,
)


def principal_minor_sum(A, N):
    """Brute force [z^N] det(I + z A) = sum of N x N principal minors."""
    M = A.shape[-1]
    total = 0.0 + 0.0j
    for subset in itertools.combinations(range(M), N):
        idx = numpy.ix_(subset, subset)
        total += numpy.linalg.det(A[idx]) if N > 0 else 1.0
    return total


@pytest.mark.unit
@pytest.mark.parametrize("M", [3, 6])
def test_canonical_coefficients_vs_minors_and_fourier(M):
    rng = numpy.random.default_rng(43 + M)
    A = rng.standard_normal((M, M)) + 1j * rng.standard_normal((M, M))
    esp = canonical_overlap_coefficients(A)
    fourier = canonical_overlap_coefficients_fourier(A, nquad=M + 3)
    for N in range(M + 1):
        minors = principal_minor_sum(A, N)
        assert esp[N] == pytest.approx(minors, rel=1e-10, abs=1e-10)
        assert fourier[N] == pytest.approx(esp[N], rel=1e-10, abs=1e-10)


@pytest.mark.unit
def test_canonical_greens_function_vs_fock_space():
    """G_N against a number-projected brute-force Fock-space evaluation."""
    M = 3
    N = 2
    rng = numpy.random.default_rng(53)
    Delta_left = rng.standard_normal((M, M)) + 1j * rng.standard_normal((M, M))
    Delta_right = rng.standard_normal((M, M)) + 1j * rng.standard_normal((M, M))

    G_N = canonical_greens_function(Delta_left, Delta_right, N)

    vL = thermofield_state_vector(Delta_left)
    vR = thermofield_state_vector(Delta_right)
    cre = creation_operators(2 * M)
    ann = [c.conj().T for c in cre]
    dim = 1 << (2 * M)
    # Projector onto N physical particles (count low M bits).
    proj = numpy.zeros(dim)
    for state in range(dim):
        if bin(state & ((1 << M) - 1)).count("1") == N:
            proj[state] = 1.0
    P_N = numpy.diag(proj)

    S_N = vL.conj() @ (P_N @ vR)
    for i in range(M):
        for j in range(M):
            brute = (vL.conj() @ (P_N @ cre[i] @ ann[j] @ vR)) / S_N
            assert G_N[j, i] == pytest.approx(brute, abs=1e-10)


@pytest.mark.unit
def test_canonical_projection_avoids_singular_fourier_points():
    """A contour rotation avoids zeros on the unrotated roots-of-unity grid."""
    M = 2
    singular_z = numpy.exp(2.0j * numpy.pi / (M + 2))
    Delta_left = numpy.eye(M, dtype=numpy.complex128)
    Delta_right = (-1.0 / singular_z) * numpy.eye(M, dtype=numpy.complex128)

    G = canonical_greens_function(Delta_left, Delta_right, nelec=1)
    numpy.testing.assert_allclose(G, 0.5 * numpy.eye(M), atol=1e-12)

    hamiltonian = build_noninteracting_hamiltonian(numpy.eye(M))
    energy, _ = canonical_replica_local_energy(
        hamiltonian,
        numpy.array([Delta_left, Delta_left]),
        numpy.array([Delta_right, Delta_right]),
        (1, 1),
    )
    assert energy == pytest.approx(2.0, abs=1e-12)


@pytest.mark.unit
def test_hubbard_dimer_canonical_ed_formula():
    """Test 5 (deterministic part): the hard-coded dimer formula matches ED
    and approaches E_minus as beta -> inf."""
    t, U = 1.0, 4.0
    hamiltonian = build_hubbard_dimer_hamiltonian(t, U)
    for beta in (0.0, 1.0, 4.0, 16.0):
        exact = hubbard_dimer_canonical_exact(t, U, beta)
        ed = canonical_ed(hamiltonian, (1, 1), beta)
        assert ed == pytest.approx(exact, abs=1e-10)
    eminus = 0.5 * (U - numpy.sqrt(U * U + 16.0 * t * t))
    assert hubbard_dimer_canonical_exact(t, U, 64.0) == pytest.approx(eminus, abs=1e-8)


@pytest.mark.unit
@pytest.mark.parametrize("beta", [0.5, 2.0, 8.0])
def test_canonical_replica_energy_noninteracting_dimer(beta):
    """Deterministic canonical replica estimator on the U = 0 dimer.

    Both replicas at Delta = exp(-beta h / 2): the canonical projected
    replica energy must match the canonical ED energy at N_up = N_down = 1.
    """
    t = 1.0
    hamiltonian = build_hubbard_dimer_hamiltonian(t, 0.0)
    h = hamiltonian.H1[0]
    Delta = scipy.linalg.expm(-0.5 * beta * numpy.array(h, dtype=numpy.complex128))
    Delta = numpy.array([Delta, Delta])

    energy, _ = canonical_replica_local_energy(hamiltonian, Delta, Delta, (1, 1))
    exact = hubbard_dimer_canonical_exact(t, 0.0, beta)
    assert energy.real == pytest.approx(exact, abs=1e-10)
    assert abs(energy.imag) < 1e-10
