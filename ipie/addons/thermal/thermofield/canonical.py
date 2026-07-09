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
"""Canonical (fixed particle number) projection utilities (phase 2).

For a thermofield overlap S = det(I + A), A = Lambda Delta, the canonical
overlap at particle number N is the coefficient

    S_N = [z^N] det(I + z A) = e_N(lambda_1, ..., lambda_M),

the N-th elementary symmetric polynomial of the eigenvalues of A.  Two
extraction methods are implemented (recursion, and Fourier projection over
K >= M + 1 unit-circle phases; the latter is exact for polynomials of degree
<= M and doubles as a cross-check in tests).

Because inserting z^{N_op} into a thermofield Gaussian expectation keeps it
Gaussian, any matrix element <Phi_L| z^{N_op} O |Phi_R> is Wick-evaluable
with the z-dependent transition Green's function

    G(z) = z Delta_R (I + z A)^{-1} Delta_L^dagger,

and, since the Fock space is finite, S(z) * Wick(G(z)) is a polynomial in z
of degree <= M whose coefficients can be extracted exactly by Fourier
quadrature.  This is used for the canonical replica local energy.
"""

import numpy

from ipie.addons.thermal.estimators.generic import local_energy_generic_cholesky


def _well_conditioned_fourier_grid(matrices, nquad):
    """Choose an exact rotated Fourier grid away from overlap zeros.

    Rotating all roots of unity by a common phase leaves coefficient
    extraction exact.  Try more offsets than there are possible determinant
    zeros and retain the grid whose inverse factors are best conditioned.
    """
    num_offsets = 1 + sum(A.shape[-1] for A in matrices)
    best_condition = numpy.inf
    best_grid = None
    for offset_index in range(num_offsets):
        offset = 2.0 * numpy.pi * (offset_index + 0.5) / (nquad * num_offsets)
        zs = numpy.exp(1.0j * (2.0 * numpy.pi * numpy.arange(nquad) / nquad + offset))
        worst_condition = 0.0
        for A in matrices:
            I = numpy.eye(A.shape[-1])
            worst_condition = max(
                worst_condition,
                max(numpy.linalg.cond(I + z * A) for z in zs),
            )
        if numpy.isfinite(worst_condition) and worst_condition < best_condition:
            best_condition = worst_condition
            best_grid = zs
    if best_grid is None:
        raise numpy.linalg.LinAlgError("No nonsingular Fourier projection grid found.")
    return best_grid


def elementary_symmetric_polynomials(eigenvalues):
    """All elementary symmetric polynomials e_0, ..., e_M of the inputs.

    Uses the standard recursion e_k^{(n)} = e_k^{(n-1)} + lam_n e_{k-1}^{(n-1)},
    so that det(I + z A) = sum_k e_k z^k for eigenvalues of A.

    Returns
    -------
    esp : :class:`numpy.ndarray`
        Shape (M + 1,), complex.
    """
    lams = numpy.asarray(eigenvalues)
    M = lams.shape[0]
    esp = numpy.zeros(M + 1, dtype=numpy.complex128)
    esp[0] = 1.0
    for n in range(M):
        esp[1 : n + 2] = esp[1 : n + 2] + lams[n] * esp[0 : n + 1].copy()
    return esp


def canonical_overlap_coefficients(A):
    """Coefficients of det(I + z A) in z via eigenvalue recursion.

    [z^N] det(I + z A) equals the sum of all N x N principal minors of A.
    """
    lams = numpy.linalg.eigvals(A)
    return elementary_symmetric_polynomials(lams)


def canonical_overlap_coefficients_fourier(A, nquad=None):
    """Coefficients of det(I + z A) via Fourier projection on the unit circle.

    Exact for K = nquad >= M + 1 phases; intended as an independent
    cross-check of the recursion in tests.
    """
    M = A.shape[-1]
    if nquad is None:
        nquad = M + 1
    assert nquad >= M + 1
    coeffs = numpy.zeros(nquad, dtype=numpy.complex128)
    I = numpy.eye(M)
    for k in range(nquad):
        z = numpy.exp(2.0j * numpy.pi * k / nquad)
        det = numpy.linalg.det(I + z * A)
        for N in range(min(nquad, M + 1)):
            coeffs[N] += det * z ** (-N) / nquad
    return coeffs[: M + 1]


def canonical_greens_function(Delta_left, Delta_right, nelec):
    """Canonical transition Green's function at fixed particle number.

    With A = Delta_left^dagger Delta_right and S(z) = det(I + z A),

        G_N = [z^N] S(z) G(z) / [z^N] S(z),
        G(z) = z Delta_right (I + z A)^{-1} Delta_left^dagger,

    extracted by exact Fourier quadrature (S(z) G(z) is a polynomial of
    degree <= M in z).

    Returns
    -------
    G_N : :class:`numpy.ndarray`
        (M, M), same physical convention as
        :func:`ipie.addons.thermal.thermofield.gaussian.thermofield_greens_function`.
    """
    Lam = Delta_left.conj().T
    A = Lam @ Delta_right
    M = A.shape[-1]
    nquad = M + 2
    I = numpy.eye(M)
    SG_N = numpy.zeros((M, M), dtype=numpy.complex128)
    S_N = 0.0 + 0.0j
    for z in _well_conditioned_fourier_grid([A], nquad):
        Sz = numpy.linalg.det(I + z * A)
        Gz = z * Delta_right @ numpy.linalg.inv(I + z * A) @ Lam
        SG_N += Sz * Gz * z ** (-nelec) / nquad
        S_N += Sz * z ** (-nelec) / nquad
    return SG_N / S_N


def canonical_replica_local_energy(hamiltonian, Delta_left, Delta_right, nelec):
    """Canonical-projected replica local energy and overlap for one pair.

    Projects each spin sector onto fixed particle number nelec = (na, nb) by
    2D Fourier quadrature over (z_a, z_b).  For each quadrature point the
    grand-canonical Wick local energy is evaluated with the z-dependent
    transition Green's functions, and S_a(z_a) S_b(z_b) E_loc(z_a, z_b) is a
    polynomial in both fugacities, so the projection is exact.

    Parameters
    ----------
    hamiltonian :
        ipie Hamiltonian (local energies use hamiltonian.H1, i.e. H).
    Delta_left, Delta_right : :class:`numpy.ndarray`
        Walker matrices, shape (2, M, M) (no gauge scale; intended for
        small-system tests and prototype canonical measurements).
    nelec : tuple(int, int)
        (N_alpha, N_beta).

    Returns
    -------
    energy_N : complex
        Canonical local energy [z^N]{S E_loc} / [z^N]{S}.
    ovlp_N : complex
        Canonical replica overlap [z^N] S_a(z_a) S_b(z_b).
    """
    M = hamiltonian.nbasis
    nquad = M + 2
    I = numpy.eye(M)
    Lam = [Delta_left[s].conj().T for s in range(2)]
    A = [Lam[s] @ Delta_right[s] for s in range(2)]

    zs = _well_conditioned_fourier_grid(A, nquad)
    numer = 0.0 + 0.0j
    denom = 0.0 + 0.0j
    for ka in range(nquad):
        za = zs[ka]
        Sa = numpy.linalg.det(I + za * A[0])
        Ga = za * Delta_right[0] @ numpy.linalg.inv(I + za * A[0]) @ Lam[0]
        for kb in range(nquad):
            zb = zs[kb]
            Sb = numpy.linalg.det(I + zb * A[1])
            Gb = zb * Delta_right[1] @ numpy.linalg.inv(I + zb * A[1]) @ Lam[1]
            P = numpy.array([Ga.T, Gb.T])
            eloc = local_energy_generic_cholesky(hamiltonian, P)[0]
            proj = za ** (-nelec[0]) * zb ** (-nelec[1]) / nquad**2
            numer += Sa * Sb * eloc * proj
            denom += Sa * Sb * proj
    return numer / denom, denom
