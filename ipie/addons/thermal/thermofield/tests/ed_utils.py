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
"""Brute-force Fock-space and analytic reference utilities (tests only)."""

import numpy
import scipy.linalg


def creation_operators(nmodes):
    """Jordan-Wigner creation matrices on the 2^nmodes Fock space.

    Basis state index encodes occupations bitwise: bit p of the index is the
    occupation of mode p, and the JW phase for mode p is (-1)^(number of
    occupied modes q < p).
    """
    dim = 1 << nmodes
    cre = []
    for p in range(nmodes):
        mat = numpy.zeros((dim, dim))
        for state in range(dim):
            if not (state >> p) & 1:
                phase = (-1) ** bin(state & ((1 << p) - 1)).count("1")
                mat[state | (1 << p), state] = phase
        cre.append(mat)
    return cre


def thermofield_state_vector(Delta):
    """|Phi(Delta)> = exp(c^dag Delta \\tilde{c}^dag)|0 \\tilde{0}> explicitly.

    Doubled Fock space of 2M modes ordered [c_0..c_{M-1}, t_0..t_{M-1}].
    """
    M = Delta.shape[0]
    cre = creation_operators(2 * M)
    dim = 1 << (2 * M)
    op = numpy.zeros((dim, dim), dtype=numpy.complex128)
    for i in range(M):
        for j in range(M):
            op += Delta[i, j] * (cre[i] @ cre[M + j])
    vec = numpy.zeros(dim, dtype=numpy.complex128)
    vec[0] = 1.0
    return scipy.linalg.expm(op) @ vec


def build_spin_hamiltonian_matrix(hamiltonian):
    """Many-body H (and N) from an ipie real-Cholesky Hamiltonian.

    Modes ordered [alpha_0..alpha_{M-1}, beta_0..beta_{M-1}].  Uses the same
    operator identity that underlies the ipie HS convention:

        H = ecore + sum_s c_s^dag h1e_mod[s] c_s + 1/2 sum_n vhat_n^2,
        vhat_n = sum_s c_s^dag L_n c_s,     h1e_mod = h1e - v0,
        v0 = 1/2 sum_n L_n L_n^dag,

    which reproduces H = ecore + sum h_ij c^dag c + 1/2 sum (ij|kl)
    c_i^dag c_k^dag c_l c_j with (ij|kl) = sum_n L_ij,n L_kl,n.
    """
    nbasis = hamiltonian.nbasis
    cre = creation_operators(2 * nbasis)
    ann = [c.conj().T for c in cre]
    dim = 1 << (2 * nbasis)

    def one_body(mat, spin):
        off = spin * nbasis
        out = numpy.zeros((dim, dim), dtype=numpy.complex128)
        for i in range(nbasis):
            for j in range(nbasis):
                if mat[i, j] != 0.0:
                    out += mat[i, j] * (cre[off + i] @ ann[off + j])
        return out

    H = hamiltonian.ecore * numpy.eye(dim, dtype=numpy.complex128)
    for s in range(2):
        H += one_body(hamiltonian.h1e_mod[s], s)

    chol = numpy.array(hamiltonian.chol).reshape(nbasis, nbasis, -1)
    for n in range(chol.shape[-1]):
        vhat = one_body(chol[:, :, n], 0) + one_body(chol[:, :, n], 1)
        H += 0.5 * (vhat @ vhat)

    N = numpy.zeros((dim, dim), dtype=numpy.complex128)
    for s in range(2):
        N += one_body(numpy.eye(nbasis), s)
    return H, N


def grand_canonical_ed(hamiltonian, mu, beta):
    """Grand-canonical thermal averages by exact diagonalization.

    Returns
    -------
    (energy, nav, logZ) : tuple of floats
        Internal energy Tr[H rho], particle number, and log Tr e^{-beta(H - mu N)}.
    """
    H, N = build_spin_hamiltonian_matrix(hamiltonian)
    K = H - mu * N
    evals, evecs = numpy.linalg.eigh(K)
    w = numpy.exp(-beta * (evals - evals.min()))
    Hd = evecs.conj().T @ H @ evecs
    Nd = evecs.conj().T @ N @ evecs
    Z = numpy.sum(w)
    energy = numpy.sum(w * numpy.diag(Hd).real) / Z
    nav = numpy.sum(w * numpy.diag(Nd).real) / Z
    logZ = numpy.log(Z) - beta * evals.min()
    return energy, nav, logZ


def canonical_ed(hamiltonian, nelec, beta):
    """Canonical thermal energy at fixed (N_alpha, N_beta) by ED."""
    nbasis = hamiltonian.nbasis
    H, _ = build_spin_hamiltonian_matrix(hamiltonian)
    dim = H.shape[0]
    mask_a = (1 << nbasis) - 1
    idx = [
        s
        for s in range(dim)
        if bin(s & mask_a).count("1") == nelec[0]
        and bin((s >> nbasis) & mask_a).count("1") == nelec[1]
    ]
    Hs = H[numpy.ix_(idx, idx)]
    evals = numpy.linalg.eigvalsh(Hs)
    w = numpy.exp(-beta * (evals - evals.min()))
    return numpy.sum(w * evals) / numpy.sum(w)


def noninteracting_exact(h, mu, beta):
    """f, N, E for one spin species: f = (I + e^{beta (h - mu)})^{-1}.

    Computed in the eigenbasis (0.5 * (1 - tanh(beta e / 2))), which stays
    accurate at large beta where the naive inverse loses the small scales.
    """
    evals, evecs = numpy.linalg.eigh(h - mu * numpy.eye(h.shape[-1]))
    occ = 0.5 * (1.0 - numpy.tanh(0.5 * beta * evals))
    f = evecs @ numpy.diag(occ) @ evecs.conj().T
    return f, numpy.trace(f).real, numpy.trace(h @ f).real


def hubbard_atom_exact(eps, U, mu, beta):
    """Analytic grand-canonical Hubbard atom (hard-coded, per test plan).

    H = eps (n_up + n_down) + U n_up n_down, K = H - mu (n_up + n_down).
    """
    a = numpy.exp(-beta * (eps - mu))
    b = numpy.exp(-beta * U)
    Z = 1.0 + 2.0 * a + a * a * b
    nav = (2.0 * a + 2.0 * a * a * b) / Z
    double_occ = a * a * b / Z
    energy = (2.0 * eps * a + (2.0 * eps + U) * a * a * b) / Z
    return {"Z": Z, "nav": nav, "double_occ": double_occ, "energy": energy}


def hubbard_dimer_canonical_exact(t, U, beta):
    """Analytic canonical Hubbard dimer, N_up = N_down = 1 (hard-coded)."""
    e0 = 0.0
    e1 = U
    eminus = 0.5 * (U - numpy.sqrt(U * U + 16.0 * t * t))
    eplus = 0.5 * (U + numpy.sqrt(U * U + 16.0 * t * t))
    evals = numpy.array([e0, e1, eminus, eplus])
    w = numpy.exp(-beta * (evals - evals.min()))
    return numpy.sum(w * evals) / numpy.sum(w)


def build_hubbard_atom_hamiltonian(eps, U):
    """Hubbard atom as an ipie GenericRealChol (requires U >= 0)."""
    from ipie.hamiltonians.generic import GenericRealChol

    assert U >= 0.0, "Attractive U is not representable with a real Cholesky factor."
    h1e = numpy.array([[[eps]], [[eps]]])
    chol = numpy.sqrt(U) * numpy.ones((1, 1))
    return GenericRealChol(h1e, chol, ecore=0.0)


def build_hubbard_dimer_hamiltonian(t, U):
    """Hubbard dimer as an ipie GenericRealChol (requires U >= 0)."""
    from ipie.hamiltonians.generic import GenericRealChol

    assert U >= 0.0
    h = numpy.array([[0.0, -t], [-t, 0.0]])
    h1e = numpy.array([h, h])
    chol = numpy.zeros((4, 2))
    chol[0, 0] = numpy.sqrt(U)  # sqrt(U) * n_0 vector, (00) entry.
    chol[3, 1] = numpy.sqrt(U)  # sqrt(U) * n_1 vector, (11) entry.
    return GenericRealChol(h1e, chol, ecore=0.0)


def build_noninteracting_hamiltonian(h):
    """Spin-symmetric noninteracting ipie Hamiltonian (one zero Cholesky vector)."""
    from ipie.hamiltonians.generic import GenericRealChol

    nbasis = h.shape[-1]
    h1e = numpy.array([h, h])
    chol = numpy.zeros((nbasis * nbasis, 1))
    return GenericRealChol(h1e, chol, ecore=0.0)


def build_random_real_hamiltonian(nbasis, nchol, scale=0.3, ecore=0.0, seed=7):
    """Random real symmetric one-body + random symmetric Cholesky vectors."""
    from ipie.hamiltonians.generic import GenericRealChol

    rng = numpy.random.default_rng(seed)
    h = rng.standard_normal((nbasis, nbasis))
    h = 0.5 * (h + h.T)
    chol = numpy.zeros((nbasis * nbasis, nchol))
    for n in range(nchol):
        L = scale * rng.standard_normal((nbasis, nbasis))
        L = 0.5 * (L + L.T)
        chol[:, n] = L.ravel()
    return GenericRealChol(numpy.array([h, h]), chol, ecore=ecore)
