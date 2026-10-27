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
import scipy.linalg

from ipie.addons.thermal.thermofield.propagation.phaseless_thermofield import ThermofieldPhaseless
from ipie.addons.thermal.thermofield.trial.thermofield import ThermofieldThermalTrial
from ipie.addons.thermal.thermofield.walkers.uhf_walkers import ThermofieldWalkers
from ipie.hamiltonians.generic import GenericRealChol


def creation_operators(nmodes):
    """Jordan-Wigner creation operators on the Fock space of `nmodes` modes.

    Bit p of a basis-state index is the occupation of mode p, and the phase of
    mode p is (-1) to the number of occupied modes q < p.

    Parameters
    ----------
    nmodes : int
        Number of modes.

    Returns
    -------
    cre : list of :class:`numpy.ndarray`
        Creation operator of every mode, shape (2**nmodes, 2**nmodes).
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
    r"""Thermofield state :math:`\exp(c^\dagger \Delta \tilde{c}^\dagger)|0 \tilde{0}\rangle`.

    Parameters
    ----------
    Delta : :class:`numpy.ndarray`
        Walker matrix, shape (nbasis, nbasis).

    Returns
    -------
    vec : :class:`numpy.ndarray`
        State in the doubled Fock space of 2 nbasis modes ordered
        [c_0, ..., c_{nbasis-1}, t_0, ..., t_{nbasis-1}], shape (4**nbasis,).
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
    r"""Many-body Hamiltonian and number operator of an ipie real-Cholesky Hamiltonian.

    Modes are ordered [alpha_0, ..., alpha_{M-1}, beta_0, ..., beta_{M-1}]. The
    Hamiltonian is built from the operator identity behind the ipie
    Hubbard-Stratonovich convention,

    .. math::

        H = E_0 + \sum_s c_s^\dagger (h_s - v_0) c_s + \frac{1}{2} \sum_n \hat{v}_n^2,
        \qquad \hat{v}_n = \sum_s c_s^\dagger L_n c_s,
        \qquad v_0 = \frac{1}{2} \sum_n L_n L_n^\dagger,

    which equals the usual two-body Hamiltonian with
    :math:`(ij|kl) = \sum_n L_{ij,n} L_{kl,n}`.

    Parameters
    ----------
    hamiltonian : :class:`ipie.hamiltonians.generic.GenericRealChol`
        Hamiltonian.

    Returns
    -------
    H : :class:`numpy.ndarray`
        Hamiltonian matrix, shape (4**nbasis, 4**nbasis).
    N : :class:`numpy.ndarray`
        Number operator, shape (4**nbasis, 4**nbasis).
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

    Parameters
    ----------
    hamiltonian : :class:`ipie.hamiltonians.generic.GenericRealChol`
        Hamiltonian.
    mu : float
        Chemical potential.
    beta : float
        Inverse temperature.

    Returns
    -------
    energy : float
        Internal energy.
    nav : float
        Average particle number.
    logZ : float
        Log of the grand partition function.
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
    """Canonical thermal energy at fixed spin populations by exact diagonalization.

    Parameters
    ----------
    hamiltonian : :class:`ipie.hamiltonians.generic.GenericRealChol`
        Hamiltonian.
    nelec : tuple of int
        Number of alpha and beta electrons.
    beta : float
        Inverse temperature.

    Returns
    -------
    energy : float
        Internal energy.
    """
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
    r"""Fermi function, particle number and energy of one noninteracting spin species.

    The occupations 0.5 (1 - tanh(beta e / 2)) are evaluated in the eigenbasis,
    which stays accurate at large beta.

    Parameters
    ----------
    h : :class:`numpy.ndarray`
        One-body Hamiltonian, shape (nbasis, nbasis).
    mu : float
        Chemical potential.
    beta : float
        Inverse temperature.

    Returns
    -------
    f : :class:`numpy.ndarray`
        Fermi function :math:`(I + e^{\beta (h - \mu)})^{-1}`, shape (nbasis, nbasis).
    nav : float
        Particle number.
    energy : float
        Energy Tr(h f).
    """
    evals, evecs = numpy.linalg.eigh(h - mu * numpy.eye(h.shape[-1]))
    occ = 0.5 * (1.0 - numpy.tanh(0.5 * beta * evals))
    f = evecs @ numpy.diag(occ) @ evecs.conj().T
    return f, numpy.trace(f).real, numpy.trace(h @ f).real


def hubbard_atom_exact(eps, U, mu, beta):
    """Grand-canonical Hubbard atom, H = eps (n_up + n_down) + U n_up n_down.

    Parameters
    ----------
    eps : float
        Site energy.
    U : float
        On-site interaction.
    mu : float
        Chemical potential.
    beta : float
        Inverse temperature.

    Returns
    -------
    exact : dict
        Partition function "Z", particle number "nav", double occupancy
        "double_occ" and internal energy "energy".
    """
    a = numpy.exp(-beta * (eps - mu))
    b = numpy.exp(-beta * U)
    Z = 1.0 + 2.0 * a + a * a * b
    nav = (2.0 * a + 2.0 * a * a * b) / Z
    double_occ = a * a * b / Z
    energy = (2.0 * eps * a + (2.0 * eps + U) * a * a * b) / Z
    return {"Z": Z, "nav": nav, "double_occ": double_occ, "energy": energy}


def hubbard_atom_member_exact(eps, U, mu, beta, eta):
    r"""Hubbard atom energy and particle number with a one-body guide over a time eta.

    Evaluates :math:`\mathrm{Tr}[e^{-\eta K_T} O e^{-(\beta - \eta) K}] / \mathrm{Tr}[\ldots]`
    with :math:`K = H - \mu N` and the one-body guide :math:`K_T = (\epsilon - \mu) N`,
    both diagonal in the occupation basis (empty, up, down, double).

    Parameters
    ----------
    eps : float
        Site energy.
    U : float
        On-site interaction.
    mu : float
        Chemical potential.
    beta : float
        Inverse temperature.
    eta : float
        Imaginary time spent under the guide.

    Returns
    -------
    energy : float
        Energy estimate.
    nav : float
        Particle number estimate.
    """
    n = numpy.array([0.0, 1.0, 1.0, 2.0])
    E = numpy.array([0.0, eps, eps, 2.0 * eps + U])
    w = numpy.exp(-eta * (eps - mu) * n - (beta - eta) * (E - mu * n))
    return (w * E).sum() / w.sum(), (w * n).sum() / w.sum()


def hubbard_dimer_canonical_exact(t, U, beta):
    """Canonical Hubbard dimer energy with one alpha and one beta electron.

    Parameters
    ----------
    t : float
        Hopping.
    U : float
        On-site interaction.
    beta : float
        Inverse temperature.

    Returns
    -------
    energy : float
        Internal energy.
    """
    e0 = 0.0
    e1 = U
    eminus = 0.5 * (U - numpy.sqrt(U * U + 16.0 * t * t))
    eplus = 0.5 * (U + numpy.sqrt(U * U + 16.0 * t * t))
    evals = numpy.array([e0, e1, eminus, eplus])
    w = numpy.exp(-beta * (evals - evals.min()))
    return numpy.sum(w * evals) / numpy.sum(w)


def build_hubbard_atom_hamiltonian(eps, U):
    """Hubbard atom as an ipie real-Cholesky Hamiltonian.

    Parameters
    ----------
    eps : float
        Site energy.
    U : float
        On-site interaction, U >= 0.

    Returns
    -------
    hamiltonian : :class:`ipie.hamiltonians.generic.GenericRealChol`
        Hamiltonian.
    """
    assert U >= 0.0, "Attractive U is not representable with a real Cholesky factor."
    h1e = numpy.array([[[eps]], [[eps]]])
    chol = numpy.sqrt(U) * numpy.ones((1, 1))
    return GenericRealChol(h1e, chol, ecore=0.0)


def build_hubbard_dimer_hamiltonian(t, U):
    """Hubbard dimer as an ipie real-Cholesky Hamiltonian.

    Parameters
    ----------
    t : float
        Hopping.
    U : float
        On-site interaction, U >= 0.

    Returns
    -------
    hamiltonian : :class:`ipie.hamiltonians.generic.GenericRealChol`
        Hamiltonian.
    """
    assert U >= 0.0
    h = numpy.array([[0.0, -t], [-t, 0.0]])
    h1e = numpy.array([h, h])
    chol = numpy.zeros((4, 2))
    # Vectors sqrt(U) n_0 and sqrt(U) n_1: entries (00) and (11).
    chol[0, 0] = numpy.sqrt(U)
    chol[3, 1] = numpy.sqrt(U)
    return GenericRealChol(h1e, chol, ecore=0.0)


def build_noninteracting_hamiltonian(h):
    """Spin-symmetric noninteracting ipie Hamiltonian with one zero Cholesky vector.

    Parameters
    ----------
    h : :class:`numpy.ndarray`
        One-body Hamiltonian, shape (nbasis, nbasis).

    Returns
    -------
    hamiltonian : :class:`ipie.hamiltonians.generic.GenericRealChol`
        Hamiltonian.
    """
    nbasis = h.shape[-1]
    h1e = numpy.array([h, h])
    chol = numpy.zeros((nbasis * nbasis, 1))
    return GenericRealChol(h1e, chol, ecore=0.0)


def build_random_real_hamiltonian(nbasis, nchol, scale=0.3, ecore=0.0, seed=7):
    """Random real symmetric one-body term and random symmetric Cholesky vectors.

    Parameters
    ----------
    nbasis : int
        Number of basis functions.
    nchol : int
        Number of Cholesky vectors.
    scale : float
        Scale of the Cholesky vectors. Default 0.3.
    ecore : float
        Constant energy. Default 0.0.
    seed : int
        Seed of the private random generator; the global numpy stream is not
        touched. Default 7.

    Returns
    -------
    hamiltonian : :class:`ipie.hamiltonians.generic.GenericRealChol`
        Hamiltonian.
    """
    rng = numpy.random.default_rng(seed)
    h = rng.standard_normal((nbasis, nbasis))
    h = 0.5 * (h + h.T)
    chol = numpy.zeros((nbasis * nbasis, nchol))
    for n in range(nchol):
        L = scale * rng.standard_normal((nbasis, nbasis))
        L = 0.5 * (L + L.T)
        chol[:, n] = L.ravel()
    return GenericRealChol(numpy.array([h, h]), chol, ecore=ecore)


def build_noninteracting_test_case_handlers(
    nbasis: int,
    mu: float,
    beta: float,
    timestep: float,
    nwalkers: int = 2,
    seed: int = 11,
):
    """Random noninteracting system with its thermofield trial, walkers and propagator.

    Seeds the global numpy stream with `seed` and draws a random real
    symmetric one-body Hamiltonian.

    Parameters
    ----------
    nbasis : int
        Number of basis functions.
    mu : float
        Chemical potential.
    beta : float
        Inverse temperature.
    timestep : float
        Time step.
    nwalkers : int
        Number of walkers. Default 2.
    seed : int
        Random seed. Default 11.

    Returns
    -------
    objs : dict
        One-body matrix "h1e", "hamiltonian", "trial", "walkers" and "propagator".
    """
    numpy.random.seed(seed)
    h1e = numpy.random.standard_normal((nbasis, nbasis))
    h1e = 0.5 * (h1e + h1e.T)
    hamiltonian = build_noninteracting_hamiltonian(h1e)
    trial = ThermofieldThermalTrial(hamiltonian, beta=beta, mu=mu)
    walkers = ThermofieldWalkers(trial, nbasis, nwalkers)
    propagator = ThermofieldPhaseless(timestep, mu)
    propagator.build(hamiltonian, trial=trial, walkers=walkers)
    return {
        "h1e": h1e,
        "hamiltonian": hamiltonian,
        "trial": trial,
        "walkers": walkers,
        "propagator": propagator,
    }
