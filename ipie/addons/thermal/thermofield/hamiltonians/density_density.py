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

# Wannier90 order of the real d orbitals (l = 2, m = 1..5).
D_ORBITALS = ("dz2", "dxz", "dyz", "dx2-y2", "dxy")


def _real_d_harmonics(x, y, z):
    """Normalized real d harmonics on unit vectors in Wannier90 order, shape (5, n)."""
    c = numpy.sqrt(15.0 / (4.0 * numpy.pi))
    return numpy.array(
        [
            numpy.sqrt(5.0 / (16.0 * numpy.pi)) * (3.0 * z * z - 1.0),
            c * x * z,
            c * y * z,
            0.5 * c * (x * x - y * y),
            c * x * y,
        ]
    )


def slater_d_interaction(U, J, f4_over_f2=0.625):
    r"""Direct and exchange interaction matrices of a d shell from Slater integrals.

    :math:`F^0 = U` and :math:`J = (F^2 + F^4)/14` with :math:`F^4/F^2` fixed, as in
    LDA+U (Liechtenstein, Anisimov and Zaanen). With the addition theorem

    .. math::

        U_{abcd} = \sum_k F^k \int\int R_a R_c P_k(\omega \cdot \omega') R_b' R_d'
                   \, d\omega \, d\omega',

    :math:`U_{mm'} = U_{m m' m m'}` and :math:`J_{mm'} = U_{m m' m' m}` in the real
    orbitals `D_ORBITALS`. The sphere quadrature is exact for these polynomials.

    Parameters
    ----------
    U : float
        Hubbard U, :math:`F^0`.
    J : float
        Hund's coupling.
    f4_over_f2 : float
        Ratio :math:`F^4/F^2`. Default 0.625 (3d).

    Returns
    -------
    direct, exchange : :class:`numpy.ndarray`
        :math:`U_{mm'}` and :math:`J_{mm'}`, shape (5, 5).
    """
    if not (numpy.isfinite(U) and numpy.isfinite(J) and numpy.isfinite(f4_over_f2)):
        raise ValueError("U, J and F4/F2 must be finite.")
    f2 = 14.0 * J / (1.0 + f4_over_f2)
    radial = {0: float(U), 2: f2, 4: f4_over_f2 * f2}
    cos_t, w_t = numpy.polynomial.legendre.leggauss(10)
    phi = 2.0 * numpy.pi * numpy.arange(20) / 20
    sin_t = numpy.sqrt(1.0 - cos_t**2)
    x = numpy.outer(sin_t, numpy.cos(phi)).ravel()
    y = numpy.outer(sin_t, numpy.sin(phi)).ravel()
    z = numpy.repeat(cos_t, len(phi))
    weights = numpy.repeat(w_t, len(phi)) * (2.0 * numpy.pi / len(phi))
    R = _real_d_harmonics(x, y, z)
    pair_density = (R[:, None, :] * R[None, :, :] * weights).reshape(25, -1)
    cosine = numpy.clip(numpy.outer(x, x) + numpy.outer(y, y) + numpy.outer(z, z), -1.0, 1.0)
    tensor = numpy.zeros((25, 25))
    for k, fk in radial.items():
        legendre = numpy.polynomial.legendre.legval(cosine, numpy.eye(k + 1)[k])
        tensor += fk * pair_density @ legendre @ pair_density.T
    # tensor[(a, c), (b, d)] = U_abcd.
    tensor = tensor.reshape(5, 5, 5, 5)
    m = numpy.arange(5)
    direct = tensor[m[:, None], m[:, None], m[None, :], m[None, :]]
    exchange = tensor[m[:, None], m[None, :], m[None, :], m[:, None]]
    return direct, exchange


def ising_pairs(direct, exchange, atoms):
    r"""Pair terms of the density-density interaction of d shells.

    Per atom this is Eq. 1 of Lichtenstein, Katsnelson and Kotliar (2001),

    .. math::

        \sum_{mm'} U_{mm'} n_{m\uparrow} n_{m'\downarrow}
        + \sum_\sigma \sum_{m < m'} (U - J)_{mm'} n_{m\sigma} n_{m'\sigma},

    that is (2 norb - 1) norb pairs.

    Parameters
    ----------
    direct, exchange : :class:`numpy.ndarray`
        :math:`U_{mm'}` and :math:`J_{mm'}`, shape (norb, norb).
    atoms : :class:`numpy.ndarray`
        Orbital index of every local orbital of every atom, shape (natoms, norb).

    Returns
    -------
    pairs : :class:`numpy.ndarray`
        Rows (s, i, s', j, V), shape (npairs, 5).
    """
    atoms = numpy.asarray(atoms)
    if atoms.ndim != 2:
        raise ValueError("atoms must have shape (natoms, norb).")
    norb = atoms.shape[1]
    if direct.shape != (norb, norb) or exchange.shape != (norb, norb):
        raise ValueError("atoms must be (natoms, norb) with (norb, norb) U and J matrices.")
    rows = []
    for orb in atoms:
        for m in range(norb):
            for n in range(norb):
                rows.append((0, orb[m], 1, orb[n], direct[m, n]))
        for s in range(2):
            for m in range(norb):
                for n in range(m + 1, norb):
                    rows.append((s, orb[m], s, orb[n], direct[m, n] - exchange[m, n]))
    return numpy.array(rows, dtype=float)


def pair_table(pairs, nbasis):
    """Validate pair terms and split them into spins, orbitals and couplings.

    Parameters
    ----------
    pairs :
        Rows (s, i, s', j, V): spins s, s' in {0, 1}, orbitals i, j and the
        coupling V of the term :math:`V n_{is} n_{js'}`.
    nbasis : int
        Number of orbitals.

    Returns
    -------
    spin, orb : :class:`numpy.ndarray`
        Spins and orbitals of the two legs, shape (npairs, 2).
    V : :class:`numpy.ndarray`
        Couplings, shape (npairs,).
    """
    table = numpy.asarray(pairs, dtype=float)
    if table.ndim != 2 or table.shape[1] != 5 or len(table) == 0:
        raise ValueError("pairs must be a nonempty table of rows (s, i, s', j, V).")
    index = table[:, :4]
    if not numpy.all(index == numpy.round(index)):
        raise ValueError("Pair spin and orbital entries must be integers.")
    spin = table[:, [0, 2]].astype(int)
    orb = table[:, [1, 3]].astype(int)
    V = table[:, 4]
    if (
        numpy.any((spin < 0) | (spin > 1))
        or numpy.any((orb < 0) | (orb >= nbasis))
        or not numpy.all(numpy.isfinite(V))
    ):
        raise ValueError("Pair spins must be 0/1, orbitals in range, and V finite.")
    if numpy.any((spin[:, 0] == spin[:, 1]) & (orb[:, 0] == orb[:, 1])):
        raise ValueError("A pair must join two distinct spin-orbitals.")
    return spin, orb, V
