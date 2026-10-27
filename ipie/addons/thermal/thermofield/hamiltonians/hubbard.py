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

from ipie.hamiltonians.generic import Generic


def hopping_matrix(lx, ly, pbc_x=True, pbc_y=True, t=1.0):
    """Nearest-neighbour hopping of an lx x ly square lattice.

    Site index i = ix + lx * iy. Boundary conditions are set per direction;
    the pinned supercells of He et al. (PRB 99) are open along x and periodic
    along y.

    Parameters
    ----------
    lx, ly : int
        Lattice dimensions.
    pbc_x, pbc_y : bool
        Periodic boundary conditions along x and y. Default True.
    t : float
        Hopping amplitude; the matrix elements are -t. Default 1.0.

    Returns
    -------
    h : :class:`numpy.ndarray`
        Hopping matrix, shape (lx * ly, lx * ly).
    """
    if (
        not isinstance(lx, (int, numpy.integer))
        or not isinstance(ly, (int, numpy.integer))
        or lx < 1
        or ly < 1
    ):
        raise ValueError("lx and ly must be positive integers.")
    if not numpy.isfinite(t):
        raise ValueError("t must be finite.")
    n = lx * ly
    h = numpy.zeros((n, n))
    for iy in range(ly):
        for ix in range(lx):
            i = ix + lx * iy
            if ix + 1 < lx:
                h[i, i + 1] = h[i + 1, i] = -t
            elif pbc_x and lx > 2:
                j = 0 + lx * iy
                h[i, j] = h[j, i] = -t
            if iy + 1 < ly:
                j = ix + lx * (iy + 1)
                h[i, j] = h[j, i] = -t
            elif pbc_y and ly > 2:
                j = ix
                h[i, j] = h[j, i] = -t
    return h


def pinning_field(lx, ly, h, edges=("left",)):
    r"""Antiferromagnetic pinning field of He et al. (PRB 99).

    :math:`v_{i\uparrow} = -v_{i\downarrow} = (-1)^{i_y} h` on the first column
    ("left") and, when requested, with the same formula on the last column
    ("right"), which gives the symmetric profiles of the 1/8-doped 16 x 4
    stripes. Here iy starts at 0, a global spin flip relative to the paper.

    Parameters
    ----------
    lx, ly : int
        Lattice dimensions.
    h : float
        Pinning strength.
    edges : tuple
        Pinned columns, "left" and/or "right". Default ("left",).

    Returns
    -------
    v : :class:`numpy.ndarray`
        Spin-dependent on-site potential, shape (2, lx * ly); add v[s] to the
        diagonal of H1[s].
    """
    n = lx * ly
    v = numpy.zeros((2, n))
    for iy in range(ly):
        s = h * (-1.0) ** iy
        if "left" in edges:
            i = 0 + lx * iy
            v[0, i] += s
            v[1, i] += -s
        if "right" in edges:
            i = (lx - 1) + lx * iy
            v[0, i] += s
            v[1, i] += -s
    return v


def build_hubbard_hamiltonian(
    lx, ly, U, mu, t=1.0, pbc_x=True, pbc_y=True, pin_h=0.0, pin_edges=("left",), verbose=False
):
    r"""Hubbard Hamiltonian of He et al. (PRB 99), Eq. 16, with pinning and chemical potential.

    The one-body part is :math:`h + (\mu - U/2) + v_\sigma` per spin and the
    interaction is :math:`U \sum_i n_{i\uparrow} n_{i\downarrow}` with one real
    Cholesky vector per site.

    Parameters
    ----------
    lx, ly : int
        Lattice dimensions.
    U : float
        Hubbard interaction.
    mu : float
        Chemical potential, entering as :math:`+\mu N`.
    t : float
        Hopping amplitude. Default 1.0.
    pbc_x, pbc_y : bool
        Periodic boundary conditions. Default True.
    pin_h : float
        Pinning strength. Default 0.0.
    pin_edges : tuple
        Pinned columns. Default ("left",).
    verbose : bool
        Print information. Default False.

    Returns
    -------
    hamiltonian : :class:`ipie.hamiltonians.generic.GenericRealChol`
        Hamiltonian.
    hop : :class:`numpy.ndarray`
        Hopping matrix.
    v : :class:`numpy.ndarray`
        Pinning potential, shape (2, nsites).
    """
    if not numpy.isfinite(U) or U < 0:
        raise ValueError("U must be finite and non-negative.")
    if not numpy.isfinite(mu) or not numpy.isfinite(pin_h):
        raise ValueError("mu and pin_h must be finite.")
    if not set(pin_edges) <= {"left", "right"}:
        raise ValueError("pin_edges may contain only left and right.")
    n = lx * ly
    hop = hopping_matrix(lx, ly, pbc_x=pbc_x, pbc_y=pbc_y, t=t)
    v = pinning_field(lx, ly, pin_h, pin_edges) if pin_h != 0.0 else numpy.zeros((2, n))
    h1 = numpy.array([hop + numpy.diag(mu - 0.5 * U + v[0]), hop + numpy.diag(mu - 0.5 * U + v[1])])
    chol = numpy.zeros((n * n, n))
    for i in range(n):
        chol[i * n + i, i] = numpy.sqrt(U)
    ham = Generic(h1e=h1, chol=chol, ecore=0.0, verbose=verbose)
    return ham, hop, v
