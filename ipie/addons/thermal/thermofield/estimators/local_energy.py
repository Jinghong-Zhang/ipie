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

from typing import Union

import numpy
import plum

from ipie.addons.thermal.estimators.generic import local_energy_generic_cholesky
from ipie.addons.thermal.thermofield.hamiltonians.ueg import UEGSparseChol
from ipie.hamiltonians.generic import GenericComplexChol, GenericRealChol
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import to_host


@local_energy_generic_cholesky.dispatch
def local_energy_generic_cholesky(hamiltonian: UEGSparseChol, P):
    r"""Calculate local energy of a 1RDM with the sparse plane-wave factors.

    .. math::

        E_2 = \frac{1}{2} \sum_n \left[(\mathrm{Tr} A_n P)^2 + (\mathrm{Tr} B_n P)^2\right]
              - \frac{1}{2} E_x,

    with :math:`P = P_0 + P_1` in the Coulomb term and the exchange sum of
    :meth:`UEGSparseChol.exchange_energy`.

    Parameters
    ----------
    hamiltonian : :class:`UEGSparseChol`
        Plane-wave Hamiltonian.
    P : :class:`numpy.ndarray`
        One-body density matrix, shape (2, nbasis, nbasis).

    Returns
    -------
    (E, T, V): tuple
        Total, one-body (with `ecore`) and two-body energies.
    """
    e1b = numpy.sum(hamiltonian.H1[0] * P[0]) + numpy.sum(hamiltonian.H1[1] * P[1])
    Pt = (P[0] + P[1]).ravel()
    XA = hamiltonian.AT.dot(Pt)
    XB = hamiltonian.BT.dot(Pt)
    ecoul = 0.5 * (numpy.dot(XA, XA) + numpy.dot(XB, XB))
    exx = 0.5 * hamiltonian.exchange_energy(numpy.asarray(P)).sum()
    e2b = ecoul - exx
    return (e1b + e2b + hamiltonian.ecore, e1b + hamiltonian.ecore, e2b)


@plum.dispatch
def local_energy_batched(hamiltonian: Union[GenericRealChol, GenericComplexChol], P):
    """Calculate local energies of a stack of 1RDMs.

    Parameters
    ----------
    hamiltonian : :class:`GenericRealChol` or :class:`GenericComplexChol`
        Hamiltonian.
    P : :class:`numpy.ndarray`
        One-body density matrices, shape (nwalkers, 2, nbasis, nbasis).

    Returns
    -------
    energies : :class:`numpy.ndarray`
        Total, one-body (with `ecore`) and two-body energies, shape (nwalkers, 3).
    """
    P = to_host(P)
    return numpy.array(
        [local_energy_generic_cholesky(hamiltonian, P[iw]) for iw in range(P.shape[0])]
    )


@plum.dispatch
def local_energy_batched(hamiltonian: UEGSparseChol, P):
    """Calculate local energies of a stack of 1RDMs with the sparse plane-wave factors.

    Parameters
    ----------
    hamiltonian : :class:`UEGSparseChol`
        Plane-wave Hamiltonian.
    P : :class:`numpy.ndarray`
        One-body density matrices, shape (nwalkers, 2, nbasis, nbasis).

    Returns
    -------
    energies : :class:`numpy.ndarray`
        Total, one-body (with `ecore`) and two-body energies, shape (nwalkers, 3).
    """
    P = xp.asarray(P)
    nwalkers, nbasis = P.shape[0], P.shape[-1]
    e1b = xp.einsum("ij,wij->w", hamiltonian.H1[0], P[:, 0]) + xp.einsum(
        "ij,wij->w", hamiltonian.H1[1], P[:, 1]
    )
    Pt = (P[:, 0] + P[:, 1]).reshape(nwalkers, nbasis * nbasis)
    XA = hamiltonian.AT.dot(Pt.T).T  # Shape (nwalkers, nchol).
    XB = hamiltonian.BT.dot(Pt.T).T
    ecoul = 0.5 * (xp.einsum("wx,wx->w", XA, XA) + xp.einsum("wx,wx->w", XB, XB))
    exx = 0.5 * hamiltonian.exchange_energy(P).sum(axis=1)
    e2b = ecoul - exx
    energies = xp.empty((nwalkers, 3), dtype=xp.complex128)
    energies[:, 0] = e1b + e2b + hamiltonian.ecore
    energies[:, 1] = e1b + hamiltonian.ecore
    energies[:, 2] = e2b
    return energies
