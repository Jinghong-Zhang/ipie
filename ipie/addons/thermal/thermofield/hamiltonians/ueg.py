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
import scipy.sparse

from ipie.config import config
from ipie.hamiltonians.generic_base import GenericBase
from ipie.utils.backend import arraylib as xp


class UEGSparseChol(GenericBase):
    r"""Plane-wave Hamiltonian with sparse complex Cholesky factors.

    Same interface as :class:`ipie.hamiltonians.generic.GenericComplexChol`, with
    the factors stored as scipy sparse matrices. For the uniform electron gas
    every Cholesky vector is a density fluctuation
    :math:`\rho_q = \sum_k c^\dagger_{k+q} c_k` with a single amplitude, so the
    exchange energy is a sum over the (k + q, k) pairs of each q. Build with
    :meth:`from_ueg`.

    Parameters
    ----------
    h1e : :class:`numpy.ndarray`
        One-body Hamiltonian, shape (2, nbasis, nbasis).
    chol : :class:`scipy.sparse.spmatrix`
        Cholesky vectors, shape (nbasis^2, nchol).
    A, B : :class:`scipy.sparse.spmatrix`
        Hermitian and anti-Hermitian parts of the Cholesky vectors, shape
        (nbasis^2, nchol), as in `GenericComplexChol`.
    ecore : float
        Constant energy. Default 0.0.
    verbose : bool
        Print information. Default False.

    Attributes
    ----------
    L, LT : :class:`scipy.sparse.csr_matrix`
        Field-major operator matrix :math:`[A | B]`, shape (nbasis^2, nfields),
        and its transpose.
    h1e_mod : :class:`numpy.ndarray`
        One-body Hamiltonian with the correction from reordering the two-body
        operators.
    """

    def __init__(self, h1e, chol, A, B, ecore=0.0, verbose=False):
        h1e = numpy.asarray(h1e)
        assert h1e.shape[0] == 2
        super().__init__(numpy.array(h1e, dtype=numpy.complex128), ecore=ecore, verbose=verbose)
        nbasis = self.nbasis
        self.chol = scipy.sparse.csc_matrix(chol, dtype=numpy.complex128)
        self.A = scipy.sparse.csr_matrix(A, dtype=numpy.complex128)
        self.B = scipy.sparse.csr_matrix(B, dtype=numpy.complex128)
        self.nchol = self.chol.shape[1]
        self.nfields = 2 * self.nchol
        assert self.chol.shape[0] == nbasis * nbasis
        self.L = scipy.sparse.hstack([self.A, self.B], format="csr", dtype=numpy.complex128)
        self.LT = scipy.sparse.csr_matrix(self.L.T)
        self.AT = scipy.sparse.csr_matrix(self.A.T)
        self.BT = scipy.sparse.csr_matrix(self.B.T)
        self.chunked = False
        self.sparse = True

        # Exchange bookkeeping: for every Cholesky vector q the (k + q, k) pairs
        # and c_q^2, grouped by the number of pairs n as
        # (KPQ (nq, n), K (nq, n), c^2 (nq,)).
        coo = self.chol.tocoo()
        kpq_all, k_all = coo.row // nbasis, coo.row % nbasis
        groups = {}
        for q in range(self.nchol):
            sel = numpy.flatnonzero(coo.col == q)
            if sel.size == 0:
                continue
            c = coo.data[sel[0]]
            assert numpy.allclose(coo.data[sel], c)
            g = groups.setdefault(sel.size, [[], [], []])
            g[0].append(kpq_all[sel])
            g[1].append(k_all[sel])
            g[2].append(c * c)
        self.exchange_groups = [
            (numpy.array(g[0]), numpy.array(g[1]), numpy.array(g[2]))
            for _, g in sorted(groups.items())
        ]

        # One-body part from reordering the two-body operators (Motta et al.
        # (2017), Eq. 17): v0 = L L^dagger / 2, diagonal for density fluctuations.
        v0 = numpy.zeros(nbasis)
        numpy.add.at(v0, kpq_all, 0.5 * numpy.abs(coo.data) ** 2)
        self.h1e_mod = numpy.array([self.H1[0] - numpy.diag(v0), self.H1[1] - numpy.diag(v0)])

    @classmethod
    def from_ueg(cls, ueg, ecore=0.0):
        """Build from a :class:`ipie.addons.thermal.utils.ueg.UEG` after `ueg.build()`.

        Parameters
        ----------
        ueg : :class:`ipie.addons.thermal.utils.ueg.UEG`
            Built UEG system.
        ecore : float
            Constant energy. Default 0.0.

        Returns
        -------
        hamiltonian : :class:`UEGSparseChol`
            Hamiltonian with chol = 2 rho_q as in the thermal addon tests.
        """
        chol = 2.0 * ueg.chol_vecs
        A = -1j * ueg.iA  # iA = i (rho + rho^T), so A = rho + rho^T.
        B = -1j * ueg.iB  # iB = -(rho - rho^T), so B = i (rho - rho^T).
        h1 = numpy.asarray(ueg.H1[0])
        return cls(numpy.array([h1, h1]), chol, A, B, ecore=ecore)

    def cast_to_cupy(self, verbose=False):
        """Copy the integrals to the GPU, the sparse factors as cupyx sparse matrices."""
        super().cast_to_cupy(verbose=verbose)
        if not config.get_option("use_gpu"):
            return
        # pylint: disable=import-error
        import cupyx.scipy.sparse

        for name in ("chol", "A", "B", "L", "LT", "AT", "BT"):
            mat = getattr(self, name)
            if scipy.sparse.isspmatrix_csc(mat):
                setattr(self, name, cupyx.scipy.sparse.csc_matrix(mat))
            else:
                setattr(self, name, cupyx.scipy.sparse.csr_matrix(mat))
        self.exchange_groups = [
            (xp.asarray(KPQ), xp.asarray(K), xp.asarray(c2)) for KPQ, K, c2 in self.exchange_groups
        ]

    # Compatibility with earlier versions.
    def to_device(self):
        """Copy the integrals to the GPU; needs IPIE_USE_GPU=1. Use `cast_to_cupy`."""
        if not config.get_option("use_gpu"):
            raise RuntimeError("to_device needs the GPU backend: set IPIE_USE_GPU=1.")
        self.cast_to_cupy()
        return self

    @property
    def _exchange_groups(self):
        return self.exchange_groups

    def vhs_flat(self, xshifted):
        return self.construct_VHS_flat(xshifted)

    def force_bias_rows(self, P):
        return self.construct_force_bias_rows(P)

    def construct_VHS_flat(self, xshifted):
        r"""Two-body operator :math:`\sum_n x_n L_n` for every walker.

        Parameters
        ----------
        xshifted : :class:`numpy.ndarray`
            Complex fields, shape (nfields, nwalkers).

        Returns
        -------
        VHS : :class:`numpy.ndarray`
            Flattened operators, shape (nbasis^2, nwalkers).
        """
        return self.L.dot(xshifted)

    def construct_force_bias_rows(self, P):
        r""":math:`\mathrm{Tr}(L_n P)` for a stack of flattened density matrices.

        Parameters
        ----------
        P : :class:`numpy.ndarray`
            Flattened density matrices, shape (nrows, nbasis^2).

        Returns
        -------
        vbias : :class:`numpy.ndarray`
            Shape (nrows, nfields).
        """
        return self.LT.dot(P.T).T

    def exchange_energy(self, P):
        r"""Exchange sum of every density matrix in a stack.

        .. math::

            E_x = \sum_q c_q^2 \sum_{k, k' \in S_q} P_{k+q, k'+q} P_{k', k},

        equal to the complex-Cholesky exchange
        :math:`\sum_n \mathrm{Tr}(P^T A_n P^T A_n) + \mathrm{Tr}(P^T B_n P^T B_n)`.

        Parameters
        ----------
        P : :class:`numpy.ndarray`
            Density matrices, shape (..., nbasis, nbasis).

        Returns
        -------
        exx : :class:`numpy.ndarray`
            One value per matrix, shape P.shape[:-2].
        """
        P = xp.asarray(P)
        lead = P.shape[:-2]
        exx = xp.zeros(lead, dtype=xp.complex128)
        for KPQ, K, c2 in self.exchange_groups:
            nq, n = KPQ.shape
            # Chunk q so that the gathered (..., chunk, n, n) stays near 1e7 elements.
            lead_size = int(numpy.prod(lead)) if lead else 1
            chunk = max(1, int(1e7 // max(1, lead_size * n * n)))
            for start in range(0, nq, chunk):
                kpq = KPQ[start : start + chunk]
                k = K[start : start + chunk]
                X = P[..., kpq[:, :, None], kpq[:, None, :]]  # Shape (..., chunk, n, n).
                Y = P[..., k[:, :, None], k[:, None, :]]
                exx += xp.einsum("...qij,...qji->...q", X, Y).dot(c2[start : start + chunk])
        return exx
