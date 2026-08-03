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
"""Hamiltonian arrays together with their lambda-tangents at lambda = 0."""

import numpy as np


class HamTangent:
    """Cholesky-factorized Hamiltonian plus tangents d/dlambda of its arrays.

    All lambda dependence of the phaseless AFQMC estimator enters through the
    arrays (h1e, chol); carrying their tangents (dh1e, dchol) covers both the
    fixed-trial workflow (dh1e = O, dchol = 0) and the relaxed-trial workflow
    (rotated arrays, trial frozen to identity columns) with one code path.

    Parameters
    ----------
    nelec0 : int
        Number of spin-up electrons (spin-restricted: ndown = nup).
    nao : int
        Number of orbitals.
    h1e : np.ndarray
        One-body Hamiltonian, (nao, nao).
    chol : np.ndarray
        Cholesky vectors of the two-body Hamiltonian, (nchol, nao, nao),
        each symmetric.
    enuc : float
        Nuclear repulsion (lambda-independent; the observable's constant part
        is added to the final gradient, never here).
    dh1e, dchol : np.ndarray, optional
        Tangents d(h1e)/dlambda and d(chol)/dlambda at lambda = 0 (default 0).
    """

    def __init__(self, nelec0, nao, h1e, chol, enuc, dh1e=None, dchol=None):
        self.nelec0 = nelec0
        self.nao = nao
        self.h1e = np.asarray(h1e)
        self.chol = np.asarray(chol)
        self.enuc = float(enuc)
        self.nchol = self.chol.shape[0]
        self.dh1e = np.zeros_like(self.h1e) if dh1e is None else np.asarray(dh1e)
        self.dchol = np.zeros_like(self.chol) if dchol is None else np.asarray(dchol)
        assert self.h1e.shape == (nao, nao)
        assert self.chol.shape == (self.nchol, nao, nao)
        assert self.dh1e.shape == self.h1e.shape
        assert self.dchol.shape == self.chol.shape

        # Subtract the one-body term from normal-reordering of the two-body
        # operators, Eq. (17) of Motta-Zhang (arXiv:1711.02242), and its tangent.
        v0 = 0.5 * np.einsum("apr,arq->pq", self.chol, self.chol)
        dv0 = 0.5 * (
            np.einsum("apr,arq->pq", self.dchol, self.chol)
            + np.einsum("apr,arq->pq", self.chol, self.dchol)
        )
        self.h1e_mod = self.h1e - v0
        self.dh1e_mod = self.dh1e - dv0

        # Packed upper-triangle Cholesky for the half-flops VHS gemms
        # (core-ipie / adafqmc convention); valid only for symmetric vectors,
        # otherwise the propagator falls back to the unpacked contraction.
        self.has_dchol = bool(self.dchol.any())
        self.sym_idx_i, self.sym_idx_j = np.triu_indices(nao)
        cholT = self.chol.transpose(0, 2, 1)
        dcholT = self.dchol.transpose(0, 2, 1)
        if np.array_equal(self.chol, cholT) or np.allclose(self.chol, cholT, atol=1e-13):
            if not self.has_dchol or np.allclose(self.dchol, dcholT, atol=1e-13):
                self.chol_packed = np.ascontiguousarray(
                    self.chol[:, self.sym_idx_i, self.sym_idx_j]
                )
                self.dchol_packed = (
                    np.ascontiguousarray(self.dchol[:, self.sym_idx_i, self.sym_idx_j])
                    if self.has_dchol
                    else None
                )
            else:
                self.chol_packed = None
                self.dchol_packed = None
        else:
            self.chol_packed = None
            self.dchol_packed = None


def build_fixed_trial_tangent(nelec0, nao, h1e, chol, enuc, obs_mat):
    """Tangent inputs for a lambda-independent trial: H(lambda) = H + lambda*O."""
    return HamTangent(
        nelec0, nao, h1e, chol, enuc, dh1e=np.asarray(obs_mat), dchol=np.zeros_like(chol)
    )


def build_relaxed_trial_tangent(nelec0, nao, h1e, chol, enuc, obs_mat, rot_mat, drot_mat):
    """Tangent inputs for the relaxed-trial workflow.

    Mirrors adafqmc: the Hamiltonian is rotated into the basis of the relaxed
    orbitals U(lambda) (value rot_mat, tangent drot_mat at lambda = 0), where
    the trial is the fixed matrix eye(nao)[:, :nelec0].  This is the exact
    lambda-derivative at lambda = 0 of rot_ham_with_orbs composed with
    ham_with_obs:

        h'(l)  = U(l)^dag (h + l*O) U(l)   -> dh'  = dU^dag h U + U^dag h dU + U^dag O U
        L'^g(l) = U(l)^dag L^g U(l)        -> dL'^g = dU^dag L^g U + U^dag L^g dU
    """
    U = np.asarray(rot_mat)
    dU = np.asarray(drot_mat)
    obs_mat = np.asarray(obs_mat)
    Uh = U.conj().T
    dUh = dU.conj().T
    h1e_rot = Uh @ h1e @ U
    dh1e_rot = dUh @ h1e @ U + Uh @ h1e @ dU + Uh @ obs_mat @ U
    chol_rot = np.einsum("qi,aij,jp->aqp", Uh, chol, U)
    dchol_rot = np.einsum("qi,aij,jp->aqp", dUh, chol, U) + np.einsum(
        "qi,aij,jp->aqp", Uh, chol, dU
    )
    return HamTangent(nelec0, nao, h1e_rot, chol_rot, enuc, dh1e=dh1e_rot, dchol=dchol_rot)


def shifted_copy(ham, eps):
    """Value-only Hamiltonian at lambda = eps along the stored tangent direction.

    Used for common-random-number finite differences: the returned object has
    zero tangents, so running the same driver on it produces the value branch
    of the estimator at lambda = eps.
    """
    return HamTangent(
        ham.nelec0,
        ham.nao,
        ham.h1e + eps * ham.dh1e,
        ham.chol + eps * ham.dchol,
        ham.enuc,
    )
