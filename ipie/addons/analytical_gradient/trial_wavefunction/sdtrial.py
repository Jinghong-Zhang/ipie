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
"""Single-determinant (RHF) trial wave function with forward-mode tangents.

Two equivalent parameterizations of trial relaxation are supported:

1. Rotated-basis (adafqmc convention): the Hamiltonian arrays are rotated to
   the relaxed-orbital basis and carry the lambda-dependence (dh1e, dchol);
   the trial is a fixed matrix (identity columns), dpsi = 0.
2. Fixed-basis: the Hamiltonian integrals are lambda-independent apart from
   h1e -> h1e + lambda*O (dchol = 0 -- for a one-body observable the two-body
   interaction never changes), and the trial carries the orbital response,
   psi(lambda) with tangent dpsi.

Both give the same converged gradient (they differ by a lambda-dependent
basis rotation plus an exponentially decaying initial-condition transient),
but the fixed-basis form keeps chol untouched, which also makes finite-delta
correlated sampling far better conditioned (the +/-delta runs share an
identical stochastic propagator).
"""

import numpy as np

from ipie.addons.analytical_gradient.utils.linalg import (
    flatten_ghalf,
    left_apply,
    rmatmul,
)


class SDTrial:
    def __init__(self, psi, nelec0, dpsi=None):
        """psi: (nao, >= nelec0) orbital coefficients; dpsi: its lambda-tangent."""
        psi = np.asarray(psi)
        self.psi = psi[:, :nelec0]
        self.dpsi = (
            np.zeros_like(self.psi) if dpsi is None else np.asarray(dpsi)[:, :nelec0]
        )
        self.nelec0 = nelec0
        self.nao = psi.shape[0]
        # Trial one-body Green's function in the adafqmc (conjugate) convention,
        # G = psi* (psi^T psi*)^{-1} psi^T, with its trial tangent.
        ovlp = self.psi.T @ self.psi.conj()
        dovlp = self.dpsi.T @ self.psi.conj() + self.psi.T @ self.dpsi.conj()
        oinv = np.linalg.inv(ovlp)
        self.G = self.psi.conj() @ oinv @ self.psi.T
        self.dG = (
            self.dpsi.conj() @ oinv @ self.psi.T
            + self.psi.conj() @ oinv @ self.dpsi.T
            - self.psi.conj() @ oinv @ dovlp @ oinv @ self.psi.T
        )
        self._has_dpsi = bool(self.dpsi.any())
        self.rh1 = None
        self.drh1 = None
        self.rchol = None
        self.drchol = None

    def half_rot(self, ham):
        """Half-rotated one-body Hamiltonian and Cholesky vectors, with tangents
        from both the Hamiltonian arrays and the trial."""
        psic = self.psi.conj()
        dpsic = self.dpsi.conj()
        self.rh1 = psic.T @ ham.h1e
        self.drh1 = dpsic.T @ ham.h1e + psic.T @ ham.dh1e
        self.rchol = np.einsum("ij,aik->ajk", psic, ham.chol)
        self.drchol = np.einsum("ij,aik->ajk", dpsic, ham.chol) + np.einsum(
            "ij,aik->ajk", psic, ham.dchol
        )
        # Flattened views/flags for the BLAS force-bias and local-energy kernels.
        self.rchol_flat = self.rchol.reshape(self.rchol.shape[0], -1)
        self.drchol_flat = self.drchol.reshape(self.drchol.shape[0], -1)
        self._has_drchol = bool(self.drchol.any())

    def calc_overlap(self, states):
        """S_w = psi^dag phi_w for a batch of (nwalkers, nao, nocc) states."""
        return left_apply(self.psi.conj().T, states)

    def calc_overlap_with_tangent(self, phi, dphi):
        """S = psi^dag phi and dS = dpsi^dag phi + psi^dag dphi."""
        S = left_apply(self.psi.conj().T, phi)
        dS = left_apply(self.psi.conj().T, dphi)
        if self._has_dpsi:
            dS = dS + left_apply(self.dpsi.conj().T, phi)
        return S, dS

    def get_ghalf_with_tangent(self, phi, dphi):
        """Half-rotated Green's function Theta = phi S^{-1} and its tangent.

        dTheta = dphi S^{-1} - phi S^{-1} dS S^{-1}, with dS carrying the
        trial tangent as well.  Returns (Ghalf, dGhalf, S, dS, Sinv).
        """
        S, dS = self.calc_overlap_with_tangent(phi, dphi)
        Sinv = np.linalg.inv(S)
        Ghalf = phi @ Sinv
        dGhalf = dphi @ Sinv - Ghalf @ (dS @ Sinv)
        return Ghalf, dGhalf, S, dS, Sinv

    def calc_force_bias_with_tangent(self, Ghalf, dGhalf):
        """vbias_g = 2 Tr(rchol_g Theta), with product-rule tangent.

        Flattened (nw, nocc*nao) @ (nocc*nao, nchol) gemms with real/imag
        splitting (core-ipie force-bias convention)."""
        Gt = flatten_ghalf(Ghalf)
        dGt = flatten_ghalf(dGhalf)
        vbias = 2.0 * rmatmul(Gt, self.rchol_flat.T)
        dvbias = 2.0 * rmatmul(dGt, self.rchol_flat.T)
        if self._has_drchol:
            dvbias = dvbias + 2.0 * rmatmul(Gt, self.drchol_flat.T)
        return vbias, dvbias

    def get_trial_ghalf_with_tangent(self):
        """Gtilde = psi (psi^dag psi)^{-1}, (nao, nocc), with trial tangent."""
        ovlp = np.einsum("ij,ik->jk", self.psi.conj(), self.psi)
        dovlp = np.einsum("ij,ik->jk", self.dpsi.conj(), self.psi) + np.einsum(
            "ij,ik->jk", self.psi.conj(), self.dpsi
        )
        oinv = np.linalg.inv(ovlp)
        G = self.psi @ oinv
        dG = self.dpsi @ oinv - G @ (dovlp @ oinv)
        return G, dG

    def eval_energy_with_tangent(self, ham):
        """Trial energy and its tangent (Hamiltonian and trial contributions).

        Matches adafqmc get_trial_energy; requires half_rot(ham) to have been
        called with the same Hamiltonian.
        """
        G, dG = self.get_trial_ghalf_with_tangent()
        e1 = 2.0 * np.einsum("ij,ji->", self.rh1, G)
        de1 = 2.0 * (
            np.einsum("ij,ji->", self.drh1, G) + np.einsum("ij,ji->", self.rh1, dG)
        )
        X = 2.0 * np.einsum("pij,ji->p", self.rchol, G)
        dX = 2.0 * (
            np.einsum("pij,ji->p", self.drchol, G) + np.einsum("pij,ji->p", self.rchol, dG)
        )
        ej = np.dot(X, X)
        dej = 2.0 * np.dot(X, dX)
        T = np.einsum("gip,pj->gij", self.rchol, G)
        dT = np.einsum("gip,pj->gij", self.drchol, G) + np.einsum(
            "gip,pj->gij", self.rchol, dG
        )
        ex = 2.0 * np.einsum("gij,gji->", T, T)
        dex = 2.0 * (np.einsum("gij,gji->", dT, T) + np.einsum("gij,gji->", T, dT))
        e = ham.enuc + e1 + 0.5 * (ej - ex)
        de = de1 + 0.5 * (dej - dex)
        return e, de
