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
"""Single-determinant (RHF) trial wave function with half-rotated tangents.

The trial itself is lambda-independent in both supported workflows (fixed
trial, or relaxed trial with the Hamiltonian rotated so the trial is identity
columns); lambda enters the half-rotated quantities only through the
Hamiltonian array tangents.
"""

import numpy as np


class SDTrial:
    def __init__(self, psi, nelec0):
        """psi: (nao, >= nelec0) orbital coefficients; must be lambda-independent."""
        psi = np.asarray(psi)
        self.psi = psi[:, :nelec0]
        self.nelec0 = nelec0
        self.nao = psi.shape[0]
        # Trial one-body Green's function in the adafqmc (conjugate) convention:
        # G = psi* (psi^T psi*)^{-1} psi^T, the transpose of psi (psi^dag psi)^{-1} psi^dag.
        ovlp = self.psi.T @ self.psi.conj()
        self.G = self.psi.conj() @ np.linalg.inv(ovlp) @ self.psi.T
        self.rh1 = None
        self.drh1 = None
        self.rchol = None
        self.drchol = None

    def half_rot(self, ham):
        """Half-rotated one-body Hamiltonian and Cholesky vectors, with tangents."""
        psic = self.psi.conj()
        self.rh1 = psic.T @ ham.h1e
        self.drh1 = psic.T @ ham.dh1e
        self.rchol = np.einsum("ij,aik->ajk", psic, ham.chol)
        self.drchol = np.einsum("ij,aik->ajk", psic, ham.dchol)

    def calc_overlap(self, states):
        """S_w = psi^dag phi_w for a batch of (nwalkers, nao, nocc) states."""
        return np.einsum("ij,wik->wjk", self.psi.conj(), states)

    def get_ghalf_with_tangent(self, phi, dphi):
        """Half-rotated Green's function Theta = phi S^{-1} and its tangent.

        dTheta = dphi S^{-1} - phi S^{-1} (psi^dag dphi) S^{-1}.
        Returns (Ghalf, dGhalf, S, dS, Sinv) so callers can reuse the overlap.
        """
        S = self.calc_overlap(phi)
        dS = self.calc_overlap(dphi)
        Sinv = np.linalg.inv(S)
        Ghalf = phi @ Sinv
        dGhalf = dphi @ Sinv - Ghalf @ (dS @ Sinv)
        return Ghalf, dGhalf, S, dS, Sinv

    def calc_force_bias_with_tangent(self, Ghalf, dGhalf):
        """vbias_g = 2 Tr(rchol_g Theta), with product-rule tangent."""
        vbias = 2.0 * np.einsum("pij,wji->wp", self.rchol, Ghalf)
        dvbias = 2.0 * (
            np.einsum("pij,wji->wp", self.drchol, Ghalf)
            + np.einsum("pij,wji->wp", self.rchol, dGhalf)
        )
        return vbias, dvbias

    def get_trial_ghalf(self):
        """Gtilde = psi (psi^dag psi)^{-1}, (nao, nocc); lambda-independent."""
        ovlp = np.einsum("ij,ik->jk", self.psi.conj(), self.psi)
        return self.psi @ np.linalg.inv(ovlp)

    def eval_energy_with_tangent(self, ham):
        """Trial energy and its tangent (through rh1/rchol only; dGtilde = 0).

        Matches adafqmc get_trial_energy; requires half_rot(ham) to have been
        called with the same Hamiltonian.
        """
        G = self.get_trial_ghalf()
        e1 = 2.0 * np.einsum("ij,ji->", self.rh1, G)
        de1 = 2.0 * np.einsum("ij,ji->", self.drh1, G)
        X = 2.0 * np.einsum("pij,ji->p", self.rchol, G)
        dX = 2.0 * np.einsum("pij,ji->p", self.drchol, G)
        ej = np.dot(X, X)
        dej = 2.0 * np.dot(X, dX)
        T = np.einsum("gip,pj->gij", self.rchol, G)
        dT = np.einsum("gip,pj->gij", self.drchol, G)
        ex = 2.0 * np.einsum("gij,gji->", T, T)
        dex = 2.0 * (np.einsum("gij,gji->", dT, T) + np.einsum("gij,gji->", T, dT))
        e = ham.enuc + e1 + 0.5 * (ej - ex)
        de = de1 + 0.5 * (dej - dex)
        return e, de
