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
"""Number-conserving thermal trial for thermofield-guided FT-AFQMC.

The guide is *fixed at the target inverse temperature beta* (it is not a
remaining-time trace object):

    D_T = exp(-beta k_T / 2),        k_T Hermitian (chemical potential included),

with guide overlap and mixed Green's function for a right walker Delta

    S_T(Delta) = det(I + D_T^dagger Delta),
    G_T(Delta) = Delta (I + D_T^dagger Delta)^{-1} D_T^dagger.

Both are evaluated in an eigenbasis gauge that never forms exp(+-beta*eps/2)
factors larger than O(1), so large beta (projector limit) is numerically
stable.
"""

import numpy

from ipie.addons.thermal.estimators.particle_number import particle_number
from ipie.addons.thermal.thermofield.gaussian import (
    principal_log_phase,
    stabilized_inverse_one_plus,
)


class ThermofieldThermalTrial:
    """Fixed target-beta thermal one-body trial for thermofield walkers.

    Parameters
    ----------
    hamiltonian : :class:`GenericRealChol` or :class:`GenericComplexChol`
        ipie Hamiltonian; used for the default k_T = H1 - mu * I.
    beta : float
        Target inverse temperature of the simulation (the guide is fixed
        here, walkers are propagated from Delta = I to theta = beta / 2).
    mu : float
        Chemical potential.  Ignored for k_trial input (assumed included).
    k_trial : :class:`numpy.ndarray`, optional
        Hermitian one-body guide matrix, shape (2, nbasis, nbasis), with the
        chemical potential already included (e.g. a thermal HF Fock matrix
        minus mu * I).  Defaults to hamiltonian.H1 - mu * I.
    verbose : bool
        Print info.
    """

    def __init__(self, hamiltonian, beta, mu=0.0, k_trial=None, verbose=False):
        self.name = "thermofield"
        self.compute_trial_energy = False
        self.verbose = verbose
        self.beta = beta
        self.mu = mu

        nbasis = hamiltonian.nbasis
        self.nbasis = nbasis

        if k_trial is None:
            muN = mu * numpy.eye(nbasis, dtype=hamiltonian.H1.dtype)
            k_trial = numpy.array([hamiltonian.H1[0] - muN, hamiltonian.H1[1] - muN])
        k_trial = numpy.asarray(k_trial)
        assert k_trial.shape == (2, nbasis, nbasis)
        for s in range(2):
            if not numpy.allclose(k_trial[s], k_trial[s].conj().T, atol=1e-10):
                raise ValueError("k_trial must be Hermitian for the thermofield trial.")
        self.k_trial = k_trial

        # Eigenbasis of k_T: k_T = C diag(eps) C^dagger with real eps.
        eps = numpy.zeros((2, nbasis))
        C = numpy.zeros((2, nbasis, nbasis), dtype=numpy.complex128)
        for s in range(2):
            eps[s], C[s] = numpy.linalg.eigh(k_trial[s])
        self.eps_T = eps
        self.C_T = C

        # Direct D_T = exp(-beta k_T / 2) for moderate beta*|eps| (may overflow
        # for extreme arguments; the stabilized methods below never use it).
        self.dmat = numpy.array(
            [C[s] @ numpy.diag(numpy.exp(-0.5 * beta * eps[s])) @ C[s].conj().T for s in range(2)]
        )

        # Thermal occupations of the guide itself: f = (I + exp(beta k_T))^{-1},
        # computed stably as 0.5 * (1 - tanh(beta eps / 2)).
        occ = 0.5 * (1.0 - numpy.tanh(0.5 * beta * eps))
        f = numpy.array([C[s] @ numpy.diag(occ[s]) @ C[s].conj().T for s in range(2)])
        # 1-RDM convention P[i, j] = <c_i^dagger c_j> = [f]_{ji}; the transpose
        # only matters for complex Hermitian k_trial.
        self.P = numpy.array([f[0].T, f[1].T])
        # Hole convention G_ij = <c_i c_j^dagger> = I - P.T used by the thermal
        # propagator machinery (construct_mean_field_shift consumes trial.G via
        # one_rdm_from_G, which inverts this transformation).
        I = numpy.eye(nbasis)
        self.G = numpy.array([I - f[0], I - f[1]])
        self.nav = particle_number(self.P).real

        if verbose:
            print("# Building ThermofieldThermalTrial.")
            print(f"# beta (fixed target) = {beta}")
            print(f"# mu = {mu}")
            print(f"# Trial average particle number: {self.nav}")

    def _stabilized_spin_block(self, Qmat, log_d, Tmat, s):
        """(log det(I + Delta Lambda), (I + Delta Lambda)^{-1}) for spin s.

        With the QDT-factored walker Delta = Q e^{D} T and the guide
        Lambda = D_T^dagger = C e^{-A} C^dagger (A = diag(beta eps / 2)),

            Delta Lambda = Q e^{D} (T C) e^{-A} C^dagger,

        evaluated with the two-sided stabilized splitting.  Note
        det(I + Lambda Delta) = det(I + Delta Lambda) (Sylvester) and
        G = Delta (I + Lambda Delta)^{-1} Lambda = I - (I + Delta Lambda)^{-1}.
        """
        a = 0.5 * self.beta * self.eps_T[s]
        C = self.C_T[s]
        d = numpy.zeros(self.nbasis) if log_d is None else log_d[s]
        X = C if Tmat is None else Tmat[s] @ C
        return stabilized_inverse_one_plus(Qmat[s], d, X, -a, C.conj().T, Vinv=C)

    def calc_log_overlap(self, Qmat, log_d=None, Tmat=None):
        r"""Complex log of the guide overlap S_T(Delta).

        Walkers are QDT-factored, Delta[s] = Qmat[s] diag(e^{log_d[s]})
        Tmat[s] (see :class:`ThermofieldWalkers`); a plain matrix walker
        corresponds to log_d = 0, Tmat = I (the defaults).

        Parameters
        ----------
        Qmat : :class:`numpy.ndarray`
            Walker Q factors, shape (2, nbasis, nbasis).
        log_d : :class:`numpy.ndarray`, optional
            Log scales, shape (2, nbasis).  Defaults to zero.
        Tmat : :class:`numpy.ndarray`, optional
            Walker T factors, shape (2, nbasis, nbasis).  Defaults to I.

        Returns
        -------
        log_ovlp : complex
            sum_s log det(I + D_T[s]^dagger Delta[s]).
        """
        log_ovlp = 0.0 + 0.0j
        for s in range(2):
            log_det, _ = self._stabilized_spin_block(Qmat, log_d, Tmat, s)
            log_ovlp += log_det
        return principal_log_phase(log_ovlp)

    def calc_greens_function(self, Qmat, log_d=None, Tmat=None):
        r"""Mixed transition Green's function G_T(Delta) per spin.

        Returns G with the convention <c_i^dagger c_j> / <1> = G[s][j, i]
        (see :mod:`ipie.addons.thermal.thermofield.gaussian`), evaluated as
        G = I - (I + Delta Lambda)^{-1} with the stabilized inverse.

        Returns
        -------
        G : :class:`numpy.ndarray`
            Shape (2, nbasis, nbasis).
        """
        nbasis = self.nbasis
        G = numpy.zeros((2, nbasis, nbasis), dtype=numpy.complex128)
        I = numpy.eye(nbasis)
        for s in range(2):
            _, inv = self._stabilized_spin_block(Qmat, log_d, Tmat, s)
            G[s] = I - inv
        return G
