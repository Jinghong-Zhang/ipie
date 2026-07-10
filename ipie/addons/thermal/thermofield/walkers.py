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
"""Walker container for open thermofield Gaussians."""

import numpy
import scipy.linalg

from ipie.addons.thermal.thermofield.trial import ThermofieldThermalTrial
from ipie.walkers.base_walkers import BaseWalkers


class ThermofieldWalkers(BaseWalkers):
    """Open thermofield Gaussian walkers.

    Each walker is an M x M matrix per spin, Delta[iw, s], representing the
    (unnormalized) purified state |Phi(Delta)> = exp(c^dagger Delta
    \\tilde{c}^dagger)|0 \\tilde{0}>.  Walkers are initialized at Delta = I
    (the identity purification, i.e. infinite temperature) with unit weight
    and are propagated only on the physical side, Delta <- B(x) Delta.

    To preserve the small scales of the accumulated propagator product (the
    standard FT-AFQMC stratification problem), each walker is stored in QDT
    factored form

        Delta[iw, s] = Qmat[iw, s] diag(exp(log_d[iw, s])) Tmat[iw, s],

    where Qmat and Tmat stay bounded and log_d carries the scales in the log
    domain.  Propagation left-multiplies Qmat; `stabilize` refactors with a
    column-pivoted QR (White et al. stratification), leaving the physical
    state unchanged.

    Cached quantities:
    - `log_ovlp[iw]`: complex log of the fixed-guide overlap S_T(Delta_iw).
    - `Ga/Gb[iw]`: mixed Green's function in the thermal *hole* convention
      G_ij = <c_i c_j^dagger> = I - P.T (P the transition 1-RDM), matching
      what `ipie.addons.thermal.propagation.force_bias.construct_force_bias`
      consumes.
    """

    def __init__(
        self,
        trial: ThermofieldThermalTrial,
        nbasis: int,
        nwalkers: int,
        mpi_handler=None,
        verbose: bool = False,
    ):
        assert isinstance(trial, ThermofieldThermalTrial)
        super().__init__(nwalkers, verbose=verbose)

        self.nbasis = nbasis
        self.mpi_handler = mpi_handler

        self.Qmat = numpy.zeros((nwalkers, 2, nbasis, nbasis), dtype=numpy.complex128)
        self.Qmat[:] = numpy.eye(nbasis)
        self.log_d = numpy.zeros((nwalkers, 2, nbasis), dtype=numpy.float64)
        self.Tmat = numpy.zeros((nwalkers, 2, nbasis, nbasis), dtype=numpy.complex128)
        self.Tmat[:] = numpy.eye(nbasis)
        # Complex log of the guide overlap (overrides the real base array).
        self.log_ovlp = numpy.zeros(nwalkers, dtype=numpy.complex128)

        self.Ga = numpy.zeros((nwalkers, nbasis, nbasis), dtype=numpy.complex128)
        self.Gb = numpy.zeros((nwalkers, nbasis, nbasis), dtype=numpy.complex128)
        self._identity = numpy.eye(nbasis)

        for iw in range(nwalkers):
            self.update_walker_cache(trial, iw)

        self.buff_names += ["Qmat", "log_d", "Tmat", "Ga", "Gb"]
        self.buff_size = round(self.set_buff_size_single_walker() / float(self.nwalkers))
        self.walker_buffer = numpy.zeros(self.buff_size, dtype=numpy.complex128)

    def get_delta(self, iw):
        """Physical walker matrices Delta[iw], shape (2, nbasis, nbasis).

        May overflow for extreme log_d; intended for estimators and tests at
        moderate beta.
        """
        return numpy.array(
            [
                self.Qmat[iw, s] @ (numpy.exp(self.log_d[iw, s])[:, None] * self.Tmat[iw, s])
                for s in range(2)
            ]
        )

    def set_delta(self, iw, Delta, trial=None):
        """Set walker `iw` from plain matrices Delta (2, nbasis, nbasis)."""
        self.Qmat[iw] = Delta
        self.log_d[iw] = 0.0
        self.Tmat[iw] = numpy.eye(self.nbasis)
        if trial is not None:
            self.update_walker_cache(trial, iw)

    def update_walker_cache(self, trial, iw):
        """Recompute cached overlap and Green's functions of walker `iw`."""
        log_ovlp, G = trial.calc_log_overlap_and_greens_function(
            self.Qmat[iw], self.log_d[iw], self.Tmat[iw]
        )
        self.set_walker_cache(iw, log_ovlp, G)

    def set_walker_cache(self, iw, log_ovlp, G):
        """Install a precomputed guide overlap and transition Green's function."""
        # Hole convention: Ga = I - P.T with P = G_spec.T, i.e. Ga = I - G_spec.
        self.Ga[iw] = self._identity - G[0]
        self.Gb[iw] = self._identity - G[1]
        self.log_ovlp[iw] = log_ovlp

    def stabilize(self):
        """Refactor Delta = Q D T by column-pivoted QR (physical no-op).

        Absorbs the accumulated propagator factors in Qmat into a fresh
        orthonormal Q, log-domain scales D, and a bounded triangular-times-
        permutation factor merged into T.
        """
        for iw in range(self.nwalkers):
            for s in range(2):
                d = self.log_d[iw, s]
                shift = numpy.max(d)
                F = self.Qmat[iw, s] * numpy.exp(d - shift)[None, :]
                Q, R, perm = scipy.linalg.qr(F, pivoting=True, check_finite=False)
                diag = R.diagonal().copy()
                # Guard exactly singular factors (dead directions).
                absd = numpy.abs(diag)
                absd[absd == 0.0] = 1.0
                S = numpy.zeros_like(R)
                S[:, perm] = (1.0 / absd)[:, None] * R
                self.Qmat[iw, s] = Q
                self.log_d[iw, s] = numpy.log(absd) + shift
                self.Tmat[iw, s] = S @ self.Tmat[iw, s]

    def reset(self, trial):
        """Reset all walkers to the identity purification with unit weight."""
        self.weight = numpy.ones(self.nwalkers)
        self.unscaled_weight = numpy.ones(self.nwalkers)
        self.phase = numpy.ones(self.nwalkers, dtype=numpy.complex128)
        self.Qmat[:] = 0.0
        self.Qmat[:] = numpy.eye(self.nbasis)
        self.log_d[:] = 0.0
        self.Tmat[:] = 0.0
        self.Tmat[:] = numpy.eye(self.nbasis)
        for iw in range(self.nwalkers):
            self.update_walker_cache(trial, iw)

    # For compatibility with the BaseWalkers interface (stabilization is the
    # QDT refactorization above, not orbital reorthogonalization).
    def reortho(self):
        pass

    def reortho_batched(self):
        pass
