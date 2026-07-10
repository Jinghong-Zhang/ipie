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
"""Phaseless propagation of open thermofield walkers.

Reuses the FT-AFQMC Hubbard-Stratonovich conventions of
:class:`ipie.addons.thermal.propagation.phaseless_generic.PhaselessGeneric`:

    B(x) = BH1 exp(VHS) BH1,
    BH1 = exp(-dt/2 (h1e_mod - i sum_n L_n mf_shift_n - mu I)),
    VHS = i sqrt(dt) sum_n x_n L_n,
    xbar = -sqrt(dt) (i vbias - mf_shift),   vbias_n = sum_s Tr(L_n P_s),

with P the mixed transition 1-RDM relative to the *fixed* target-beta guide.
The only change relative to the trace algorithm is the propagation rule
Delta <- B(x) Delta and the importance function, which uses the fixed-guide
overlap ratio

    S_T(Delta_new) / S_T(Delta_old),   S_T = prod_s det(I + D_T[s]^dag Delta_s),

instead of the closed-trace det(I + B_path) ratio.  The weight update mirrors
`PhaselessBase.update_weight_legacy`: with

    log R = log S_T(Delta') - log S_T(Delta) + cfb + cmf,
    cfb = xi . xbar - xbar . xbar / 2,      cmf = -sqrt(dt) xshifted . mf_shift,

the walker weight is multiplied by |exp(-dt mf_core) exp(log R)| and the
cosine projection uses dtheta = Arg exp(log R - cfb).  With phaseless=False
(debug/free projection) the phase is accumulated in walkers.phase instead of
being projected.
"""

import cmath
import math

import numpy

from ipie.addons.thermal.propagation.phaseless_generic import PhaselessGeneric
from ipie.propagation.operations import apply_exponential
from ipie.utils.backend import arraylib as xp


class ThermofieldPhaseless(PhaselessGeneric):
    """Phaseless propagator for thermofield walkers, Delta <- B(x) Delta."""

    def __init__(self, time_step, mu, exp_nmax=6, phaseless=True, verbose=False):
        super().__init__(time_step, mu, exp_nmax=exp_nmax, lowrank=False, verbose=verbose)
        # phaseless=False keeps complex weights (weight * phase) for
        # free-projection debugging on small systems.
        self.phaseless = phaseless

    def propagate_walkers(self, walkers, hamiltonian, trial, eshift=0.0, debug=False):
        """Propagate all walkers by one time slice.

        The inherited `construct_two_body_propagator` supplies the force bias
        (from walkers.Ga/Gb, which cache the mixed Green's function relative
        to the fixed guide), the shifted fields, and VHS.
        """
        cmf, cfb, xshifted, VHS = self.construct_two_body_propagator(
            walkers, hamiltonian, trial, debug=debug
        )
        assert walkers.nwalkers == xshifted.shape[-1]

        for iw in range(walkers.nwalkers):
            phi = xp.identity(VHS[iw].shape[-1], dtype=xp.complex128)
            BV = apply_exponential(phi, VHS[iw], self.exp_nmax)  # exp(VHS[iw]).
            # Delta <- B Delta acts on the Q factor of the QDT-factored walker.
            Q_new = numpy.zeros_like(walkers.Qmat[iw])
            for s in range(2):
                B = self.BH1[s] @ BV @ self.BH1[s]  # Symmetric Trotter split.
                Q_new[s] = B @ walkers.Qmat[iw, s]

            log_ovlp_new, G_new = trial.calc_log_overlap_and_greens_function(
                Q_new, walkers.log_d[iw], walkers.Tmat[iw]
            )
            self.update_walker_weight(walkers, iw, log_ovlp_new, cfb[iw], cmf[iw])

            # Commit the new walker state and refresh cached overlap and
            # mixed Green's function (consumed by the force bias next slice).
            walkers.Qmat[iw] = Q_new
            walkers.set_walker_cache(iw, log_ovlp_new, G_new)

    def update_walker_weight(self, walkers, iw, log_ovlp_new, cfb, cmf):
        """Update walker `iw` following the legacy FT-AFQMC hybrid update.

        The overlap ratio is the fixed-guide ratio S_T(Delta')/S_T(Delta)
        evaluated in log space; the constant exp(-dt * mf_core) with
        mf_core = ecore + mf_shift . mf_shift / 2 matches
        `PhaselessBase.update_weight_legacy`.
        """
        log_oratio = log_ovlp_new - walkers.log_ovlp[iw]
        log_R = log_oratio + cfb + cmf

        try:
            expQ = self.mf_const_fac * cmath.exp(log_R)
            magn = abs(expQ)
        except OverflowError:
            magn = math.inf

        if math.isinf(magn) or math.isnan(magn):
            walkers.weight[iw] = 0.0
            return

        if self.phaseless:
            # Cosine phase from Arg(S_T(Delta')/S_T(Delta)) and cmf, excluding
            # the exponential factor from shifting the probability distribution.
            dtheta = cmath.phase(cmath.exp(log_R - cfb))
            cosine_fac = max(0.0, math.cos(dtheta))
            walkers.weight[iw] *= magn * cosine_fac
        else:
            walkers.weight[iw] *= magn
            if magn > 0.0:
                walkers.phase[iw] *= expQ / magn
