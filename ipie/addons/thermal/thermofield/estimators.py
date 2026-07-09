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
"""Estimators for thermofield-guided FT-AFQMC.

Two estimators are provided:

1. The one-sided *mixed* estimator relative to the fixed target-beta guide,
   E_mix = sum_i w_i E_L^T(Delta_i) / sum_i w_i.  This is a cheap diagnostic
   and low-temperature/projector estimator; it is NOT the exact
   finite-temperature Gibbs estimator.

2. The *two-replica* estimator, which is the intended finite-temperature
   estimator: two independent populations L and R are propagated from
   Delta = I to theta = beta / 2 with the same fixed guide, and

       E_rep = sum_{(i,j)} w_i^L w_j^R Q_ij E_ij / sum_{(i,j)} w_i^L w_j^R Q_ij,
       Q_ij  = S_ij^{LR} / [S_T(Delta_i^L)^* S_T(Delta_j^R)],
       S_ij^{LR} = prod_s det(I + Delta_i^{L dagger} Delta_j^R),

   with the pair local energy evaluated from the replica transition Green's
   function G_ij^{LR} = Delta_j^R (I + Delta_i^{L dagger} Delta_j^R)^{-1}
   Delta_i^{L dagger}.  Local energies use hamiltonian.H1 (i.e. H, not
   K = H - mu N), so the reported quantity is the internal energy.
"""

from typing import Union

import numpy

from ipie.addons.thermal.estimators.generic import local_energy_generic_cholesky
from ipie.addons.thermal.estimators.particle_number import particle_number
from ipie.addons.thermal.estimators.thermal import one_rdm_from_G
from ipie.addons.thermal.thermofield.gaussian import (
    factored_pair_greens_function,
    factored_pair_log_overlap,
)
from ipie.addons.thermal.thermofield.walkers import ThermofieldWalkers
from ipie.estimators.energy import EnergyEstimator
from ipie.estimators.estimator_base import EstimatorBase
from ipie.hamiltonians.generic import GenericComplexChol, GenericRealChol
from ipie.utils.backend import arraylib as xp


def local_energy_thermofield_mixed(
    hamiltonian: Union[GenericRealChol, GenericComplexChol], walkers: ThermofieldWalkers
):
    """Mixed local energies E_L^T(Delta_i) of all walkers.

    Uses the cached mixed Green's functions (relative to the fixed guide) and
    the standard ipie Cholesky local energy, which implements the transition
    Wick contraction ecoul - exx for the 2-body term.

    Returns
    -------
    energies : :class:`numpy.ndarray`
        Shape (nwalkers, 3): (ETotal, E1Body + ecore, E2Body).
    """
    energies = xp.zeros((walkers.nwalkers, 3), dtype=xp.complex128)
    for iw in range(walkers.nwalkers):
        P = one_rdm_from_G(xp.array([walkers.Ga[iw], walkers.Gb[iw]]))
        energies[iw] = local_energy_generic_cholesky(hamiltonian, P)
    return energies


def particle_number_thermofield_mixed(walkers: ThermofieldWalkers):
    """Mixed particle numbers of all walkers."""
    nav = xp.zeros(walkers.nwalkers, dtype=xp.complex128)
    for iw in range(walkers.nwalkers):
        P = one_rdm_from_G(xp.array([walkers.Ga[iw], walkers.Gb[iw]]))
        nav[iw] = particle_number(P)
    return nav


class ThermofieldEnergyEstimator(EnergyEstimator):
    """One-sided mixed energy estimator (diagnostic only).

    NOT the exact finite-temperature Gibbs estimator; see
    :func:`replica_energy_estimate` for the intended finite-T estimator.
    """

    def __init__(self, hamiltonian=None, trial=None, filename=None):
        super().__init__(system=None, ham=hamiltonian, trial=trial, filename=filename)

    def compute_estimator(self, system=None, walkers=None, hamiltonian=None, trial=None):
        if hamiltonian is None:
            raise ValueError("Hamiltonian must not be none.")
        if walkers is None:
            raise ValueError("walkers must not be none.")
        energy = local_energy_thermofield_mixed(hamiltonian, walkers)
        weights = walkers.weight * walkers.phase
        self._data["ENumer"] = xp.sum(weights * energy[:, 0])
        self._data["EDenom"] = xp.sum(weights)
        self._data["E1Body"] = xp.sum(weights * energy[:, 1])
        self._data["E2Body"] = xp.sum(weights * energy[:, 2])
        return self.data


class ThermofieldNumberEstimator(EstimatorBase):
    """Mixed average particle number estimator."""

    def __init__(self, hamiltonian=None, trial=None, filename=None):
        super().__init__()
        self._data = {"NavNumer": 0.0j, "NavDenom": 0.0j, "Nav": 0.0j}
        self._shape = (len(self.names),)
        self._data_index = {k: i for i, k in enumerate(list(self._data.keys()))}
        self.print_to_stdout = True
        self.ascii_filename = filename
        self.scalar_estimator = True

    def compute_estimator(self, system=None, walkers=None, hamiltonian=None, trial=None):
        nav = particle_number_thermofield_mixed(walkers)
        weights = walkers.weight * walkers.phase
        self._data["NavNumer"] = xp.sum(weights * nav)
        self._data["NavDenom"] = xp.sum(weights)
        return self.data

    def get_index(self, name):
        index = self._data_index.get(name, None)
        if index is None:
            raise RuntimeError(f"Unknown estimator {name}")
        return index

    def post_reduce_hook(self, data):
        ix_numer = self._data_index["NavNumer"]
        ix_denom = self._data_index["NavDenom"]
        ix_nav = self._data_index["Nav"]
        data[ix_nav] = data[ix_numer] / data[ix_denom]


def build_replica_pairs(
    nwalkers_left, nwalkers_right, pairing="random_permutation", rng=None, num_partners=1
):
    """Build the pair set P for the two-replica estimator.

    Pairing options:
    - "all_pairs": O(Nw^2), tests / small systems only.
    - "random_permutation": O(Nw), default for production.
    - "k_random_partners": O(k Nw), optional variance reduction (uses
      `num_partners` independent random permutations).
    """
    if pairing == "all_pairs":
        return [(i, j) for i in range(nwalkers_left) for j in range(nwalkers_right)]
    assert nwalkers_left == nwalkers_right, "Permutation pairing needs equal population sizes."
    if rng is None:
        rng = numpy.random.default_rng()
    if pairing == "random_permutation":
        perm = rng.permutation(nwalkers_right)
        return [(i, perm[i]) for i in range(nwalkers_left)]
    if pairing == "k_random_partners":
        pairs = []
        for _ in range(num_partners):
            perm = rng.permutation(nwalkers_right)
            pairs += [(i, perm[i]) for i in range(nwalkers_left)]
        return pairs
    raise ValueError(f"Unknown pairing option: {pairing}")


def replica_pair_quantities(hamiltonian, walkers_left, walkers_right, i, j):
    """Overlap and local observables of the replica pair (i, j).

    Returns
    -------
    log_S : complex
        sum_s log det(I + Delta_i^{L dagger} Delta_j^R) including the walkers'
        gauge exponents.
    energy : tuple
        (ETotal, E1Body + ecore, E2Body) from the replica transition 1-RDM,
        using hamiltonian.H1 (the internal energy, not H - mu N).
    nav : complex
        Tr of the replica transition 1-RDM.
    """
    nbasis = walkers_left.nbasis
    log_S = 0.0 + 0.0j
    P = numpy.zeros((2, nbasis, nbasis), dtype=numpy.complex128)
    for s in range(2):
        # QDT-factored pair formulas; the left walker enters as Delta_L^dagger
        # (its real log scales are unchanged by conjugation).
        args = (
            walkers_left.Qmat[i, s],
            walkers_left.log_d[i, s],
            walkers_left.Tmat[i, s],
            walkers_right.Qmat[j, s],
            walkers_right.log_d[j, s],
            walkers_right.Tmat[j, s],
        )
        log_S += factored_pair_log_overlap(*args)
        P[s] = factored_pair_greens_function(*args).T
    energy = local_energy_generic_cholesky(hamiltonian, P)
    nav = particle_number(P)
    return log_S, energy, nav


def replica_energy_estimate(
    hamiltonian,
    walkers_left,
    walkers_right,
    pairing="random_permutation",
    mode="complex",
    rng=None,
    num_partners=1,
    free_projection=False,
):
    """Two-replica thermofield estimate of internal energy and particle number.

    This is the intended finite-temperature estimator: both populations must
    have been propagated from Delta = I to theta = beta / 2 with the same
    fixed target-beta guide.

    Parameters
    ----------
    mode : str
        "complex": keep complex Q_ij and take the real part of the final
        ratio (exact reweighting).
        "phaseless": replace Q_ij by |Q_ij| max(0, cos(arg Q_ij)).
        "mixed": fall back to the one-sided mixed estimator on the right
        population.
    free_projection : bool
        Multiply weights by the accumulated walker phases (conjugated for the
        left/bra population), for phaseless=False debug runs.

    Returns
    -------
    results : dict
        etotal, e1b, e2b, nav (real block estimates), and the raw complex
        numer / denom (with the common log normalization removed).
    """
    if mode == "mixed":
        energies = local_energy_thermofield_mixed(hamiltonian, walkers_right)
        weights = walkers_right.weight.copy().astype(numpy.complex128)
        if free_projection:
            weights *= walkers_right.phase
        denom = numpy.sum(weights)
        nav = particle_number_thermofield_mixed(walkers_right)
        return {
            "etotal": (numpy.sum(weights * energies[:, 0]) / denom).real,
            "e1b": (numpy.sum(weights * energies[:, 1]) / denom).real,
            "e2b": (numpy.sum(weights * energies[:, 2]) / denom).real,
            "nav": (numpy.sum(weights * nav) / denom).real,
            "numer": numpy.sum(weights * energies[:, 0]),
            "numer_e1b": numpy.sum(weights * energies[:, 1]),
            "numer_e2b": numpy.sum(weights * energies[:, 2]),
            "numer_nav": numpy.sum(weights * nav),
            "denom": denom,
            "log_shift": numpy.float64(0.0),
        }

    pairs = build_replica_pairs(
        walkers_left.nwalkers,
        walkers_right.nwalkers,
        pairing=pairing,
        rng=rng,
        num_partners=num_partners,
    )

    npairs = len(pairs)
    log_q = numpy.zeros(npairs, dtype=numpy.complex128)
    w_pair = numpy.zeros(npairs, dtype=numpy.complex128)
    energies = numpy.zeros((npairs, 3), dtype=numpy.complex128)
    navs = numpy.zeros(npairs, dtype=numpy.complex128)

    for ip, (i, j) in enumerate(pairs):
        log_S, energy, nav = replica_pair_quantities(hamiltonian, walkers_left, walkers_right, i, j)
        # Q_ij = S_ij / (S_T(Delta_i^L)^* S_T(Delta_j^R)) in log space.
        log_q[ip] = log_S - numpy.conj(walkers_left.log_ovlp[i]) - walkers_right.log_ovlp[j]
        w_ij = walkers_left.weight[i] * walkers_right.weight[j]
        if free_projection:
            # The left walker enters as a bra: conjugate its phase.
            w_ij = w_ij * numpy.conj(walkers_left.phase[i]) * walkers_right.phase[j]
        w_pair[ip] = w_ij
        energies[ip] = energy
        navs[ip] = nav

    # Remove the common scale before exponentiating (log-sum-exp).
    shift = numpy.max(log_q.real)
    q = numpy.exp(log_q - shift)
    if mode == "phaseless":
        q = numpy.abs(q) * numpy.maximum(0.0, numpy.cos(numpy.angle(q)))
    elif mode != "complex":
        raise ValueError(f"Unknown replica measurement mode: {mode}")

    wq = w_pair * q
    denom = numpy.sum(wq)
    numer = numpy.sum(wq * energies[:, 0])
    return {
        "etotal": (numer / denom).real,
        "e1b": (numpy.sum(wq * energies[:, 1]) / denom).real,
        "e2b": (numpy.sum(wq * energies[:, 2]) / denom).real,
        "nav": (numpy.sum(wq * navs) / denom).real,
        # Raw complex sums (all carrying the common factor exp(-log_shift)):
        # combining across MPI ranks must use these, rescaled to a shared
        # shift, NOT the real-part ratios above.
        "numer": numer,
        "numer_e1b": numpy.sum(wq * energies[:, 1]),
        "numer_e2b": numpy.sum(wq * energies[:, 2]),
        "numer_nav": numpy.sum(wq * navs),
        "denom": denom,
        "log_shift": shift,
    }
