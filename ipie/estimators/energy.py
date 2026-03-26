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
# Author: Fionn Malone <fmalone@google.com>
#

from typing import Union

import numpy

import plum

from ipie.estimators.estimator_base import EstimatorBase
from ipie.estimators.local_energy_batch import (
    local_energy_batch,
    local_energy_multi_det_trial_batch,
)
from ipie.estimators.local_energy_noci import local_energy_noci
from ipie.estimators.local_energy_sd import (
    local_energy_single_det_uhf_batch,
    local_energy_single_det_ghf_batch,
)
from ipie.estimators.local_energy_sd_isdf import local_energy_single_det_isdf_batch_gpu
from ipie.estimators.local_energy_sd_chunked import (
    local_energy_single_det_uhf_batch_isdf_chunked_gpu,
)
from ipie.estimators.local_energy_wicks import (
    local_energy_multi_det_trial_wicks_batch,
    local_energy_multi_det_trial_wicks_batch_opt,
    local_energy_multi_det_trial_wicks_batch_opt_chunked,
)
from ipie.estimators.local_energy_kpt_sd import local_energy_kpt_single_det_uhf
from ipie.estimators.local_energy_kpt_sd_isdf import local_energy_kpt_single_det_uhf_isdf_gpu
from ipie.estimators.local_energy_kpt_sd_chunked import local_energy_kpt_single_det_uhf_chunked
from ipie.hamiltonians.generic import GenericComplexChol, GenericRealChol
from ipie.hamiltonians.isdf import GenericRealISDF
from ipie.hamiltonians.generic_chunked import GenericRealCholChunked
from ipie.hamiltonians.chunked_isdf import GenericRealISDFChunked
from ipie.systems.generic import Generic
from ipie.trial_wavefunction.noci import NOCI
from ipie.trial_wavefunction.particle_hole import (
    ParticleHole,
    ParticleHoleNaive,
    ParticleHoleNonChunked,
    ParticleHoleSlow,
)
from ipie.trial_wavefunction.single_det_kpt import KptSingleDet
from ipie.hamiltonians.kpt_hamiltonian import KptComplexChol, KptComplexCholSymm
from ipie.hamiltonians.kpt_isdf_hamiltonian import KptISDF
from ipie.hamiltonians.kpt_chunked import KptComplexCholChunked
from ipie.walkers.uhf_walkers import UHFWalkers
from ipie.trial_wavefunction.single_det import SingleDet
from ipie.trial_wavefunction.single_det_ghf import SingleDetGHF
from ipie.utils.backend import arraylib as xp
from ipie.walkers.ghf_walkers import GHFWalkers
from ipie.walkers.correlated_walkers import CorrelatedWalkers


@plum.dispatch
def local_energy(
    system: Generic,
    hamiltonian: Union[GenericRealChol, GenericRealCholChunked],
    walkers: UHFWalkers,
    trial: SingleDet,
):
    return local_energy_batch(system, hamiltonian, walkers, trial)


@plum.dispatch
def local_energy(
    system: Generic,
    hamiltonian: GenericComplexChol,
    walkers: UHFWalkers,
    trial: SingleDet,
):
    return local_energy_single_det_uhf_batch(system, hamiltonian, walkers, trial)


@plum.dispatch
def local_energy(
    system: Generic,
    hamiltonian: GenericRealChol,
    walkers: UHFWalkers,
    trial: ParticleHoleNaive,
):
    return local_energy_multi_det_trial_batch(system, hamiltonian, walkers, trial)


@plum.dispatch
def local_energy(
    system: Generic,
    hamiltonian: GenericRealChol,
    walkers: UHFWalkers,
    trial: ParticleHole,
):
    return local_energy_multi_det_trial_wicks_batch_opt_chunked(system, hamiltonian, walkers, trial)


@plum.dispatch
def local_energy(
    system: Generic,
    hamiltonian: GenericRealChol,
    walkers: UHFWalkers,
    trial: ParticleHoleNonChunked,
):
    return local_energy_multi_det_trial_wicks_batch_opt(system, hamiltonian, walkers, trial)


@plum.dispatch
def local_energy(
    system: Generic,
    hamiltonian: GenericRealChol,
    walkers: UHFWalkers,
    trial: ParticleHoleSlow,
):
    return local_energy_multi_det_trial_wicks_batch(system, hamiltonian, walkers, trial)


@plum.dispatch
def local_energy(system: Generic, hamiltonian: GenericRealChol, walkers: UHFWalkers, trial: NOCI):
    return local_energy_noci(system, hamiltonian, walkers, trial)


@plum.dispatch
def local_energy(
    system: Generic, hamiltonian: KptComplexChol, walkers: UHFWalkers, trial: KptSingleDet
):
    return local_energy_kpt_single_det_uhf(system, hamiltonian, walkers, trial)


@plum.dispatch
def local_energy(
    system: Generic, hamiltonian: KptComplexCholSymm, walkers: UHFWalkers, trial: KptSingleDet
):
    return local_energy_kpt_single_det_uhf(system, hamiltonian, walkers, trial)


@plum.dispatch
def local_energy(
    system: Generic, hamiltonian: KptComplexCholChunked, walkers: UHFWalkers, trial: KptSingleDet
):
    return local_energy_kpt_single_det_uhf_chunked(system, hamiltonian, walkers, trial)


@plum.dispatch
def local_energy(system: Generic, hamiltonian: KptISDF, walkers: UHFWalkers, trial: KptSingleDet):
    return local_energy_kpt_single_det_uhf_isdf_gpu(system, hamiltonian, walkers, trial)


@plum.dispatch
def local_energy(
    system: Generic, hamiltonian: GenericRealChol, walkers: GHFWalkers, trial: SingleDetGHF
):
    return local_energy_single_det_ghf_batch(system, hamiltonian, walkers, trial)


@plum.dispatch
def local_energy(
    system: Generic, hamiltonian: GenericComplexChol, walkers: GHFWalkers, trial: SingleDetGHF
):
    return local_energy_single_det_ghf_batch(system, hamiltonian, walkers, trial)


@plum.dispatch
def local_energy(
    system: Generic, hamiltonian: GenericRealISDF, walkers: UHFWalkers, trial: SingleDet
):
    return local_energy_single_det_isdf_batch_gpu(system, hamiltonian, walkers, trial)


@plum.dispatch
def local_energy(
    system: Generic, hamiltonian: GenericRealISDFChunked, walkers: UHFWalkers, trial: SingleDet
):
    return local_energy_single_det_uhf_batch_isdf_chunked_gpu(system, hamiltonian, walkers, trial)


class EnergyEstimator(EstimatorBase):
    def __init__(
        self,
        system=None,
        ham=None,
        trial=None,
        filename=None,
    ):
        super().__init__()
        self._eshift = 0.0
        self.scalar_estimator = True
        self._data = {
            "ENumer": 0.0j,
            "EDenom": 0.0j,
            "ETotal": 0.0j,
            "E1Body": 0.0j,
            "E2Body": 0.0j,
        }
        self._shape = (len(self.names),)
        self._data_index = {k: i for i, k in enumerate(list(self._data.keys()))}
        self.print_to_stdout = True
        self.ascii_filename = filename

    def compute_estimator(self, system=None, walkers=None, hamiltonian=None, trial=None):
        trial.calc_greens_function(walkers)
        # Need to be able to dispatch here
        energy = local_energy(system, hamiltonian, walkers, trial)
        self._data["ENumer"] = xp.sum(walkers.weight * energy[:, 0].real)
        self._data["EDenom"] = xp.sum(walkers.weight)
        self._data["E1Body"] = xp.sum(walkers.weight * energy[:, 1].real)
        self._data["E2Body"] = xp.sum(walkers.weight * energy[:, 2].real)

        return self.data

    def get_index(self, name):
        index = self._data_index.get(name, None)
        if index is None:
            raise RuntimeError(f"Unknown estimator {name}")
        return index

    def post_reduce_hook(self, data):
        ix_proj = self._data_index["ETotal"]
        ix_nume = self._data_index["ENumer"]
        ix_deno = self._data_index["EDenom"]
        data[ix_proj] = data[ix_nume] / data[ix_deno]
        ix_nume = self._data_index["E1Body"]
        data[ix_nume] = data[ix_nume] / data[ix_deno]
        ix_nume = self._data_index["E2Body"]
        data[ix_nume] = data[ix_nume] / data[ix_deno]


class CorrelatedEnergyEstimator(EstimatorBase):
    """Minimal estimator for weighted energy difference in correlated runs.

    This estimator expects paired inputs for ``system``, ``hamiltonian`` and
    ``trial``, along with a ``CorrelatedWalkers`` instance. It computes
    ``DeltaE = E_B - E_A`` with total walker weight
    ``w = w_A * w_B``.
    """

    def __init__(self, filename=None):
        super().__init__()
        self.scalar_estimator = True
        self._data = {
            "EDiffNumer": 0.0j,
            "EDiffDenom": 0.0j,
            "EANumer": 0.0j,
            "EADenom": 0.0j,
            "EBNumer": 0.0j,
            "EBDenom": 0.0j,
            "EDiff": 0.0j,
            "EA": 0.0j,
            "EB": 0.0j,
            "EBMinusEA": 0.0j,
            "E1BodyDiff": 0.0j,
            "E2BodyDiff": 0.0j,
        }
        self._shape = (len(self.names),)
        self._data_index = {k: i for i, k in enumerate(list(self._data.keys()))}
        self.print_to_stdout = False
        self.ascii_filename = filename

    @staticmethod
    def _to_numpy(values):
        if isinstance(values, numpy.ndarray):
            return values
        if hasattr(xp, "asnumpy"):
            return xp.asnumpy(values)
        return numpy.asarray(values)

    @classmethod
    def _first_three(cls, values):
        return cls._to_numpy(values)[:3]

    @staticmethod
    def _safe_ratio(numerator, denominator):
        if abs(denominator) < 1e-14:
            return numpy.nan
        return numerator / denominator

    @staticmethod
    def _unpack_pair(obj, name):
        if isinstance(obj, (tuple, list)) and len(obj) == 2:
            return obj[0], obj[1]
        if isinstance(obj, dict):
            if "A" in obj and "B" in obj:
                return obj["A"], obj["B"]
            if "reference" in obj and "sample" in obj:
                return obj["reference"], obj["sample"]
        raise ValueError(
            f"{name} must be a 2-tuple/list or dict with keys ('A','B') or ('reference','sample')."
        )

    def compute_estimator(self, system=None, walkers=None, hamiltonian=None, trial=None):
        if not isinstance(walkers, CorrelatedWalkers):
            raise ValueError("CorrelatedEnergyEstimator requires a CorrelatedWalkers instance.")

        systemA, systemB = self._unpack_pair(system, "system")
        hamiltonianA, hamiltonianB = self._unpack_pair(hamiltonian, "hamiltonian")
        trialA, trialB = self._unpack_pair(trial, "trial")

        trialA.calc_greens_function(walkers.walkers_A)
        trialB.calc_greens_function(walkers.walkers_B)

        energyA = local_energy(systemA, hamiltonianA, walkers.walkers_A, trialA)
        energyB = local_energy(systemB, hamiltonianB, walkers.walkers_B, trialB)

        ediff = energyB - energyA
        wt = walkers.weight
        wt_a = walkers.walkers_A.weight
        wt_b = walkers.walkers_B.weight

        ediff_numer = xp.sum(wt * ediff[:, 0].real)
        ediff_denom = xp.sum(wt)
        e_a_numer = xp.sum(wt_a * energyA[:, 0].real)
        e_a_denom = xp.sum(wt_a)
        e_b_numer = xp.sum(wt_b * energyB[:, 0].real)
        e_b_denom = xp.sum(wt_b)

        self._data["EDiffNumer"] = ediff_numer
        self._data["EDiffDenom"] = ediff_denom
        self._data["EANumer"] = e_a_numer
        self._data["EADenom"] = e_a_denom
        self._data["EBNumer"] = e_b_numer
        self._data["EBDenom"] = e_b_denom
        self._data["E1BodyDiff"] = xp.sum(wt * ediff[:, 1].real)
        self._data["E2BodyDiff"] = xp.sum(wt * ediff[:, 2].real)
        return self.data

    def post_reduce_hook(self, data):
        ix_diff = self._data_index["EDiff"]
        ix_nume = self._data_index["EDiffNumer"]
        ix_deno = self._data_index["EDiffDenom"]
        data[ix_diff] = self._safe_ratio(data[ix_nume], data[ix_deno])

        ix_ea = self._data_index["EA"]
        ix_ea_nume = self._data_index["EANumer"]
        ix_ea_deno = self._data_index["EADenom"]
        data[ix_ea] = self._safe_ratio(data[ix_ea_nume], data[ix_ea_deno])

        ix_eb = self._data_index["EB"]
        ix_eb_nume = self._data_index["EBNumer"]
        ix_eb_deno = self._data_index["EBDenom"]
        data[ix_eb] = self._safe_ratio(data[ix_eb_nume], data[ix_eb_deno])

        ix_delta = self._data_index["EBMinusEA"]
        data[ix_delta] = data[ix_eb] - data[ix_ea]

        ix_e1 = self._data_index["E1BodyDiff"]
        ix_e2 = self._data_index["E2BodyDiff"]
        data[ix_e1] = self._safe_ratio(data[ix_e1], data[ix_deno])
        data[ix_e2] = self._safe_ratio(data[ix_e2], data[ix_deno])
