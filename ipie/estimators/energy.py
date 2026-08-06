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

import plum

from ipie.estimators.estimator_base import EstimatorBase
from ipie.estimators.local_energy_batch import (
    local_energy_batch,
    local_energy_multi_det_trial_batch,
)
from ipie.estimators.local_energy_noci import local_energy_noci
from ipie.estimators.local_energy_sd import local_energy_single_det_uhf, local_energy_single_det_ghf
from ipie.estimators.local_energy_wicks import (
    local_energy_multi_det_trial_wicks_batch,
    local_energy_multi_det_trial_wicks_batch_opt,
    local_energy_multi_det_trial_wicks_batch_opt_chunked,
)
from ipie.estimators.local_energy_kpt_sd import local_energy_kpt_single_det_uhf
from ipie.estimators.local_energy_kpt_sd_chunked import local_energy_kpt_single_det_uhf_chunked
from ipie.hamiltonians.generic import GenericComplexChol, GenericRealChol
from ipie.hamiltonians.generic_chunked import GenericRealCholChunked
from ipie.hamiltonians.thc import GenericRealTHC, GenericRealTHCUhf, GenericComplexTHC
from ipie.lno_thc import frag_2body_thc, _env_flag
from ipie.lno_thc_uhf import local_energy_thc_uhf
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
from ipie.hamiltonians.kpt_chunked import KptComplexCholChunked 
from ipie.walkers.uhf_walkers import UHFWalkers
from ipie.trial_wavefunction.single_det import SingleDet
from ipie.trial_wavefunction.single_det_ghf import SingleDetGHF
from ipie.utils.backend import arraylib as xp
import numpy
from ipie.walkers.uhf_walkers import UHFWalkers
from ipie.walkers.ghf_walkers import GHFWalkers

from ipie.utils.backend import get_device_memory


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
    return local_energy_single_det_uhf(system, hamiltonian, walkers, trial)


def local_energy_thc(system, hamiltonian, walkers, trial):
    """Local energy for the factored THC Hamiltonian.  e1 from the half-rotated
    one-body integrals; e2 from frag_2body_thc.

    The phaseless weight update and population-control eshift use the HYBRID energy
    (overlap ratio + force bias), NOT this local energy, so when only the per-fragment
    observable is needed the full cluster two-body (n_frag = nocc, the dominant
    O(nocc*Nmu^2) cost) is wasteful.  If `hamiltonian.n_frag` is set, e2 is the cheap
    fragment two-body (n_frag occ, fused exchange); otherwise the full cluster e2.
    ETotal is then an embedded (e1 + fragment-e2) diagnostic; the physical observable
    is EFragCorr from the LNO estimator.

    LNO_FAST_ESTIMATOR=1: this default estimator's e2 is SKIPPED (set to zero) --
    it exactly duplicated the LNOFrag additional estimator's fragment two-body
    every block.  E2Body/ETotal then report e1-only diagnostics; the physical
    observable (EFragCorr from the LNOFrag estimator) is unaffected."""
    # Stay on the active backend (xp = numpy on CPU, cupy on GPU): the walker
    # half-rotated densities live on the GPU under use_gpu, and numpy.asarray on a
    # cupy array raises (implicit host conversion forbidden).  xp.asarray is a no-op
    # on CPU and keeps the contraction on-device on GPU.
    Ga = xp.asarray(walkers.Ghalfa)                # (nw, nalpha, nbasis)
    rhf = bool(getattr(walkers, "rhf", False))
    # Under rhf walkers, phib/Ghalfb are never propagated/recomputed: beta == alpha.
    Gb = Ga if rhf else xp.asarray(walkers.Ghalfb)  # (nw, nbeta, nbasis)
    nocc = Ga.shape[1]
    nfrag = getattr(hamiltonian, "n_frag", None) or nocc
    rH1a = xp.asarray(trial._rH1a)
    rH1b = xp.asarray(trial._rH1b)
    e1 = xp.einsum("ij,wij->w", rH1a, Ga) + xp.einsum("ij,wij->w", rH1b, Gb)
    if _env_flag("LNO_FAST_ESTIMATOR"):
        e2 = xp.zeros(walkers.nwalkers, dtype=numpy.complex128)
    else:
        e2 = frag_2body_thc(
            hamiltonian, trial._thc_Xocca, trial._thc_Xoccb, Ga, None if rhf else Gb, nfrag
        )
    energy = xp.zeros((walkers.nwalkers, 3), dtype=numpy.complex128)
    energy[:, 1] = e1
    energy[:, 2] = e2
    energy[:, 0] = hamiltonian.ecore + e1 + e2
    return xp.array(energy)


@plum.dispatch
def local_energy(
    system: Generic,
    hamiltonian: GenericRealTHC,
    walkers: UHFWalkers,
    trial: SingleDet,
):
    return local_energy_thc(system, hamiltonian, walkers, trial)


def local_energy_thc_cx(system, hamiltonian, walkers, trial):
    """Local energy for the COMPLEX factored THC Hamiltonian (no-TRS metal path).

    e1 from the half-rotated one-body integrals; e2 is the FULL-CLUSTER complex
    two-body.  As on the real path, the phaseless weight update uses the HYBRID
    energy (overlap ratio + force bias), not this local energy, so under
    LNO_FAST_ESTIMATOR=1 the (expensive, O(Nmu^2)) e2 is skipped and E2Body/ETotal
    become e1-only diagnostics; the physical observable is the fragment estimator."""
    from ipie.lno_thc_cx import two_body_energy_thc_cx

    Ga = xp.asarray(walkers.Ghalfa)
    rhf = bool(getattr(walkers, "rhf", False))
    Gb = Ga if rhf else xp.asarray(walkers.Ghalfb)
    rH1a = xp.asarray(trial._rH1a)
    rH1b = xp.asarray(trial._rH1b)
    e1 = xp.einsum("ij,wij->w", rH1a, Ga) + xp.einsum("ij,wij->w", rH1b, Gb)
    if _env_flag("LNO_FAST_ESTIMATOR"):
        e2 = xp.zeros(walkers.nwalkers, dtype=numpy.complex128)
    else:
        e2 = two_body_energy_thc_cx(
            hamiltonian, trial._thc_Xocca, trial._thc_Xoccac, Ga, None if rhf else Gb
        )
    energy = xp.zeros((walkers.nwalkers, 3), dtype=numpy.complex128)
    energy[:, 1] = e1
    energy[:, 2] = e2
    energy[:, 0] = hamiltonian.ecore + e1 + e2
    return xp.array(energy)


@plum.dispatch
def local_energy(
    system: Generic,
    hamiltonian: GenericComplexTHC,
    walkers: UHFWalkers,
    trial: SingleDet,
):
    return local_energy_thc_cx(system, hamiltonian, walkers, trial)


@plum.dispatch
def local_energy(
    system: Generic,
    hamiltonian: GenericRealTHCUhf,
    walkers: UHFWalkers,
    trial: SingleDet,
):
    # Per-spin-basis THC (more specific than the GenericRealTHC overload above).
    return local_energy_thc_uhf(system, hamiltonian, walkers, trial)


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
def local_energy(system: Generic, hamiltonian: KptComplexChol, walkers: UHFWalkers, trial: KptSingleDet):
    return local_energy_kpt_single_det_uhf(system, hamiltonian, walkers, trial)

@plum.dispatch
def local_energy(system: Generic, hamiltonian: KptComplexCholSymm, walkers: UHFWalkers, trial: KptSingleDet):
    return local_energy_kpt_single_det_uhf(system, hamiltonian, walkers, trial)

@plum.dispatch
def local_energy(system: Generic, hamiltonian: KptComplexCholChunked, walkers: UHFWalkers, trial: KptSingleDet):
    return local_energy_kpt_single_det_uhf_chunked(system, hamiltonian, walkers, trial)

@plum.dispatch
def local_energy(
    system: Generic, hamiltonian: GenericRealChol, walkers: GHFWalkers, trial: SingleDetGHF
):
    return local_energy_single_det_ghf(system, hamiltonian, walkers, trial)


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
