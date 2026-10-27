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


from ipie.addons.thermal.estimators.generic import local_energy_generic_cholesky
from ipie.addons.thermal.estimators.thermal import one_rdm_from_G
from ipie.estimators.energy import EnergyEstimator
from ipie.utils.backend import arraylib as xp


def local_energy_thermofield_mixed(hamiltonian, walkers):
    """Mixed local energies of all walkers with the fixed guide.

    Parameters
    ----------
    hamiltonian :
        Hamiltonian.
    walkers : :class:`ThermofieldWalkers`
        Walkers with cached Green's functions.

    Returns
    -------
    energies : :class:`numpy.ndarray`
        Total, one-body (with `ecore`) and two-body energies, shape (nwalkers, 3).
    """
    energies = xp.zeros((walkers.nwalkers, 3), dtype=xp.complex128)
    for iw in range(walkers.nwalkers):
        P = one_rdm_from_G(xp.array([walkers.Ga[iw], walkers.Gb[iw]]))
        energies[iw] = local_energy_generic_cholesky(hamiltonian, P)
    return energies


class ThermofieldEnergyEstimator(EnergyEstimator):
    """Mixed energy with the fixed guide.

    The one-sided mixed estimator, not the finite-temperature energy.
    """

    def __init__(self, hamiltonian=None, trial=None, filename=None):
        super().__init__(system=None, ham=hamiltonian, trial=trial, filename=filename)

    def compute_estimator(self, system=None, walkers=None, hamiltonian=None, trial=None):
        if hamiltonian is None:
            raise ValueError("Hamiltonian must not be none.")
        if walkers is None:
            raise ValueError("Walkers must not be none.")
        energy = local_energy_thermofield_mixed(hamiltonian, walkers)
        weights = walkers.weight * walkers.phase
        self._data["ENumer"] = xp.sum(weights * energy[:, 0])
        self._data["EDenom"] = xp.sum(weights)
        self._data["E1Body"] = xp.sum(weights * energy[:, 1])
        self._data["E2Body"] = xp.sum(weights * energy[:, 2])
        return self.data
