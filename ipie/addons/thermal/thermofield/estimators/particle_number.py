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

from ipie.addons.thermal.estimators.particle_number import particle_number
from ipie.addons.thermal.estimators.thermal import one_rdm_from_G
from ipie.estimators.estimator_base import EstimatorBase
from ipie.utils.backend import arraylib as xp


def particle_number_thermofield_mixed(walkers):
    """Mixed particle numbers of all walkers with the fixed guide.

    Parameters
    ----------
    walkers : :class:`ThermofieldWalkers`
        Walkers with cached Green's functions.

    Returns
    -------
    nav : :class:`numpy.ndarray`
        Particle numbers, shape (nwalkers,).
    """
    nav = xp.zeros(walkers.nwalkers, dtype=xp.complex128)
    for iw in range(walkers.nwalkers):
        P = one_rdm_from_G(xp.array([walkers.Ga[iw], walkers.Gb[iw]]))
        nav[iw] = particle_number(P)
    return nav


class ThermofieldNumberEstimator(EstimatorBase):
    """Mixed average particle number with the fixed guide."""

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
