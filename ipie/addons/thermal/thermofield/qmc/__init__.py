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

# Names of the earlier flat module `qmc`.
# pylint: disable=unused-import
from ipie.addons.thermal.thermofield.analysis.thermofield_analysis import (
    extrapolate_eta,
    pooled_ratio,
    summarize,
)
from ipie.addons.thermal.thermofield.estimators.local_energy import local_energy_batched
from ipie.addons.thermal.thermofield.qmc.thermofield_afqmc import ThermofieldAFQMC
from ipie.utils.backend import to_host


def local_energies(hamiltonian, P):
    """Local energies of a stack of 1RDMs on the host. Use `local_energy_batched`."""
    return to_host(local_energy_batched(hamiltonian, P))
