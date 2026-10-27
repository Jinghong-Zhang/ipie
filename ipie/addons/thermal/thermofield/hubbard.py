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
"""Names of the earlier flat module, now in the hamiltonians, trial, propagation and qmc subpackages"""

# pylint: disable=unused-import
from ipie.addons.thermal.thermofield.estimators.cyclic_average import cyclic_average_hubbard
from ipie.addons.thermal.thermofield.hamiltonians.hubbard import (
    build_hubbard_hamiltonian,
    hopping_matrix,
    pinning_field,
)
from ipie.addons.thermal.thermofield.propagation.hubbard import (
    ThermofieldHubbardCP,
    update_trace_weight,
)
from ipie.addons.thermal.thermofield.qmc.hubbard_afqmc import propagate_population_cp
from ipie.addons.thermal.thermofield.trial.hubbard import (
    fit_hubbard_density_trial,
    fit_hubbard_trial,
    rhf_k_trial,
    thermal_ip_scf,
    thermal_uhf_scf,
    ThermofieldCPTrial,
)

_trace_weight_update = update_trace_weight
wrap_measure_fullpath = cyclic_average_hubbard
