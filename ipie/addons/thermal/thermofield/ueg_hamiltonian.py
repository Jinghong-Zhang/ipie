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
"""Names of the earlier flat module, now in hamiltonians.ueg and estimators.local_energy"""

# pylint: disable=unused-import
from ipie.addons.thermal.propagation.phaseless_base import construct_mean_field_shift
from ipie.addons.thermal.thermofield.estimators.local_energy import (
    local_energy_batched,
    local_energy_generic_cholesky,
)
from ipie.addons.thermal.thermofield.hamiltonians.ueg import UEGSparseChol
from ipie.addons.thermal.thermofield.propagation.phaseless_thermofield import ThermofieldPhaseless

UEGSparseHamiltonian = UEGSparseChol
local_energy_ueg_sparse_batched = local_energy_batched
