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
"""Names of the earlier flat module, now in utils.linalg and estimators.greens_function"""

# pylint: disable=unused-import
import numpy

from ipie.addons.thermal.thermofield.estimators.greens_function import (
    thermofield_greens_function,
    thermofield_log_overlap,
    thermofield_one_rdm,
    thermofield_overlap,
    thermofield_wick_normal_ordered_square,
)
from ipie.addons.thermal.thermofield.utils.linalg import (
    _log_complex_det,
    inv_stack,
    one_plus_qdy,
    one_plus_qdy_batched,
    pivoted_qr,
    principal_log_phase,
    stabilized_inverse_one_plus,
    stabilized_inverse_one_plus_batched,
)

_inv_stack = inv_stack


def array_module(a):
    """numpy, or cupy for a cupy array."""
    if type(a).__module__.startswith("cupy"):
        # pylint: disable=import-error
        import cupy

        return cupy
    return numpy


def to_host(a):
    """A cupy array copied to the host; anything else unchanged."""
    return a.get() if type(a).__module__.startswith("cupy") else a
