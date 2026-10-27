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
"""Names of the earlier flat module, now in estimators.cyclic_average and utils.linalg"""

# pylint: disable=unused-import
from ipie.addons.thermal.thermofield.estimators.cyclic_average import (
    build_slice_propagators,
    cyclic_average_coarse,
    cyclic_average_engine,
    cyclic_average_fields,
    cyclic_average_full_path,
)
from ipie.addons.thermal.thermofield.utils.linalg import (
    _floor_scales,
    _inverse_permutation,
    _pivoted_qr_stack,
    qdt_left_restratify,
    qdt_left_restratify_stack,
    qdt_right_restratify,
    qdt_right_restratify_stack,
    restratify_left_batched,
    restratify_right_batched,
)

_wrap_engine = cyclic_average_engine
wrap_measure_fullpath_batched = cyclic_average_full_path
wrap_measure_fullpath_fields = cyclic_average_fields
wrap_measure_coarse = cyclic_average_coarse


def _restratify_left_xp(Q, d, T, xp):
    return restratify_left_batched(Q, d, T)


def _restratify_right_xp(Q, d, T, xp):
    return restratify_right_batched(Q, d, T)
