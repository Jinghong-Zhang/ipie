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

import numpy as np
import pytest

from ipie.addons.analytical_gradient.utils.fields import RandomFields, ScriptedFields


@pytest.mark.unit
def test_scripted_fields_replays_and_validates():
    rng = np.random.default_rng(5)
    normals = [rng.standard_normal((3, 4)) for _ in range(2)]
    uniforms = [0.25]
    fields = ScriptedFields(normals, uniforms)
    np.testing.assert_array_equal(fields.normal(3, 4), normals[0])
    assert fields.uniform() == 0.25
    with pytest.raises(RuntimeError, match="shape mismatch"):
        fields.normal(2, 4)
    np.testing.assert_array_equal(fields.normal(3, 4), normals[1])
    fields.assert_exhausted()
    with pytest.raises(RuntimeError, match="exhausted"):
        fields.normal(3, 4)


@pytest.mark.unit
def test_scripted_fields_unconsumed_raises():
    fields = ScriptedFields([np.zeros((2, 2))], [])
    with pytest.raises(RuntimeError, match="not fully consumed"):
        fields.assert_exhausted()


@pytest.mark.unit
def test_random_fields_deterministic():
    a = RandomFields(9)
    b = RandomFields(9)
    np.testing.assert_array_equal(a.normal(2, 3), b.normal(2, 3))
    assert a.uniform() == b.uniform()
