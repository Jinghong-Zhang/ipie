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
"""Injectable random-field sources.

The propagator and stochastic reconfiguration never draw randomness
themselves; every draw flows through one of these objects.  Consumption
contract: exactly one (nwalkers, nchol) standard-normal array per propagation
step and exactly one uniform scalar per stochastic-reconfiguration event, in
program order.  ScriptedFields replays a prerecorded stream, which is how
common-random-number finite differences and cross-code (torch) parity tests
guarantee identical fields.
"""

import numpy as np


class RandomFields:
    """Standard RNG-backed field source."""

    def __init__(self, seed):
        self._rng = np.random.default_rng(seed)
        self.n_normal = 0
        self.n_uniform = 0

    def normal(self, nwalkers, nchol):
        self.n_normal += 1
        return self._rng.standard_normal((nwalkers, nchol))

    def uniform(self):
        self.n_uniform += 1
        return float(self._rng.random())


class ScriptedFields:
    """Plays back a fixed stream of normals and uniforms, validating shapes."""

    def __init__(self, normals, uniforms):
        self._normals = [np.asarray(x, dtype=np.float64) for x in normals]
        self._uniforms = [float(u) for u in uniforms]
        self.n_normal = 0
        self.n_uniform = 0

    def normal(self, nwalkers, nchol):
        if self.n_normal >= len(self._normals):
            raise RuntimeError("ScriptedFields: normal stream exhausted")
        x = self._normals[self.n_normal]
        if x.shape != (nwalkers, nchol):
            raise RuntimeError(
                f"ScriptedFields: shape mismatch, requested {(nwalkers, nchol)}, "
                f"stream entry {self.n_normal} has {x.shape}"
            )
        self.n_normal += 1
        return x

    def uniform(self):
        if self.n_uniform >= len(self._uniforms):
            raise RuntimeError("ScriptedFields: uniform stream exhausted")
        u = self._uniforms[self.n_uniform]
        self.n_uniform += 1
        return u

    def assert_exhausted(self):
        if self.n_normal != len(self._normals) or self.n_uniform != len(self._uniforms):
            raise RuntimeError(
                f"ScriptedFields: stream not fully consumed "
                f"({self.n_normal}/{len(self._normals)} normals, "
                f"{self.n_uniform}/{len(self._uniforms)} uniforms)"
            )
