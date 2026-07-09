# Copyright 2026 The ipie Developers. All Rights Reserved.
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
# Author: Jinghong Zhang <jinghongzhang@fas.harvard.edu>
#
"""Prototype thermofield-guided finite-temperature AFQMC.

The sampled walker is an open thermofield Gaussian represented by an M x M
matrix Delta per spin,

    |Phi(Delta)> = exp(c^dagger Delta \tilde{c}^dagger) |0 \tilde{0}>,

initialized at Delta = I (infinite temperature) and propagated only on the
physical side, Delta <- B(x) Delta, using the standard ipie FT-AFQMC one-body
propagator matrices.  The phaseless guide is a *fixed* target-temperature
thermal one-body trial D_T = exp(-beta k_T / 2); the guiding overlap is
det(I + D_T^dagger Delta) per spin, NOT the closed-trace det(I + B_path).

The one-sided mixed estimator is a cheap diagnostic and is *not* the exact
finite-temperature Gibbs estimator.  The intended finite-temperature
estimator is the two-replica estimator in
:mod:`ipie.addons.thermal.thermofield.estimators`.
"""
