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
"""Analytical (forward-mode, hand-derived) gradient of the phaseless AFQMC energy.

This addon computes dE/dlambda for H(lambda) = H0 + lambda*O (O a one-body
observable) by propagating tangent (sensitivity) quantities alongside the
walkers, with no automatic differentiation framework.  The algorithm mirrors
ipie.addons.adafqmc step for step so that, driven by identical auxiliary
fields, the two implementations agree to floating-point accuracy.  Runtime
dependencies are numpy and scipy only.
"""
