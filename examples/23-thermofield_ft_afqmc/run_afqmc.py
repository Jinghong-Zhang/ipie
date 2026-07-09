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
"""Prototype thermofield-guided finite-temperature AFQMC example.

The walker is an M x M matrix Delta per spin representing the open
thermofield Gaussian |Phi(Delta)> = exp(c^dag Delta \tilde{c}^dag)|0>,
initialized at Delta = I and propagated to theta = beta / 2 with the usual
one-body AFQMC propagator matrices, guided by a *fixed* target-beta thermal
trial D_T = exp(-beta k_T / 2).  The one-sided mixed estimator is only a
diagnostic; the reported finite-temperature observable is the two-replica
estimator.

This example runs the half-filled Hubbard atom, where the grand-canonical
internal energy is known in closed form.
"""

import numpy

from ipie.addons.thermal.thermofield.qmc import ThermofieldAFQMC
from ipie.hamiltonians.generic import GenericRealChol

# Hubbard atom: H = eps (n_up + n_down) + U n_up n_down, via a single
# Cholesky vector L = sqrt(U) (requires U >= 0).
eps, U = 0.1, 2.0
mu = 0.5
beta = 2.0
hamiltonian = GenericRealChol(
    numpy.array([[[eps]], [[eps]]]), numpy.sqrt(U) * numpy.ones((1, 1)), ecore=0.0
)

# Closed-form reference.
a = numpy.exp(-beta * (eps - mu))
b = numpy.exp(-beta * U)
Z = 1.0 + 2.0 * a + a * a * b
e_exact = (2.0 * eps * a + (2.0 * eps + U) * a * a * b) / Z
nav_exact = (2.0 * a + 2.0 * a * a * b) / Z

afqmc = ThermofieldAFQMC.build(
    mu,
    beta,
    hamiltonian,
    nwalkers=128,
    nblocks=20,
    timestep=0.05,
    seed=7,
    phaseless=False,  # Free projection: exact within statistics on this system.
    pairing="random_permutation",
    measure_mode="complex",
    verbose=1,
)
results = afqmc.run()

print(
    f"\n# E_rep   = {numpy.mean(results['e_rep']): .6f} "
    f"+/- {numpy.std(results['e_rep']) / numpy.sqrt(len(results['e_rep'])): .6f}"
)
print(f"# Nav_rep = {numpy.mean(results['nav_rep']): .6f}")
print(f"# Exact   : E = {e_exact: .6f}, Nav = {nav_exact: .6f}")
