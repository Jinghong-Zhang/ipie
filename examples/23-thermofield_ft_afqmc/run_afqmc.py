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
"""Thermofield finite-temperature AFQMC for the Hubbard atom."""

import numpy

from ipie.addons.thermal.thermofield.analysis.thermofield_analysis import summarize
from ipie.addons.thermal.thermofield.qmc.thermofield_afqmc import ThermofieldAFQMC
from ipie.config import MPI
from ipie.hamiltonians.generic import GenericRealChol

comm = MPI.COMM_WORLD
verbose = False if (comm.rank != 0) else True

# 1. Build the Hubbard atom, H = eps (n_up + n_down) + U n_up n_down, with a
# single Cholesky vector sqrt(U).
eps, U = 0.1, 0.5
mu = 0.5
beta = 2.0
hamiltonian = GenericRealChol(
    numpy.array([[[eps]], [[eps]]]), numpy.sqrt(U) * numpy.ones((1, 1)), ecore=0.0
)

# 2. Build thermofield AFQMC driver.
# Every block walks the population from the infinite-temperature purification
# Delta = I over [0, beta] with the fixed guide exp(-beta k_T / 2), and gives the
# mixed estimator with trial time eta at theta = beta - eta, for every eta in
# `eta_members`, and the closed-trace estimator at theta = beta. Free projection
# is exact within statistics for this system.
afqmc = ThermofieldAFQMC.build(
    mu,
    beta,
    hamiltonian,
    nwalkers=128,
    nblocks=20,
    timestep=0.05,
    seed=7,
    phaseless=False,
    eta_members=(beta / 2, beta / 4, beta / 8),
    verbose=verbose,
)

# 3. Run thermofield AFQMC calculation.
results = afqmc.run(verbose=verbose)
afqmc.finalise(verbose)

# 4. Combine the blocks as a ratio of means. The guide k_T = eps - mu ignores U,
# so the mixed estimator at eta = beta / 2 is biased; the extrapolation of the
# members to eta = 0 and the trace estimator are not.
if comm.rank == 0:
    a = numpy.exp(-beta * (eps - mu))
    b = numpy.exp(-beta * U)
    Z = 1.0 + 2.0 * a + a * a * b
    e_exact = (2.0 * eps * a + (2.0 * eps + U) * a * a * b) / Z
    nav_exact = (2.0 * a + 2.0 * a * a * b) / Z

    summary = summarize(results, guide_time=afqmc.trial.guide_time)
    print("# Estimator               E                    N")
    for k, eta in enumerate(summary["members"]["eta"]):
        e, de = summary["members"]["energy"][0][k], summary["members"]["energy"][1][k]
        n, dn = summary["members"]["nav"][0][k], summary["members"]["nav"][1][k]
        label = f"mixed, eta = {eta:g}"
        print(f"# {label:<22} {e: .6f} +/- {de:.6f}  {n: .6f} +/- {dn:.6f}")
    for label, key in (("eta -> 0 extrapolated", "eta_extrapolated"), ("trace", "trace")):
        (e, de), (n, dn) = summary[key]["energy"], summary[key]["nav"]
        print(f"# {label:<22} {e: .6f} +/- {de:.6f}  {n: .6f} +/- {dn:.6f}")
    print(f"# {'exact':<22} {e_exact: .6f}              {nav_exact: .6f}")
