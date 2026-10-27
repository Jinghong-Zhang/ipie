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

import numpy
import pytest

from ipie.addons.thermal.estimators.generic import local_energy_generic_cholesky
from ipie.addons.thermal.thermofield.estimators.local_energy import local_energy_batched
from ipie.addons.thermal.thermofield.hamiltonians.ueg import UEGSparseChol
from ipie.addons.thermal.thermofield.propagation.phaseless_thermofield import ThermofieldPhaseless
from ipie.addons.thermal.thermofield.trial.thermofield import ThermofieldThermalTrial
from ipie.addons.thermal.thermofield.walkers.uhf_walkers import ThermofieldWalkers
from ipie.addons.thermal.utils.ueg import UEG
from ipie.hamiltonians.generic import GenericComplexChol


def build_ueg_pair(nup, ndown, rs, ecut):
    # The sparse plane-wave Hamiltonian and the dense complex-Cholesky one of the same UEG.
    ueg = UEG(
        {
            "nup": nup,
            "ndown": ndown,
            "rs": rs,
            "ecut": ecut,
            "thermal": True,
            "write_integrals": False,
        },
        verbose=False,
    )
    ueg.build(verbose=False)
    sparse = UEGSparseChol.from_ueg(ueg)
    h1e = numpy.array([ueg.H1[0], ueg.H1[0]], dtype=numpy.complex128)
    dense = GenericComplexChol(h1e, (2.0 * ueg.chol_vecs).toarray(), ecore=0.0)
    return sparse, dense


@pytest.mark.unit
def test_sparse_factors_match_dense():
    # System params.
    nup = 3
    ndown = 3
    rs = 1.0
    ecut = 2.0

    sparse, dense = build_ueg_pair(nup, ndown, rs, ecut)
    assert sparse.nfields == dense.nfields
    numpy.testing.assert_allclose(sparse.A.toarray(), dense.A, atol=1e-14)
    numpy.testing.assert_allclose(sparse.B.toarray(), dense.B, atol=1e-14)
    numpy.testing.assert_allclose(sparse.h1e_mod, dense.h1e_mod, atol=1e-12)


@pytest.mark.unit
def test_local_energy_matches_dense():
    # System params.
    nup = 3
    ndown = 3
    rs = 1.0
    ecut = 2.0
    nwalkers = 4
    seed = 7

    sparse, dense = build_ueg_pair(nup, ndown, rs, ecut)
    nbasis = sparse.nbasis
    numpy.random.seed(seed)
    shape = (nwalkers, 2, nbasis, nbasis)
    P = numpy.random.standard_normal(shape) + 1j * numpy.random.standard_normal(shape)
    ref = numpy.array([local_energy_generic_cholesky(dense, P[iw]) for iw in range(nwalkers)])
    scalar = numpy.array([local_energy_generic_cholesky(sparse, P[iw]) for iw in range(nwalkers)])
    numpy.testing.assert_allclose(local_energy_batched(sparse, P), ref, atol=1e-10)
    numpy.testing.assert_allclose(scalar, ref, atol=1e-10)


@pytest.mark.unit
def test_propagation_matches_dense():
    # System params.
    nup = 3
    ndown = 3
    rs = 1.0
    ecut = 2.0

    # Thermofield AFQMC params.
    beta = 1.0
    timestep = 0.05
    nwalkers = 4
    nslices = 6
    seed = 7

    # The same fields give the same walkers with the sparse and the dense
    # factors: the mean-field shift, the one-body propagator, the force bias
    # and VHS agree to rounding.
    sparse, dense = build_ueg_pair(nup, ndown, rs, ecut)
    nbasis = sparse.nbasis
    eps = numpy.diagonal(sparse.h1e_mod[0]).real
    mu = 0.5 * (numpy.sort(eps)[nup - 1] + numpy.sort(eps)[nup])
    k_trial = numpy.array([numpy.diag(eps) - mu * numpy.eye(nbasis)] * 2)
    runs = []
    for hamiltonian in (sparse, dense):
        trial = ThermofieldThermalTrial(hamiltonian, beta, mu=mu, k_trial=k_trial, guide_time=beta)
        walkers = ThermofieldWalkers(trial, nbasis, nwalkers)
        propagator = ThermofieldPhaseless(timestep, mu)
        propagator.build(hamiltonian, trial=trial, walkers=walkers)
        numpy.random.seed(seed)
        for _ in range(nslices):
            propagator.propagate_walkers(walkers, hamiltonian, trial)
        runs.append((propagator, walkers))
    (prop_s, walkers_s), (prop_d, walkers_d) = runs
    numpy.testing.assert_allclose(prop_s.mf_shift, prop_d.mf_shift, atol=1e-12)
    numpy.testing.assert_allclose(prop_s.BH1, prop_d.BH1, atol=1e-12)
    numpy.testing.assert_allclose(walkers_s.weight, walkers_d.weight, rtol=1e-9)
    numpy.testing.assert_allclose(walkers_s.Qmat, walkers_d.Qmat, atol=1e-9)
    numpy.testing.assert_allclose(walkers_s.log_ovlp, walkers_d.log_ovlp, atol=1e-9)


if __name__ == "__main__":
    test_sparse_factors_match_dense()
    test_local_energy_matches_dense()
    test_propagation_matches_dense()
