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

from ipie.addons.thermal.thermofield.estimators.cyclic_average import (
    cyclic_average_coarse,
    cyclic_average_full_path,
)
from ipie.addons.thermal.thermofield.utils.testing import build_noninteracting_test_case_handlers
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import to_host


def random_path(nwalkers, nslices, nbasis, spin_shared, scale=0.35):
    # Slice propagators B = exp(-E / 2) (I + A) exp(-E / 2) with random A and a
    # wide spread of scales E = diag(eps), shape (nwalkers, nslices, 2, nbasis, nbasis).
    eps = numpy.linspace(-1.5, 3.0, nbasis)
    expeps = numpy.diag(numpy.exp(-0.5 * eps))
    B = numpy.empty((nwalkers, nslices, 2, nbasis, nbasis), dtype=numpy.complex128)
    for iw in range(nwalkers):
        for l in range(nslices):
            for s in range(2):
                if s == 1 and spin_shared:
                    B[iw, l, 1] = B[iw, l, 0]
                    continue
                A = numpy.random.standard_normal((nbasis, nbasis))
                A = scale * (A + 1j * numpy.random.standard_normal((nbasis, nbasis)))
                B[iw, l, s] = expeps @ (numpy.eye(nbasis) + A) @ expeps
    return B


def group_products(B, stride, nspin):
    # Group products Bg[k, s] = B_{k stride + stride - 1} ... B_{k stride}.
    nwalkers, nslices, _, nbasis, _ = B.shape
    ngroups = (nslices + stride - 1) // stride
    Bg = numpy.empty((ngroups, nspin, nwalkers, nbasis, nbasis), dtype=numpy.complex128)
    for k in range(ngroups):
        for s in range(nspin):
            acc = numpy.tile(numpy.eye(nbasis, dtype=numpy.complex128), (nwalkers, 1, 1))
            for l in range(k * stride, min((k + 1) * stride, nslices)):
                acc = B[:, l, s] @ acc
            Bg[k, s] = acc
    return Bg


def collect_fine(weight, B, stride, stab_every=20):
    out = {}

    def measure_walker(iw, G, w):
        out[(iw, w)] = G.copy()

    dev, ld0 = cyclic_average_full_path(weight, B, stride, measure_walker, stab_every=stab_every)
    return dev, ld0, out


def collect_coarse(weight, Bg, stride, stab_every=1):
    out = {}

    def measure_batch(ids, G, k):
        G = to_host(G)
        for j, iw in enumerate(ids):
            out[(int(iw), k * stride)] = G[j].copy()

    dev, ld0 = cyclic_average_coarse(weight, Bg, measure_batch, stab_every=stab_every)
    return dev, ld0, out


@pytest.mark.unit
def test_coarse_engine_reproduces_fine_engine_slice_by_slice():
    # System params.
    nbasis = 5
    nslices = 12

    # Thermofield AFQMC params.
    nwalkers = 3
    stab_every = 4
    seed = 3

    # With one slice per group the coarse sweep is the fine sweep, bit for bit.
    weight = numpy.array([1.0, 0.0, 0.7])  # Walker 1 is skipped.
    for spin_shared in (False, True):
        numpy.random.seed(seed)
        B = random_path(nwalkers, nslices, nbasis, spin_shared)
        dev_f, ld0_f, G_f = collect_fine(weight, B, 1, stab_every=stab_every)
        nspin = 1 if spin_shared else 2
        Bg = numpy.ascontiguousarray(B.transpose(1, 2, 0, 3, 4))[:, :nspin]
        dev_c, ld0_c, G_c = collect_coarse(weight, xp.asarray(Bg), 1, stab_every=stab_every)
        assert numpy.isnan(ld0_f[1]) and numpy.isnan(ld0_c[1])
        for iw in (0, 2):
            assert ld0_c[iw] == ld0_f[iw]
            for w in range(nslices):
                assert numpy.array_equal(G_c[(iw, w)], G_f[(iw, w)])
        assert dev_c == dev_f


@pytest.mark.unit
def test_group_products_match_fine_cuts():
    # System params.
    nbasis = 5
    nslices = 12

    # Thermofield AFQMC params.
    nwalkers = 2
    stride = 3
    seed = 11

    # Products of stride consecutive slices reproduce the fine cuts at the same positions.
    weight = numpy.ones(nwalkers)
    for spin_shared in (False, True):
        numpy.random.seed(seed)
        B = random_path(nwalkers, nslices, nbasis, spin_shared)
        dev_f, ld0_f, G_f = collect_fine(weight, B, stride, stab_every=3)
        nspin = 1 if spin_shared else 2
        Bg = group_products(B, stride, nspin)
        dev_c, ld0_c, G_c = collect_coarse(weight, xp.asarray(Bg), stride, stab_every=1)
        numpy.testing.assert_allclose(ld0_c, ld0_f, rtol=1e-11, atol=1e-11)
        for iw in range(nwalkers):
            for w in range(0, nslices, stride):
                numpy.testing.assert_allclose(G_c[(iw, w)], G_f[(iw, w)], rtol=1e-9, atol=1e-11)
        # The trace weight is a cyclic invariant across the coarse cuts.
        assert dev_c < 1e-10


@pytest.mark.unit
def test_coarse_engine_matches_direct_inverse():
    # System params.
    nbasis = 6
    nslices = 9

    # Thermofield AFQMC params.
    nwalkers = 3
    stride = 3
    seed = 7

    # Direct evaluation of G = I - (I + P_k S_k)^{-1} at every coarse cut; runs
    # on the device when ipie is configured for the GPU. I + P_k S_k has a
    # condition number up to about 1e6, which limits the direct reference.
    weight = numpy.ones(nwalkers)
    ngroups = nslices // stride
    eye = numpy.eye(nbasis)
    for spin_shared in (False, True):
        numpy.random.seed(seed)
        B = random_path(nwalkers, nslices, nbasis, spin_shared)
        nspin = 1 if spin_shared else 2
        Bg = group_products(B, stride, nspin)
        dev, ld0, G = collect_coarse(weight, xp.asarray(Bg), stride)
        assert dev < 1e-10
        for iw in range(nwalkers):
            log_det = 0.0
            for s in range(2):
                sp = 0 if spin_shared else s
                for k in range(ngroups):
                    # Prefix P_k = Bg_{k-1} ... Bg_0 and suffix S_k = Bg_{K-1} ... Bg_k.
                    P = eye.astype(numpy.complex128)
                    for j in range(k):
                        P = Bg[j, sp, iw] @ P
                    S = eye.astype(numpy.complex128)
                    for j in range(k, ngroups):
                        S = Bg[j, sp, iw] @ S
                    G_ref = eye - numpy.linalg.inv(eye + P @ S)
                    numpy.testing.assert_allclose(G[(iw, k * stride)][s], G_ref, atol=1e-8)
                    if k == 0:
                        sign, logabs = numpy.linalg.slogdet(eye + S)
                        log_det += logabs + numpy.log(sign)
            assert ld0[iw].real == pytest.approx(log_det.real, rel=1e-10)
            assert numpy.exp(1j * ld0[iw].imag) == pytest.approx(numpy.exp(1j * log_det.imag))


@pytest.mark.unit
def test_propagator_group_accumulator_matches_recorded_slices():
    # System params.
    nbasis = 4

    # Thermofield AFQMC params.
    mu = 0.1
    beta = 1.0
    timestep = 0.05
    nwalkers = 3
    nslices = 5
    seed = 5

    objs = build_noninteracting_test_case_handlers(nbasis, mu, beta, timestep, nwalkers=nwalkers)
    hamiltonian = objs["hamiltonian"]
    trial = objs["trial"]
    walkers = objs["walkers"]
    propagator = objs["propagator"]

    # The group accumulator collects exactly the applied slice propagators, B_t ... B_1.
    acc = numpy.tile(numpy.eye(nbasis, dtype=numpy.complex128), (nwalkers, 2, 1, 1))
    propagator.group_accum = acc
    rec = numpy.empty((nslices, nwalkers, 2, nbasis, nbasis), dtype=numpy.complex128)
    ref = numpy.tile(numpy.eye(nbasis, dtype=numpy.complex128), (nwalkers, 2, 1, 1))
    numpy.random.seed(seed)
    for t in range(nslices):
        propagator.record_B_slice = rec[t]
        propagator.propagate_walkers(walkers, hamiltonian, trial)
        ref = rec[t] @ ref
    propagator.record_B_slice = None
    propagator.group_accum = None
    assert numpy.array_equal(acc, ref)
    assert not propagator.group_accum_spin_mismatch


if __name__ == "__main__":
    test_coarse_engine_reproduces_fine_engine_slice_by_slice()
    test_group_products_match_fine_cuts()
    test_coarse_engine_matches_direct_inverse()
    test_propagator_group_accumulator_matches_recorded_slices()
