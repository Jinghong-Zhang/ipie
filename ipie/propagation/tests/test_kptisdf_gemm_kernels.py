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
# Author: Jinghong Zhang <jinghongzhang@fas.harvard.edu>
#

"""Parity tests for the GEMM-based KptISDF propagation and force-bias kernels.

The kernels are backend agnostic, so these tests run on CPU with numpy and
compare the optimized implementations against the retained reference paths
and direct einsum evaluations.
"""

import numpy
import pytest

from ipie.propagation.force_bias import (
    _contract_force_bias_spin_sum,
    _contract_force_bias_x,
    contract_qkPp_kPr_qkwpr_to_qwP_cupy,
)
from ipie.propagation.phaseless_kpt import (
    construct_full_l_batch_for_gemm,
    construct_full_Lx_batch,
    contract_lowmem_vhs_walkers,
    contract_lowmem_vhs_walkers_from_l_batch,
)


def complex_rand(rng, shape):
    return rng.standard_normal(shape) + 1.0j * rng.standard_normal(shape)


def kpoint_maps(nk):
    q = numpy.arange(nk)[:, None]
    k = numpy.arange(nk)[None, :]
    return (k + q) % nk, (k - q) % nk


def make_vhs_inputs(rng, nw, nk, nq, nisdf, nbsf, nocc):
    kpq_mat, kmq_mat = kpoint_maps(nk)
    unique_qs = numpy.arange(nq)
    cgto = complex_rand(rng, (nk, nisdf, nbsf))
    Lx = complex_rand(rng, (nw, nq, nisdf))
    Lconjx = complex_rand(rng, (nw, nq, nisdf))
    phi = complex_rand(rng, (nw, nk * nbsf, nk * nocc))
    return kpq_mat, kmq_mat, unique_qs, cgto, Lx, Lconjx, phi


def vhs_from_l_batch(cgto, Lx, Lconjx, phi, kpq_mat, kmq_mat, unique_qs, max_mem=4.0):
    nw, nq, nisdf = Lx.shape
    nk = kpq_mat.shape[1]
    nbsf = cgto.shape[2]
    nocc = phi.shape[-1] // nk
    l_batch = construct_full_l_batch_for_gemm(Lx, Lconjx, kpq_mat, kmq_mat, unique_qs)
    phi_for_cgto = (
        phi.reshape(nw, nk, nbsf, nk, nocc).transpose(1, 2, 0, 3, 4).reshape(nk, nbsf, nw * nk * nocc)
    )
    out = contract_lowmem_vhs_walkers_from_l_batch(
        l_batch, cgto, phi_for_cgto, nw, nk, nisdf, nbsf, nocc, max_mem=max_mem
    )
    return out.reshape(nw, nk * nbsf, nk * nocc)


@pytest.mark.unit
def test_vhs_l_batch_kernel_matches_old_path():
    rng = numpy.random.default_rng(7)
    nw, nk, nq, nisdf, nbsf, nocc = 3, 4, 3, 11, 5, 2
    kpq_mat, kmq_mat, unique_qs, cgto, Lx, Lconjx, phi = make_vhs_inputs(
        rng, nw, nk, nq, nisdf, nbsf, nocc
    )

    full_Lx = construct_full_Lx_batch(Lx, kpq_mat, unique_qs)
    full_Lconjx = construct_full_Lx_batch(Lconjx, kmq_mat, unique_qs)
    fullLpLconjx = full_Lx + full_Lconjx
    phi_reshape = phi.reshape(nw, nk, nbsf, nk, nocc)
    out_old = contract_lowmem_vhs_walkers(
        fullLpLconjx, cgto, phi_reshape, nw, nk, nisdf, nbsf, nocc
    ).reshape(nw, nk * nbsf, nk * nocc)

    out_einsum = numpy.einsum(
        "wKkP,kPp,KPr,wKrQi->wkpQi",
        fullLpLconjx,
        cgto.conj(),
        cgto,
        phi_reshape,
        optimize=True,
    ).reshape(nw, nk * nbsf, nk * nocc)

    out_new = vhs_from_l_batch(cgto, Lx, Lconjx, phi, kpq_mat, kmq_mat, unique_qs)
    # tiny max_mem forces the internal ISDF chunking
    out_new_chunked = vhs_from_l_batch(
        cgto, Lx, Lconjx, phi, kpq_mat, kmq_mat, unique_qs, max_mem=1e-7
    )

    numpy.testing.assert_allclose(out_old, out_einsum, atol=1e-10)
    numpy.testing.assert_allclose(out_new, out_old, atol=1e-10)
    numpy.testing.assert_allclose(out_new_chunked, out_old, atol=1e-10)


@pytest.mark.unit
def test_vhs_kernel_spin_concatenation():
    # validates the fused alpha/beta Taylor expansion in apply_VHS: spins
    # concatenated along the occupied index propagate independently
    rng = numpy.random.default_rng(11)
    nw, nk, nq, nisdf, nbsf = 2, 4, 4, 9, 5
    nocca, noccb = 2, 1
    kpq_mat, kmq_mat, unique_qs, cgto, Lx, Lconjx, phia = make_vhs_inputs(
        rng, nw, nk, nq, nisdf, nbsf, nocca
    )
    phib = complex_rand(rng, (nw, nk * nbsf, nk * noccb))

    out_a = vhs_from_l_batch(cgto, Lx, Lconjx, phia, kpq_mat, kmq_mat, unique_qs)
    out_b = vhs_from_l_batch(cgto, Lx, Lconjx, phib, kpq_mat, kmq_mat, unique_qs)

    phi_comb = numpy.concatenate(
        (
            phia.reshape(nw, nk * nbsf, nk, nocca),
            phib.reshape(nw, nk * nbsf, nk, noccb),
        ),
        axis=3,
    ).reshape(nw, nk * nbsf, nk * (nocca + noccb))
    out_comb = vhs_from_l_batch(
        cgto, Lx, Lconjx, phi_comb, kpq_mat, kmq_mat, unique_qs
    ).reshape(nw, nk * nbsf, nk, nocca + noccb)

    numpy.testing.assert_allclose(
        out_comb[:, :, :, :nocca].reshape(nw, nk * nbsf, nk * nocca), out_a, atol=1e-10
    )
    numpy.testing.assert_allclose(
        out_comb[:, :, :, nocca:].reshape(nw, nk * nbsf, nk * noccb), out_b, atol=1e-10
    )


@pytest.mark.unit
def test_force_bias_x_kernel_matches_einsum():
    rng = numpy.random.default_rng(13)
    nq, nk, nisdf, nocc, nw, nbasis = 3, 4, 10, 2, 3, 5
    rcgto = complex_rand(rng, (nq, nk, nisdf, nocc))
    halfrot = complex_rand(rng, (nk, nisdf, nbasis))
    g = complex_rand(rng, (nq, nk, nw, nocc, nbasis))

    ref = numpy.einsum("qkPp,kPr,qkwpr->qwP", rcgto, halfrot, g, optimize=True)
    out = contract_qkPp_kPr_qkwpr_to_qwP_cupy(rcgto, halfrot, g, max_mem=4.0)
    # tiny max_mem forces both the walker and ISDF chunking
    out_chunked = contract_qkPp_kPr_qkwpr_to_qwP_cupy(rcgto, halfrot, g, max_mem=1e-8)

    numpy.testing.assert_allclose(out, ref, atol=1e-10)
    numpy.testing.assert_allclose(out_chunked, ref, atol=1e-10)


@pytest.mark.unit
def test_force_bias_spin_sum_matches_separate():
    rng = numpy.random.default_rng(17)
    nq, nk, nisdf, nw, nbasis = 3, 4, 10, 3, 5
    nocca, noccb = 2, 1
    rcgtoa = complex_rand(rng, (nq, nk, nisdf, nocca))
    rcgtob = complex_rand(rng, (nq, nk, nisdf, noccb))
    halfrot = complex_rand(rng, (nk, nisdf, nbasis))
    ga = complex_rand(rng, (nq, nk, nw, nocca, nbasis))
    gb = complex_rand(rng, (nq, nk, nw, noccb, nbasis))

    ref = _contract_force_bias_x(rcgtoa, halfrot, ga, 4.0) + _contract_force_bias_x(
        rcgtob, halfrot, gb, 4.0
    )
    out = _contract_force_bias_spin_sum(rcgtoa, ga, rcgtob, gb, halfrot, 4.0)

    numpy.testing.assert_allclose(out, ref, atol=1e-10)


if __name__ == "__main__":
    test_vhs_l_batch_kernel_matches_old_path()
    test_vhs_kernel_spin_concatenation()
    test_force_bias_x_kernel_matches_einsum()
    test_force_bias_spin_sum_matches_separate()
