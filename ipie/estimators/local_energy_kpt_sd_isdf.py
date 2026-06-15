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

from math import ceil

import numpy
from numba import jit

from ipie.config import config
from ipie.utils.backend import arraylib as xp
from ipie.utils.contract_gf_cgto import (
    contract_gf_cgto12_kpq_k,
    contract_gf_cgto12_k_kpq,
    slice_gf_k_kpq_given_q,
    slice_gf_kpq_k_given_q,
)
from ipie.utils.cuquantum_backend import (
    NetworkOptions_required as NetworkOptions,
    contract_required as contract,
    cutensornet_required as cutensornet,
)


def _kptisdf_local_energy_max_mem_gb(default_fraction=0.25):
    return max(default_fraction * xp.cuda.Device().mem_info[0] / 1024**3, 0.01)


def _kptisdf_effective_max_mem_gb(max_mem, default_fraction=0.25, safety_fraction=0.65):
    free_gb = xp.cuda.Device().mem_info[0] / 1024**3
    requested = (
        _kptisdf_local_energy_max_mem_gb(default_fraction=default_fraction)
        if max_mem is None
        else max(float(max_mem), 0.01)
    )
    return max(min(requested, safety_fraction * free_gb), 0.01)


def _use_lowk_exchange_cupy(nk, nbsf, nocc, nwalker=1):
    # The low-k path wins only for small walker batches. For larger batches,
    # dense GEMM or cuTensorNet was consistently faster in the H200 scans.
    if nwalker > 8:
        return False
    if nk <= 1:
        if 750 <= nbsf < 1500 and nocc != 8:
            return True
        return nwalker == 1 and nbsf >= 1500 and nocc <= 4
    if nk <= 8 and nwalker == 1:
        return nbsf == 200 and nocc >= 16
    return False


def _use_dense_exchange_cupy(nk, nbsf, nocc, nwalker=1):
    # These branches are deliberately table-like: they encode the measured H200
    # crossover regions and leave unmeasured larger-k/multi-walker cases on
    # cuTensorNet.
    if nk <= 1:
        if nwalker >= 64:
            return nbsf <= 1500
        if nbsf < 750:
            return nocc <= 4
        if 750 <= nbsf < 1500:
            return False
        return nwalker > 1 and nocc <= 4

    if nk <= 8:
        if nwalker == 1:
            if nbsf <= 100:
                return nocc <= 25
            if nbsf <= 200:
                return nocc <= 8
            return nbsf <= 500 and nocc <= 4
        if nwalker <= 8:
            return nbsf <= 500 and nocc <= 32
        if nwalker <= 64:
            return (nbsf <= 100 and nocc <= 25) or (nbsf <= 200 and 8 < nocc <= 16)
        if nwalker <= 256:
            return nbsf <= 100 and 8 < nocc <= 25
        return False

    if nk < 64:
        if nwalker <= 8:
            return (nbsf <= 200 and nocc <= 4) or (nbsf <= 128 and nocc <= 25)
        return False

    if nk == 64:
        if nwalker == 1:
            if nbsf <= 100:
                return nocc <= 8
            return nbsf <= 200 and nocc <= 32
        return False

    if nk >= 125:
        if nwalker == 1:
            return (nbsf <= 100 and nocc <= 8) or (nbsf <= 200 and nocc <= 4)
        return False
    return False


def _select_kpt_isdf_exx_kernel_gpu(nk, nbsf, nocc, nwalker=1):
    if _use_lowk_exchange_cupy(nk, nbsf, nocc, nwalker=nwalker):
        return "lowk_cupy", kpt_isdf_exx_kernel_gpu_lowk_cupy
    if _use_dense_exchange_cupy(nk, nbsf, nocc, nwalker=nwalker):
        return "dense_cupy", kpt_isdf_exx_kernel_gpu_dense_cupy
    return "original_cuquantum", kpt_isdf_exx_kernel_gpu_original_cuquantum


def kpt_isdf_ecoul_kernel_gpu(
    MPQ, halfrot_cgtoa, halfrot_cgtob, cgto, Ghalfa_batch, Ghalfb_batch, kpq_mat, Sset, Qplus
):
    nk = cgto.shape[0]
    nwalkers = Ghalfa_batch.shape[0]
    ecoul = xp.zeros(nwalkers, dtype=numpy.complex128)
    handle = cutensornet.create()
    network_opts = NetworkOptions(handle=handle)
    for iq in range(len(Sset)):
        iq_real = Sset[iq]
        MPQ_iq = MPQ[iq]
        ikpq = kpq_mat[iq_real]
        cgto_kpq = cgto[ikpq]
        rcgtoa_kpq = halfrot_cgtoa[ikpq]
        rcgtob_kpq = halfrot_cgtob[ikpq]
        ga_k_kpq = slice_gf_k_kpq_given_q(Ghalfa_batch, iq_real, kpq_mat)
        gb_k_kpq = slice_gf_k_kpq_given_q(Ghalfb_batch, iq_real, kpq_mat)
        v1_wP = contract_gf_cgto12_k_kpq(
            ga_k_kpq, halfrot_cgtoa, cgto_kpq, iq_real, network_opts
        ) + contract_gf_cgto12_k_kpq(gb_k_kpq, halfrot_cgtob, cgto_kpq, iq_real, network_opts)
        del ga_k_kpq
        del gb_k_kpq
        ga_kpq_k = slice_gf_kpq_k_given_q(Ghalfa_batch, iq_real, kpq_mat)
        gb_kpq_k = slice_gf_kpq_k_given_q(Ghalfb_batch, iq_real, kpq_mat)
        v2_wP = contract_gf_cgto12_kpq_k(
            ga_kpq_k, rcgtoa_kpq, cgto, iq_real, network_opts
        ) + contract_gf_cgto12_kpq_k(gb_kpq_k, rcgtob_kpq, cgto, iq_real, network_opts)
        del ga_kpq_k
        del gb_kpq_k
        ecoul += xp.sum((v1_wP @ MPQ_iq) * v2_wP, axis=1)

    for iq in range(len(Sset), len(Sset) + len(Qplus)):
        iq_real = Qplus[iq - len(Sset)]
        MPQ_iq = MPQ[iq]
        ikpq = kpq_mat[iq_real]
        cgto_kpq = cgto[ikpq]
        rcgtoa_kpq = halfrot_cgtoa[ikpq]
        rcgtob_kpq = halfrot_cgtob[ikpq]
        ga_k_kpq = slice_gf_k_kpq_given_q(Ghalfa_batch, iq_real, kpq_mat)
        gb_k_kpq = slice_gf_k_kpq_given_q(Ghalfb_batch, iq_real, kpq_mat)
        v1_wP = contract_gf_cgto12_k_kpq(
            ga_k_kpq, halfrot_cgtoa, cgto_kpq, iq_real, network_opts
        ) + contract_gf_cgto12_k_kpq(gb_k_kpq, halfrot_cgtob, cgto_kpq, iq_real, network_opts)
        del ga_k_kpq
        del gb_k_kpq
        ga_kpq_k = slice_gf_kpq_k_given_q(Ghalfa_batch, iq_real, kpq_mat)
        gb_kpq_k = slice_gf_kpq_k_given_q(Ghalfb_batch, iq_real, kpq_mat)
        v2_wP = contract_gf_cgto12_kpq_k(
            ga_kpq_k, rcgtoa_kpq, cgto, iq_real, network_opts
        ) + contract_gf_cgto12_kpq_k(gb_kpq_k, rcgtob_kpq, cgto, iq_real, network_opts)
        del ga_kpq_k
        del gb_kpq_k
        ecoul += 2.0 * xp.sum((v1_wP @ MPQ_iq) * v2_wP, axis=1)
    cutensornet.destroy(handle)
    return 0.5 * ecoul / nk


def kpt_isdf_ecoul_rhf_kernel_gpu(MPQ, halfrot_cgtoa, cgto, Ghalfa_batch, kpq_mat, Sset, Qplus):
    nk = cgto.shape[0]
    nwalkers = Ghalfa_batch.shape[0]
    ecoul = xp.zeros(nwalkers, dtype=numpy.complex128)
    handle = cutensornet.create()
    network_opts = NetworkOptions(handle=handle)
    for iq in range(len(Sset)):
        iq_real = Sset[iq]
        MPQ_iq = MPQ[iq]
        ikpq = kpq_mat[iq_real]
        cgto_kpq = cgto[ikpq]
        rcgtoa_kpq = halfrot_cgtoa[ikpq]
        ga_k_kpq = slice_gf_k_kpq_given_q(Ghalfa_batch, iq_real, kpq_mat)
        v1_wP = contract_gf_cgto12_k_kpq(ga_k_kpq, halfrot_cgtoa, cgto_kpq, iq_real, network_opts)
        del ga_k_kpq
        ga_kpq_k = slice_gf_kpq_k_given_q(Ghalfa_batch, iq_real, kpq_mat)
        v2_wP = contract_gf_cgto12_kpq_k(ga_kpq_k, rcgtoa_kpq, cgto, iq_real, network_opts)
        del ga_kpq_k
        ecoul += xp.sum((v1_wP @ MPQ_iq) * v2_wP, axis=1)

    for iq in range(len(Sset), len(Sset) + len(Qplus)):
        iq_real = Qplus[iq - len(Sset)]
        MPQ_iq = MPQ[iq]
        ikpq = kpq_mat[iq_real]
        cgto_kpq = cgto[ikpq]
        rcgtoa_kpq = halfrot_cgtoa[ikpq]
        ga_k_kpq = slice_gf_k_kpq_given_q(Ghalfa_batch, iq_real, kpq_mat)
        v1_wP = contract_gf_cgto12_k_kpq(ga_k_kpq, halfrot_cgtoa, cgto_kpq, iq_real, network_opts)
        del ga_k_kpq
        ga_kpq_k = slice_gf_kpq_k_given_q(Ghalfa_batch, iq_real, kpq_mat)
        v2_wP = contract_gf_cgto12_kpq_k(ga_kpq_k, rcgtoa_kpq, cgto, iq_real, network_opts)
        del ga_kpq_k
        ecoul += 2.0 * xp.sum((v1_wP @ MPQ_iq) * v2_wP, axis=1)
    cutensornet.destroy(handle)
    return 2.0 * ecoul / nk


def contract_psi_G_psi(psiiP_slice, psiqQ_slice, G, buff, nP, nQ):
    nw, nk, nocc, nbsf = G.shape[0], G.shape[1], G.shape[2], G.shape[4]
    G = G.transpose(3, 0, 1, 2, 4).reshape(nk, nw * nk * nocc, nbsf)
    size_Gpsi = nw * nocc * nk**2 * nQ
    Gpsi = buff[:size_Gpsi].reshape(nk, nw * nocc * nk, nQ)
    xp.matmul(G, psiqQ_slice, out=Gpsi)
    Gpsi = (
        Gpsi.reshape(nk, nw, nk, nocc, nQ).transpose(2, 1, 4, 0, 3).reshape(nk, nw * nQ * nk, nocc)
    )
    size_TPQ = nk**2 * nw * nP * nQ
    TPQ = buff[:size_TPQ].reshape(nk, nw * nQ * nk, nP)
    xp.matmul(Gpsi, psiiP_slice, out=TPQ)
    return TPQ


def X_contract_cupy(
    halfrot_cgtoa,
    phikr_kpq,
    M_PQ_iq,
    phiki_kpq,
    cgto,
    Ga_chunk,
    G_kpq_kprimepq_chunk,
    buff1,
    buff2,
    slices_isdf,
):
    nw = Ga_chunk.shape[0]
    nk = cgto.shape[0]
    exx = xp.zeros((nw,), dtype=xp.complex128)

    for P in slices_isdf:
        psi_iP_k = halfrot_cgtoa[:, P, :].transpose(0, 2, 1).conj()
        phip_kpq_P = phikr_kpq[:, P, :].transpose(0, 2, 1)
        nP = P.stop - P.start
        for Q in slices_isdf:
            nQ = Q.stop - Q.start
            psi_iQ_kp = cgto[:, Q, :].transpose(0, 2, 1)
            phij_kpq_Q = phiki_kpq[:, Q, :].transpose(0, 2, 1).conj()
            TPQ = contract_psi_G_psi(psi_iP_k, psi_iQ_kp, Ga_chunk, buff1, nP, nQ)
            TQP_kpq = contract_psi_G_psi(
                phij_kpq_Q, phip_kpq_P, G_kpq_kprimepq_chunk, buff2, nQ, nP
            )
            TPQ = (
                TPQ.reshape(nk, nw, nQ, nk, nP)
                .transpose(1, 0, 3, 4, 2)
                .reshape(nw, nk * nk, nP, nQ)
            )
            TQP_kpq = (
                TQP_kpq.reshape(nk, nw, nP, nk, nQ)
                .transpose(1, 3, 0, 2, 4)
                .reshape(nw, nk * nk, nP, nQ)
            )
            TPQ *= TQP_kpq
            Tsq = TPQ.sum(axis=1)
            M_PQ_iq_sliced = M_PQ_iq[P, Q].astype(xp.complex128, copy=False)
            exx += Tsq.reshape(nw, nP * nQ) @ M_PQ_iq_sliced.ravel()
    return exx


def _estimate_exx_cupy_bytes(nwalker, nk, nocc, nbsf, isdf_chunk, itemsize):
    max_dim = max(isdf_chunk, nocc)
    shifted_g = nwalker * nk**2 * nocc * nbsf * itemsize
    tpq_buffers = 2 * nwalker * nk**2 * max_dim**2 * itemsize
    gpsi_scratch = 2 * nwalker * nk**2 * nocc * isdf_chunk * itemsize
    return int(1.25 * (shifted_g + tpq_buffers + gpsi_scratch))


def _choose_exx_cupy_chunks(nwalker, nk, nocc, nbsf, nisdf, itemsize, max_mem):
    max_mem_bytes = int(max(float(max_mem), 0.01) * 1024**3)
    walker_chunk = nwalker
    while (
        walker_chunk > 1
        and _estimate_exx_cupy_bytes(walker_chunk, nk, nocc, nbsf, 1, itemsize)
        > max_mem_bytes
    ):
        walker_chunk = max(1, walker_chunk // 2)

    isdf_chunk = nisdf
    while (
        isdf_chunk > 1
        and _estimate_exx_cupy_bytes(walker_chunk, nk, nocc, nbsf, isdf_chunk, itemsize)
        > max_mem_bytes
    ):
        estimate = _estimate_exx_cupy_bytes(
            walker_chunk, nk, nocc, nbsf, isdf_chunk, itemsize
        )
        scale = (max_mem_bytes / estimate) ** 0.5
        next_chunk = max(1, int(0.85 * isdf_chunk * scale))
        if next_chunk >= isdf_chunk:
            next_chunk = max(1, isdf_chunk // 2)
        isdf_chunk = next_chunk
    return walker_chunk, max(1, isdf_chunk)


def _estimate_exx_dense_cupy_bytes(nwalker, nk, nocc, nbsf, nisdf, pair_chunk, itemsize):
    shifted_g = nwalker * nk**2 * nocc * nbsf * itemsize
    pair_g = 2 * nwalker * pair_chunk * nocc * nbsf * itemsize
    thin_intermediates = 2 * nwalker * pair_chunk * nisdf * nbsf * itemsize
    dense_intermediates = 2 * nwalker * pair_chunk * nisdf**2 * itemsize
    return int(1.35 * (shifted_g + pair_g + thin_intermediates + dense_intermediates))


def _choose_exx_dense_cupy_chunks(nwalker, nk, nocc, nbsf, nisdf, itemsize, max_mem):
    max_mem_bytes = int(max(float(max_mem), 0.01) * 1024**3)
    walker_chunk = nwalker
    while (
        walker_chunk > 1
        and _estimate_exx_dense_cupy_bytes(
            walker_chunk, nk, nocc, nbsf, nisdf, 1, itemsize
        )
        > max_mem_bytes
    ):
        walker_chunk = max(1, walker_chunk // 2)

    max_pairs = nk * nk
    pair_chunk = max_pairs
    while (
        pair_chunk > 1
        and _estimate_exx_dense_cupy_bytes(
            walker_chunk, nk, nocc, nbsf, nisdf, pair_chunk, itemsize
        )
        > max_mem_bytes
    ):
        estimate = _estimate_exx_dense_cupy_bytes(
            walker_chunk, nk, nocc, nbsf, nisdf, pair_chunk, itemsize
        )
        scale = max_mem_bytes / estimate
        next_chunk = max(1, int(0.85 * pair_chunk * scale))
        if next_chunk >= pair_chunk:
            next_chunk = max(1, pair_chunk // 2)
        pair_chunk = next_chunk
    return walker_chunk, max(1, pair_chunk)


def _estimate_exx_dense_optimized_bytes(
    nwalker, nk, nocc, nbsf, nisdf, q_chunk, p_chunk, k_chunk, itemsize
):
    shifted_g = nwalker * k_chunk * nk * nocc * nbsf * itemsize
    rho = nk * nocc * nbsf * p_chunk * itemsize
    a_q = nk * nocc * nbsf * q_chunk * itemsize
    b_q = k_chunk * nk * nwalker * nbsf * q_chunk * itemsize
    c_q = k_chunk * nk * nwalker * nocc * q_chunk * itemsize
    d = nwalker * nk * nocc * k_chunk * nbsf * itemsize
    return int(1.35 * (shifted_g + rho + 2 * a_q + b_q + c_q + d))


def _choose_exx_dense_optimized_chunks(
    nwalker, nk, nocc, nbsf, nisdf, itemsize, max_mem
):
    max_mem_bytes = int(max(float(max_mem), 0.01) * 1024**3)
    q_chunk = nisdf
    p_chunk = nisdf
    k_chunk = nk
    while (
        k_chunk > 1
        and _estimate_exx_dense_optimized_bytes(
            nwalker, nk, nocc, nbsf, nisdf, q_chunk, p_chunk, k_chunk, itemsize
        )
        > max_mem_bytes
    ):
        estimate = _estimate_exx_dense_optimized_bytes(
            nwalker, nk, nocc, nbsf, nisdf, q_chunk, p_chunk, k_chunk, itemsize
        )
        scale = max_mem_bytes / estimate
        next_chunk = max(1, int(0.85 * k_chunk * scale))
        if next_chunk >= k_chunk:
            next_chunk = max(1, k_chunk // 2)
        k_chunk = next_chunk

    while (
        q_chunk > 1
        and _estimate_exx_dense_optimized_bytes(
            nwalker, nk, nocc, nbsf, nisdf, q_chunk, p_chunk, k_chunk, itemsize
        )
        > max_mem_bytes
    ):
        estimate = _estimate_exx_dense_optimized_bytes(
            nwalker, nk, nocc, nbsf, nisdf, q_chunk, p_chunk, k_chunk, itemsize
        )
        scale = max_mem_bytes / estimate
        next_chunk = max(1, int(0.85 * q_chunk * scale))
        if next_chunk >= q_chunk:
            next_chunk = max(1, q_chunk // 2)
        q_chunk = next_chunk

    while (
        p_chunk > 1
        and _estimate_exx_dense_optimized_bytes(
            nwalker, nk, nocc, nbsf, nisdf, q_chunk, p_chunk, k_chunk, itemsize
        )
        > max_mem_bytes
    ):
        estimate = _estimate_exx_dense_optimized_bytes(
            nwalker, nk, nocc, nbsf, nisdf, q_chunk, p_chunk, k_chunk, itemsize
        )
        scale = max_mem_bytes / estimate
        next_chunk = max(1, int(0.85 * p_chunk * scale))
        if next_chunk >= p_chunk:
            next_chunk = max(1, p_chunk // 2)
        p_chunk = next_chunk
    return max(1, q_chunk), max(1, p_chunk), max(1, k_chunk)


def _estimate_exx_cutn_path_cupy_bytes(
    nwalker, nk, nocc, nbsf, nisdf, q_chunk, p_chunk, k_chunk, itemsize
):
    rho = nk * nocc * nbsf * p_chunk * itemsize
    a_q = nk * nocc * nbsf * q_chunk * itemsize
    b_q = k_chunk * q_chunk * nwalker * nk * nocc * itemsize
    c_q = q_chunk * nk * nwalker * k_chunk * nbsf * itemsize
    shifted_g = nwalker * k_chunk * nocc * nk * nbsf * itemsize
    d_q = nwalker * k_chunk * q_chunk * nocc * itemsize
    return int(1.55 * (rho + a_q + b_q + c_q + shifted_g + d_q))


def _choose_exx_cutn_path_cupy_chunks(
    nwalker, nk, nocc, nbsf, nisdf, itemsize, max_mem
):
    max_mem_bytes = int(max(float(max_mem), 0.01) * 1024**3)
    q_chunk = nisdf
    p_chunk = nisdf
    k_chunk = nk
    while (
        k_chunk > 1
        and _estimate_exx_cutn_path_cupy_bytes(
            nwalker, nk, nocc, nbsf, nisdf, q_chunk, p_chunk, k_chunk, itemsize
        )
        > max_mem_bytes
    ):
        estimate = _estimate_exx_cutn_path_cupy_bytes(
            nwalker, nk, nocc, nbsf, nisdf, q_chunk, p_chunk, k_chunk, itemsize
        )
        scale = max_mem_bytes / estimate
        next_chunk = max(1, int(0.85 * k_chunk * scale))
        if next_chunk >= k_chunk:
            next_chunk = max(1, k_chunk // 2)
        k_chunk = next_chunk

    while (
        q_chunk > 1
        and _estimate_exx_cutn_path_cupy_bytes(
            nwalker, nk, nocc, nbsf, nisdf, q_chunk, p_chunk, k_chunk, itemsize
        )
        > max_mem_bytes
    ):
        estimate = _estimate_exx_cutn_path_cupy_bytes(
            nwalker, nk, nocc, nbsf, nisdf, q_chunk, p_chunk, k_chunk, itemsize
        )
        scale = max_mem_bytes / estimate
        next_chunk = max(1, int(0.85 * q_chunk * scale**0.5))
        if next_chunk >= q_chunk:
            next_chunk = max(1, q_chunk // 2)
        q_chunk = next_chunk

    while (
        p_chunk > 1
        and _estimate_exx_cutn_path_cupy_bytes(
            nwalker, nk, nocc, nbsf, nisdf, q_chunk, p_chunk, k_chunk, itemsize
        )
        > max_mem_bytes
    ):
        estimate = _estimate_exx_cutn_path_cupy_bytes(
            nwalker, nk, nocc, nbsf, nisdf, q_chunk, p_chunk, k_chunk, itemsize
        )
        scale = max_mem_bytes / estimate
        next_chunk = max(1, int(0.85 * p_chunk * scale))
        if next_chunk >= p_chunk:
            next_chunk = max(1, p_chunk // 2)
        p_chunk = next_chunk
    return max(1, q_chunk), max(1, p_chunk), max(1, k_chunk)


def kpt_isdf_exx_kernel_gpu(MPQ, halfrot_cgtoa, cgto, Ghalfa_batch, kpq_mat, Sset, Qplus):
    nwalker, nk, nocc, _, nbsf = Ghalfa_batch.shape
    _, kernel = _select_kpt_isdf_exx_kernel_gpu(nk, nbsf, nocc, nwalker=nwalker)
    return kernel(MPQ, halfrot_cgtoa, cgto, Ghalfa_batch, kpq_mat, Sset, Qplus)


def kpt_isdf_exx_kernel_gpu_lowk_cupy(
    MPQ, halfrot_cgtoa, cgto, Ghalfa_batch, kpq_mat, Sset, Qplus, max_mem=None
):
    nwalker, nk, nocc, _, nbsf = Ghalfa_batch.shape
    nisdf = MPQ.shape[-1]
    dtype = xp.result_type(MPQ, halfrot_cgtoa, cgto, Ghalfa_batch)
    itemsize = xp.dtype(dtype).itemsize
    if max_mem is None:
        max_mem = _kptisdf_local_energy_max_mem_gb(default_fraction=0.25)
    walker_chunk, nisdf_per_chunk = _choose_exx_cupy_chunks(
        nwalker, nk, nocc, nbsf, nisdf, itemsize, max_mem
    )
    slices_isdf = [
        slice(start, min(start + nisdf_per_chunk, nisdf))
        for start in range(0, nisdf, nisdf_per_chunk)
    ]
    exx = xp.zeros((nwalker,), dtype=dtype)

    k_idx = xp.arange(nk)[None, :, None, None, None]
    i_idx = xp.arange(nocc)[None, None, :, None, None]
    kprime_idx = xp.arange(nk)[None, None, None, :, None]
    p_idx = xp.arange(nbsf)[None, None, None, None, :]

    for wstart in range(0, nwalker, walker_chunk):
        wstop = min(wstart + walker_chunk, nwalker)
        Ga_chunk = xp.ascontiguousarray(Ghalfa_batch[wstart:wstop])
        n_chunk = wstop - wstart
        w_chunk_idx = xp.arange(n_chunk)[:, None, None, None, None]
        max_dim = max(nisdf_per_chunk, nocc)
        buff1 = xp.empty(n_chunk * nk**2 * max_dim**2, dtype=dtype)
        buff2 = xp.empty(n_chunk * nk**2 * max_dim**2, dtype=dtype)

        for iq in range(len(Sset)):
            iq_real = Sset[iq]
            ikpq = kpq_mat[iq_real]
            phikr_kpq = cgto[ikpq]
            phiki_kpq = halfrot_cgtoa[ikpq]
            kpq_idx = kpq_mat[k_idx, iq_real]
            kprimepq_idx = kpq_mat[kprime_idx, iq_real]
            G_kpq_kprimepq_chunk = Ga_chunk[
                w_chunk_idx, kpq_idx, i_idx, kprimepq_idx, p_idx
            ]
            MPQ_iq = MPQ[iq]
            exx_iq = X_contract_cupy(
                halfrot_cgtoa,
                phikr_kpq,
                MPQ_iq,
                phiki_kpq,
                cgto,
                Ga_chunk,
                G_kpq_kprimepq_chunk,
                buff1,
                buff2,
                slices_isdf,
            )
            exx[wstart:wstop] -= exx_iq
            del G_kpq_kprimepq_chunk

        for iq in range(len(Sset), len(Sset) + len(Qplus)):
            iq_real = Qplus[iq - len(Sset)]
            ikpq = kpq_mat[iq_real]
            phikr_kpq = cgto[ikpq]
            phiki_kpq = halfrot_cgtoa[ikpq]
            kpq_idx = kpq_mat[k_idx, iq_real]
            kprimepq_idx = kpq_mat[kprime_idx, iq_real]
            G_kpq_kprimepq_chunk = Ga_chunk[
                w_chunk_idx, kpq_idx, i_idx, kprimepq_idx, p_idx
            ]
            MPQ_iq = MPQ[iq]
            exx_iq = X_contract_cupy(
                halfrot_cgtoa,
                phikr_kpq,
                MPQ_iq,
                phiki_kpq,
                cgto,
                Ga_chunk,
                G_kpq_kprimepq_chunk,
                buff1,
                buff2,
                slices_isdf,
            )
            exx[wstart:wstop] -= 2.0 * exx_iq
            del G_kpq_kprimepq_chunk
        del Ga_chunk, buff1, buff2
    return 0.5 * exx / nk


def kpt_isdf_exx_kernel_gpu_cupy(
    MPQ, halfrot_cgtoa, cgto, Ghalfa_batch, kpq_mat, Sset, Qplus, max_mem=None
):
    return kpt_isdf_exx_kernel_gpu_lowk_cupy(
        MPQ, halfrot_cgtoa, cgto, Ghalfa_batch, kpq_mat, Sset, Qplus, max_mem=max_mem
    )


def X_contract_dense_cupy(
    halfrot_cgtoa_conj,
    phikr_kpq,
    M_PQ_iq,
    phiki_kpq_conj,
    cgto,
    Ga_chunk,
    iq_real,
    kpq_mat,
    max_mem=None,
):
    nw, nk, nocc, _, nbsf = Ga_chunk.shape
    nisdf = M_PQ_iq.shape[-1]
    dtype = xp.result_type(
        halfrot_cgtoa_conj, phikr_kpq, M_PQ_iq, phiki_kpq_conj, cgto, Ga_chunk
    )
    itemsize = xp.dtype(dtype).itemsize
    max_mem = _kptisdf_effective_max_mem_gb(max_mem, default_fraction=0.25)
    q_chunk, p_chunk, k_chunk = _choose_exx_dense_optimized_chunks(
        nw, nk, nocc, nbsf, nisdf, itemsize, max_mem
    )
    slices_q = [slice(start, min(start + q_chunk, nisdf)) for start in range(0, nisdf, q_chunk)]
    slices_p = [slice(start, min(start + p_chunk, nisdf)) for start in range(0, nisdf, p_chunk)]
    slices_k = [slice(start, min(start + k_chunk, nk)) for start in range(0, nk, k_chunk)]

    exx = xp.zeros((nw,), dtype=dtype)
    w_idx = xp.arange(nw)[:, None, None, None, None]
    j_idx = xp.arange(nocc)[None, None, :, None, None]
    k_idx = xp.arange(nk)[None, None, None, :, None]
    p_idx = xp.arange(nbsf)[None, None, None, None, :]
    kpq_k_idx = kpq_mat[k_idx, iq_real]

    for Q in slices_q:
        nQ = Q.stop - Q.start
        A_kipQ = xp.zeros((nk * nocc * nbsf, nQ), dtype=dtype)
        for P in slices_p:
            nP = P.stop - P.start
            rho_kpq_ipP = halfrot_cgtoa_conj[:, P, :, None] * phikr_kpq[:, P, None, :]
            rho_kpq_ipP = rho_kpq_ipP.transpose(0, 2, 3, 1).reshape(
                nk * nocc * nbsf, nP
            )
            A_kipQ += rho_kpq_ipP @ M_PQ_iq[P, Q]
            del rho_kpq_ipP

        A_kipQ = (
            A_kipQ.reshape(nk, nocc, nbsf, nQ)
            .transpose(0, 3, 2, 1)
            .reshape(nk * nQ, nbsf, nocc)
        )
        for K in slices_k:
            nK = K.stop - K.start
            K_idx = xp.arange(K.start, K.stop)[None, :, None, None, None]
            kpq_K_idx = kpq_mat[K_idx, iq_real]
            G_shifted = Ga_chunk[w_idx, kpq_K_idx, j_idx, kpq_k_idx, p_idx]
            G_shifted = G_shifted.transpose(1, 2, 0, 3, 4).reshape(
                nK, nocc, nw * nk * nbsf
            )

            psi_KQj = phiki_kpq_conj[K, Q, :]
            B_kpQwkp = psi_KQj @ G_shifted
            B_kpQwkp = (
                B_kpQwkp.reshape(nK, nQ, nw, nk, nbsf)
                .transpose(3, 1, 2, 0, 4)
                .reshape(nk * nQ, nw * nK, nbsf)
            )
            C_kQwkpi = B_kpQwkp @ A_kipQ
            C_kQwkpi = (
                C_kQwkpi.reshape(nk, nQ, nw, nK, nocc)
                .transpose(3, 0, 2, 4, 1)
                .reshape(nK, nk * nw * nocc, nQ)
            )
            D_kpkwiq = C_kQwkpi @ cgto[K, Q, :]
            D_kpkwiq = D_kpkwiq.reshape(nK, nk, nw, nocc, nbsf).transpose(
                2, 1, 3, 0, 4
            )
            D_kpkwiq *= Ga_chunk[:, :, :, K, :]
            exx += D_kpkwiq.sum(axis=(1, 2, 3, 4))
            del G_shifted, B_kpQwkp, C_kQwkpi, D_kpkwiq
        del A_kipQ
    return exx


def _accumulate_dense_exchange_cupy(
    exx,
    factor,
    wstart,
    Ga_chunk,
    G_shifted,
    halfrot_cgtoa_conj,
    phikr_kpq,
    phiki_kpq_conj,
    cgto,
    MPQ_iq,
    pair_k,
    pair_K,
    pair_chunk,
):
    n_pairs = pair_k.size
    for pstart in range(0, n_pairs, pair_chunk):
        pstop = min(pstart + pair_chunk, n_pairs)
        k_pair = pair_k[pstart:pstop]
        K_pair = pair_K[pstart:pstop]

        G1_pair = Ga_chunk[:, k_pair, :, K_pair, :].transpose(1, 0, 2, 3)
        G2_pair = G_shifted[:, K_pair, :, k_pair, :].transpose(1, 0, 2, 3)

        T1_left = xp.matmul(
            halfrot_cgtoa_conj[k_pair][None, :, :, :],
            G1_pair,
        )
        T1 = xp.matmul(T1_left, cgto[K_pair].transpose(0, 2, 1)[None, :, :, :])
        del T1_left

        T2_left = xp.matmul(
            phikr_kpq[k_pair][None, :, :, :],
            G2_pair.transpose(0, 1, 3, 2),
        )
        T2 = xp.matmul(T2_left, phiki_kpq_conj[K_pair].transpose(0, 2, 1)[None, :, :, :])
        del T2_left, G1_pair, G2_pair

        T1 *= T2
        T1 *= MPQ_iq[None, None, :, :]
        exx[wstart : wstart + Ga_chunk.shape[0]] -= factor * T1.sum(axis=(1, 2, 3))
        del T1, T2


def X_contract_cutn_path_cupy(
    halfrot_cgtoa_conj,
    phikr_kpq,
    M_PQ_iq,
    phiki_kpq_conj,
    cgto,
    Ga_chunk,
    iq_real,
    kpq_mat,
    max_mem=None,
):
    nw, nk, nocc, _, nbsf = Ga_chunk.shape
    nisdf = M_PQ_iq.shape[-1]
    dtype = xp.result_type(
        halfrot_cgtoa_conj, phikr_kpq, M_PQ_iq, phiki_kpq_conj, cgto, Ga_chunk
    )
    itemsize = xp.dtype(dtype).itemsize
    max_mem = _kptisdf_effective_max_mem_gb(max_mem, default_fraction=0.25)
    q_chunk, p_chunk, k_chunk = _choose_exx_cutn_path_cupy_chunks(
        nw, nk, nocc, nbsf, nisdf, itemsize, max_mem
    )
    slices_q = [slice(start, min(start + q_chunk, nisdf)) for start in range(0, nisdf, q_chunk)]
    slices_p = [slice(start, min(start + p_chunk, nisdf)) for start in range(0, nisdf, p_chunk)]
    slices_k = [slice(start, min(start + k_chunk, nk)) for start in range(0, nk, k_chunk)]

    exx = xp.zeros((nw,), dtype=dtype)
    w_idx = xp.arange(nw)[:, None, None, None, None]
    j_idx = xp.arange(nocc)[None, None, :, None, None]
    p_idx = xp.arange(nbsf)[None, None, None, None, :]
    max_mem_bytes = int(max(float(max_mem), 0.01) * 1024**3)

    for Q in slices_q:
        nQ = Q.stop - Q.start
        A_flat = xp.zeros((nk * nocc * nbsf, nQ), dtype=dtype)
        for P in slices_p:
            nP = P.stop - P.start
            rho_kipP = halfrot_cgtoa_conj[:, P, :, None] * phikr_kpq[:, P, None, :]
            rho_kipP = rho_kipP.transpose(0, 2, 3, 1).reshape(
                nk * nocc * nbsf, nP
            )
            A_flat += rho_kipP @ M_PQ_iq[P, Q]
            del rho_kipP
        A_qkip = A_flat.reshape(nk, nocc, nbsf, nQ).transpose(3, 0, 1, 2)

        for K in slices_k:
            nK = K.stop - K.start
            G1 = Ga_chunk[:, :, :, K, :].transpose(3, 4, 0, 1, 2).reshape(
                nK, nbsf, nw * nk * nocc
            )
            B = cgto[K, Q, :] @ G1
            B = (
                B.reshape(nK, nQ, nw, nk, nocc)
                .transpose(1, 3, 2, 0, 4)
            )
            del G1

            K_idx = xp.arange(K.start, K.stop)[None, :, None, None, None]
            kpq_K_idx = kpq_mat[K_idx, iq_real]

            base_bytes = (
                2 * nk * nocc * nbsf * nQ * itemsize
                + nK * nQ * nw * nk * nocc * itemsize
                + nw * nK * nQ * nocc * itemsize
            )
            reduce_k_chunk = nk
            while reduce_k_chunk > 1:
                c_bytes = nQ * reduce_k_chunk * nw * nK * nbsf * itemsize
                a_batch_bytes = nQ * reduce_k_chunk * nocc * nbsf * itemsize
                b_batch_bytes = nQ * reduce_k_chunk * nw * nK * nocc * itemsize
                g2_bytes = nw * nK * nocc * reduce_k_chunk * nbsf * itemsize
                estimate = int(
                    1.35
                    * (base_bytes + a_batch_bytes + b_batch_bytes + 2 * c_bytes + g2_bytes)
                )
                if estimate <= max_mem_bytes:
                    break
                scale = max_mem_bytes / estimate
                next_chunk = max(1, int(0.85 * reduce_k_chunk * scale))
                if next_chunk >= reduce_k_chunk:
                    next_chunk = max(1, reduce_k_chunk // 2)
                reduce_k_chunk = next_chunk

            for R in range(0, nk, reduce_k_chunk):
                R_slice = slice(R, min(R + reduce_k_chunk, nk))
                nR = R_slice.stop - R_slice.start
                A_batch = A_qkip[:, R_slice, :, :].reshape(nQ * nR, nocc, nbsf)
                B_batch = B[:, R_slice, :, :, :].reshape(nQ * nR, nw * nK, nocc)
                C = B_batch @ A_batch
                del A_batch, B_batch

                C = C.reshape(nQ, nR, nw, nK, nbsf).transpose(2, 3, 0, 1, 4)
                C = C.reshape(nw * nK, nQ, nR * nbsf)

                R_idx = xp.arange(R_slice.start, R_slice.stop)[None, None, None, :, None]
                kpq_R_idx = kpq_mat[R_idx, iq_real]
                G2 = Ga_chunk[w_idx, kpq_K_idx, j_idx, kpq_R_idx, p_idx]
                G2 = G2.reshape(nw * nK, nocc, nR * nbsf)

                D = C @ G2.transpose(0, 2, 1)
                D = D.reshape(nw, nK, nQ, nocc)
                exx += xp.sum(
                    D * phiki_kpq_conj[K, Q, :][None, :, :, :], axis=(1, 2, 3)
                )
                del C, G2, D
            del B
        del A_flat, A_qkip
    return exx


def kpt_isdf_exx_kernel_gpu_cutn_path_cupy(
    MPQ, halfrot_cgtoa, cgto, Ghalfa_batch, kpq_mat, Sset, Qplus, max_mem=None
):
    nwalker, nk, nocc, _, nbsf = Ghalfa_batch.shape
    nisdf = MPQ.shape[-1]
    dtype = xp.result_type(MPQ, halfrot_cgtoa, cgto, Ghalfa_batch)
    itemsize = xp.dtype(dtype).itemsize
    max_mem = _kptisdf_effective_max_mem_gb(max_mem, default_fraction=0.25)
    max_mem_bytes = int(max(float(max_mem), 0.01) * 1024**3)
    walker_chunk = nwalker
    while (
        walker_chunk > 1
        and _estimate_exx_cutn_path_cupy_bytes(
            walker_chunk, nk, nocc, nbsf, nisdf, 1, 1, 1, itemsize
        )
        > max_mem_bytes
    ):
        walker_chunk = max(1, walker_chunk // 2)

    exx = xp.zeros(nwalker, dtype=dtype)
    halfrot_cgtoa_conj = halfrot_cgtoa.conj()

    for wstart in range(0, nwalker, walker_chunk):
        wstop = min(wstart + walker_chunk, nwalker)
        Ga_chunk = xp.ascontiguousarray(Ghalfa_batch[wstart:wstop])

        for iq in range(len(Sset)):
            iq_real = Sset[iq]
            ikpq = kpq_mat[iq_real]
            exx[wstart:wstop] -= X_contract_cutn_path_cupy(
                halfrot_cgtoa_conj,
                cgto[ikpq],
                MPQ[iq],
                halfrot_cgtoa_conj[ikpq],
                cgto,
                Ga_chunk,
                iq_real,
                kpq_mat,
                max_mem=max_mem,
            )

        for iq in range(len(Sset), len(Sset) + len(Qplus)):
            iq_real = Qplus[iq - len(Sset)]
            ikpq = kpq_mat[iq_real]
            exx[wstart:wstop] -= 2.0 * X_contract_cutn_path_cupy(
                halfrot_cgtoa_conj,
                cgto[ikpq],
                MPQ[iq],
                halfrot_cgtoa_conj[ikpq],
                cgto,
                Ga_chunk,
                iq_real,
                kpq_mat,
                max_mem=max_mem,
            )
        del Ga_chunk
    return 0.5 * exx / nk


def kpt_isdf_exx_kernel_gpu_dense_cupy(
    MPQ, halfrot_cgtoa, cgto, Ghalfa_batch, kpq_mat, Sset, Qplus, max_mem=None
):
    nwalker, nk, nocc, _, nbsf = Ghalfa_batch.shape
    nisdf = MPQ.shape[-1]
    dtype = xp.result_type(MPQ, halfrot_cgtoa, cgto, Ghalfa_batch)
    itemsize = xp.dtype(dtype).itemsize
    max_mem = _kptisdf_effective_max_mem_gb(max_mem, default_fraction=0.25)
    max_mem_bytes = int(max(float(max_mem), 0.01) * 1024**3)
    walker_chunk = nwalker
    while (
        walker_chunk > 1
        and _estimate_exx_dense_optimized_bytes(
            walker_chunk, nk, nocc, nbsf, nisdf, 1, 1, 1, itemsize
        )
        > max_mem_bytes
    ):
        walker_chunk = max(1, walker_chunk // 2)

    exx = xp.zeros(nwalker, dtype=dtype)
    halfrot_cgtoa_conj = halfrot_cgtoa.conj()

    for wstart in range(0, nwalker, walker_chunk):
        wstop = min(wstart + walker_chunk, nwalker)
        Ga_chunk = xp.ascontiguousarray(Ghalfa_batch[wstart:wstop])

        for iq in range(len(Sset)):
            iq_real = Sset[iq]
            ikpq = kpq_mat[iq_real]
            exx[wstart:wstop] -= X_contract_dense_cupy(
                halfrot_cgtoa_conj,
                cgto[ikpq],
                MPQ[iq],
                halfrot_cgtoa_conj[ikpq],
                cgto,
                Ga_chunk,
                iq_real,
                kpq_mat,
                max_mem=max_mem,
            )

        for iq in range(len(Sset), len(Sset) + len(Qplus)):
            iq_real = Qplus[iq - len(Sset)]
            ikpq = kpq_mat[iq_real]
            exx[wstart:wstop] -= 2.0 * X_contract_dense_cupy(
                halfrot_cgtoa_conj,
                cgto[ikpq],
                MPQ[iq],
                halfrot_cgtoa_conj[ikpq],
                cgto,
                Ga_chunk,
                iq_real,
                kpq_mat,
                max_mem=max_mem,
            )
        del Ga_chunk
    return 0.5 * exx / nk


def kpt_isdf_exx_kernel_gpu_original_cuquantum(
    MPQ, halfrot_cgtoa, cgto, Ghalfa_batch, kpq_mat, Sset, Qplus
):
    nwalker, nk, nocc, _, nbsf = Ghalfa_batch.shape
    nisdf = MPQ.shape[-1]
    w_idx = xp.arange(nwalker)[:, None, None, None, None]
    k_idx = xp.arange(nk)[None, :, None, None, None]
    i_idx = xp.arange(nocc)[None, None, :, None, None]
    kprime_idx = xp.arange(nk)[None, None, None, :, None]
    p_idx = xp.arange(nbsf)[None, None, None, None, :]
    handle = cutensornet.create()

    exx = xp.zeros(nwalker, dtype=numpy.complex128)

    if nk < 64:
        intermediate_mem = nwalker * nisdf * nk * nk * nbsf * 6 * 16 / 1024**3
    else:
        intermediate_mem = nwalker * nisdf * nk * nbsf * 6 * 16 / 1024**3
    free_bytes = xp.cuda.Device().mem_info[0]
    free_gb = free_bytes / 1024**3.0
    max_mem = 0.7 * free_gb
    num_chunks = max(1, ceil(intermediate_mem / max_mem))
    chunk_size = ceil(nwalker / num_chunks)
    nw_left = nwalker
    for i_chunk in range(num_chunks):
        if nw_left == 0:
            break
        n_chunk = min(nw_left, chunk_size)
        nw_left -= n_chunk
        w_sls = xp.arange(nwalker)[i_chunk * chunk_size : i_chunk * chunk_size + n_chunk]
        Ga_chunk = Ghalfa_batch[w_sls]
        w_chunk_idx = xp.arange(n_chunk)[:, None, None, None, None]

        for iq in range(len(Sset)):
            iq_real = Sset[iq]
            ikpq = kpq_mat[iq_real]
            phikr_kpq = cgto[ikpq]
            phiki_kpq = halfrot_cgtoa[ikpq]
            kpq_idx = kpq_mat[k_idx, iq_real]
            kprimepq_idx = kpq_mat[kprime_idx, iq_real]
            G_kpq_kprimepq_chunk = Ga_chunk[
                w_chunk_idx, kpq_idx, i_idx, kprimepq_idx, p_idx
            ]
            MPQ_iq = MPQ[iq]
            network_opts = NetworkOptions(
                handle=handle, memory_limit=0.7 * xp.cuda.Device().mem_info[0]
            )
            exx[w_sls] -= contract(
                "kPi, kPp, PQ, KQj, KQq, wkiKq, wKjkp -> w",
                halfrot_cgtoa.conj(),
                phikr_kpq,
                MPQ_iq,
                phiki_kpq.conj(),
                cgto,
                Ga_chunk,
                G_kpq_kprimepq_chunk,
                options=network_opts,
            )
            xp.cuda.get_current_stream().synchronize()
            del G_kpq_kprimepq_chunk

        for iq in range(len(Sset), len(Sset) + len(Qplus)):
            iq_real = Qplus[iq - len(Sset)]
            ikpq = kpq_mat[iq_real]
            phikr_kpq = cgto[ikpq]
            phiki_kpq = halfrot_cgtoa[ikpq]
            kpq_idx = kpq_mat[k_idx, iq_real]
            kprimepq_idx = kpq_mat[kprime_idx, iq_real]
            G_kpq_kprimepq_chunk = Ga_chunk[
                w_chunk_idx, kpq_idx, i_idx, kprimepq_idx, p_idx
            ]
            MPQ_iq = MPQ[iq]
            network_opts = NetworkOptions(
                handle=handle, memory_limit=0.7 * xp.cuda.Device().mem_info[0]
            )
            exx[w_sls] -= 2.0 * contract(
                "kPi, kPp, PQ, KQj, KQq, wkiKq, wKjkp -> w",
                halfrot_cgtoa.conj(),
                phikr_kpq,
                MPQ_iq,
                phiki_kpq.conj(),
                cgto,
                Ga_chunk,
                G_kpq_kprimepq_chunk,
                options=network_opts,
            )
            xp.cuda.get_current_stream().synchronize()
            del G_kpq_kprimepq_chunk

    cutensornet.destroy(handle)
    return 0.5 * exx / nk


def kpt_isdf_exx_kernel_gpu_old_cuquantum(
    MPQ, halfrot_cgtoa, cgto, Ghalfa_batch, kpq_mat, Sset, Qplus
):
    return kpt_isdf_exx_kernel_gpu_original_cuquantum(
        MPQ, halfrot_cgtoa, cgto, Ghalfa_batch, kpq_mat, Sset, Qplus
    )


def kpt_isdf_ecoul_kernel_rhf():
    raise NotImplementedError("CPU ISDF Coulomb kernel for RHF not implemented yet.")


@jit(nopython=True, fastmath=True)
def kpt_isdf_ecoul_kernel_uhf():
    raise NotImplementedError("CPU ISDF Coulomb kernel for UHF not implemented yet.")


def local_energy_kpt_single_det_uhf_isdf_gpu(system, hamiltonian, walkers, trial):
    if not config.get_option("use_gpu"):
        raise NotImplementedError("CPU ISDF Coulomb kernel for UHF not implemented yet.")

    nwalkers = walkers.Ghalfa.shape[0]
    nk = hamiltonian.nk
    nalpha = trial.nalpha
    nbeta = trial.nbeta
    nbasis = hamiltonian.nbasis

    if walkers.rhf:
        ghalfa = walkers.Ghalfa.reshape(nwalkers, nk, nalpha, nk, nbasis)
        diagGhalfa = xp.zeros((nwalkers, nk, nalpha, nbasis), dtype=numpy.complex128)
        for ik in range(nk):
            diagGhalfa[:, ik, :, :] = ghalfa[:, ik, :, ik, :]
        diagGhalfa = diagGhalfa.reshape(nwalkers, nk * nalpha * nbasis)
        e1b = 2.0 * diagGhalfa.dot(trial._rH1a.ravel())
        e1b /= nk
        e1b += hamiltonian.ecore

        ecoul = kpt_isdf_ecoul_rhf_kernel_gpu(
            hamiltonian.MPQ,
            trial._rcgtoa,
            hamiltonian.cgto,
            ghalfa,
            hamiltonian.ikpq_mat,
            hamiltonian.Sset,
            hamiltonian.Qplus,
        )

        exxa = 2.0 * kpt_isdf_exx_kernel_gpu(
            hamiltonian.MPQ,
            trial._rcgtoa,
            hamiltonian.cgto,
            ghalfa,
            hamiltonian.ikpq_mat,
            hamiltonian.Sset,
            hamiltonian.Qplus,
        )

        e2b = ecoul + exxa
    else:
        ghalfa = walkers.Ghalfa.reshape(nwalkers, nk, nalpha, nk, nbasis)
        ghalfb = walkers.Ghalfb.reshape(nwalkers, nk, nbeta, nk, nbasis)

        diagGhalfa = xp.zeros((nwalkers, nk, nalpha, nbasis), dtype=numpy.complex128)
        diagGhalfb = xp.zeros((nwalkers, nk, nbeta, nbasis), dtype=numpy.complex128)
        for ik in range(nk):
            diagGhalfa[:, ik, :, :] = ghalfa[:, ik, :, ik, :]
            diagGhalfb[:, ik, :, :] = ghalfb[:, ik, :, ik, :]
        diagGhalfa = diagGhalfa.reshape(nwalkers, nk * nalpha * nbasis)
        diagGhalfb = diagGhalfb.reshape(nwalkers, nk * nbeta * nbasis)
        e1b = diagGhalfa.dot(trial._rH1a.ravel())
        e1b += diagGhalfb.dot(trial._rH1b.ravel())
        e1b /= nk
        e1b += hamiltonian.ecore

        ecoul = kpt_isdf_ecoul_kernel_gpu(
            hamiltonian.MPQ,
            trial._rcgtoa,
            trial._rcgtob,
            hamiltonian.cgto,
            ghalfa,
            ghalfb,
            hamiltonian.ikpq_mat,
            hamiltonian.Sset,
            hamiltonian.Qplus,
        )

        exxa = kpt_isdf_exx_kernel_gpu(
            hamiltonian.MPQ,
            trial._rcgtoa,
            hamiltonian.cgto,
            ghalfa,
            hamiltonian.ikpq_mat,
            hamiltonian.Sset,
            hamiltonian.Qplus,
        )
        exxb = kpt_isdf_exx_kernel_gpu(
            hamiltonian.MPQ,
            trial._rcgtob,
            hamiltonian.cgto,
            ghalfb,
            hamiltonian.ikpq_mat,
            hamiltonian.Sset,
            hamiltonian.Qplus,
        )

        e2b = ecoul + exxa + exxb

    energy = xp.zeros((nwalkers, 3), dtype=numpy.complex128)
    energy[:, 0] = e1b + e2b
    energy[:, 1] = e1b
    energy[:, 2] = e2b

    xp._default_memory_pool.free_all_blocks()
    return energy
