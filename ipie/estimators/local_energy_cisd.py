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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either eprXess or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Jinghong Zhang <jinghongzhang@fas.harvard.edu>
#
from math import ceil
import os

from ipie.utils.backend import arraylib as xp
from ipie.utils.cuquantum_backend import HAS_CUQUANTUM, NetworkOptions_optional, contract_optional
from ipie.hamiltonians.generic import GenericRealChol
from ipie.walkers.uhf_walkers import UHFWalkers
from ipie.trial_wavefunction.cisd import CISD

try:
    from cuquantum import ComputeType
except Exception:
    ComputeType = None


_CUQ_OPTIONS_CACHE = None


def _get_cuquantum_options():
    global _CUQ_OPTIONS_CACHE

    if not HAS_CUQUANTUM:
        return None
    if _CUQ_OPTIONS_CACHE is not None:
        return _CUQ_OPTIONS_CACHE

    if ComputeType is None:
        _CUQ_OPTIONS_CACHE = NetworkOptions_optional(blocking=True)
    else:
        _CUQ_OPTIONS_CACHE = NetworkOptions_optional(compute_type=ComputeType.COMPUTE_64F, blocking=True)
    return _CUQ_OPTIONS_CACHE


def _use_cuquantum_einsum() -> bool:
    return HAS_CUQUANTUM and hasattr(xp, "cuda")


def _einsum_dispatch(subscripts, *operands, optimize=True, use_cuquantum=None, cuq_options=None):
    """Dispatch einsum with mixed real/complex handling for cuQuantum.

    cuQuantum requires uniform dtypes across all operands. If there is exactly one
    complex operand and the rest are real-valued, evaluate real and imaginary parts
    separately using real contractions and combine the result.
    """
    if use_cuquantum is None:
        use_cuquantum = _use_cuquantum_einsum()

    if not use_cuquantum:
        return xp.einsum(subscripts, *operands, optimize=optimize)

    dtypes = [op.dtype for op in operands]
    if all(dt == dtypes[0] for dt in dtypes):
        return contract_optional(subscripts, *operands, options=cuq_options)

    # Mixed dtypes: fall back to backend einsum to avoid extra split contractions.
    # This keeps cuQuantum on uniform-dtype contractions and avoids overhead.
    return xp.einsum(subscripts, *operands, optimize=optimize)


def _local_energy_rcisd_kernel(
    h1e, chol, ghalfa, rh1, c1a, c2aa, rchol, ovlp_ratio, use_cuquantum=None
):
    """Array-only RCISD local-energy kernel for a walker chunk."""
    if use_cuquantum is None:
        use_cuquantum = _use_cuquantum_einsum()

    cuq_options = _get_cuquantum_options() if use_cuquantum else None

    def einsum(subscripts, *operands, optimize=True):
        return _einsum_dispatch(
            subscripts,
            *operands,
            optimize=optimize,
            use_cuquantum=use_cuquantum,
            cuq_options=cuq_options,
        )

    nalpha = c1a.shape[0]
    gova = ghalfa[:, :, nalpha:]
    nvira = h1e.shape[0] - nalpha
    eye_a = -xp.broadcast_to(xp.eye(nvira)[None, :, :], (gova.shape[0], nvira, nvira))
    calg_a = xp.concatenate([gova, eye_a], axis=1)

    h1g = einsum("kq,wkq->w", rh1, ghalfa, optimize=True)
    c1gov = einsum("ia,wia->w", c1a, gova, optimize=True)
    c2g_dir = einsum("iajb,wia->wjb", c2aa, gova, optimize=True)
    c2g_cross = einsum("iajb,wib->wja", c2aa, gova, optimize=True)
    c2g_dir_G = einsum("wia,wiq->wqa", c2g_dir, ghalfa, optimize=True)
    calg_c2g_dir_G = einsum("wpa,wqa->wpq", calg_a, c2g_dir_G, optimize=True)
    c2g_cross_G = einsum("wja,wjq->wqa", c2g_cross, ghalfa, optimize=True)
    calg_c2g_cross_G = einsum("wpa,wqa->wpq", calg_a, c2g_cross_G, optimize=True)
    chol_g = einsum("kqX,wkq->wX", rchol, ghalfa, optimize=True)
    cholg_mat = einsum("krX,wks->wXsr", rchol, ghalfa, optimize=True)
    c1calg = einsum("ia,wpa->wpi", c1a, calg_a, optimize=True)
    c1calgG = einsum("wpi,wiq->wpq", c1calg, ghalfa, optimize=True)
    lc1calgG = einsum("prX,wpr->wX", chol, c1calgG, optimize=True)
    c1g_mat = einsum("ia,wir->wra", c1a, ghalfa, optimize=True)
    c1g_matcalg = einsum("wra,wqa->wrq", c1g_mat, calg_a, optimize=True)
    cholg_c1gcalg = einsum("wXsr,wrq->wXsq", cholg_mat, c1g_matcalg, optimize=True)
    c2g_dir_calg = einsum("wpa,wia->wip", calg_a, c2g_dir, optimize=True)
    c2g_cross_calg = einsum("wpa,wja->wjp", calg_a, c2g_cross, optimize=True)
    c2gcalG = 2.0 * c2g_dir_calg - c2g_cross_calg
    c2gcalGG = einsum("wip,wir->wpr", c2gcalG, ghalfa, optimize=True)
    lc2gcalGG_c = einsum("prX,wpr->wX", chol, c2gcalGG, optimize=True)
    lc2gcalGG_x = einsum("qsX,wqr->wXsr", chol, c2gcalGG, optimize=True)
    cholgT = einsum("prX,wir->wXpi", chol, ghalfa, optimize=True)
    cholgTcalg = einsum("wXpi,wpa->wXai", cholgT, calg_a, optimize=True)

    e1_0 = 2.0 * h1g
    e1_1_1 = 4.0 * einsum("w,w->w", h1g, c1gov, optimize=True)
    e1_1_2 = -2.0 * einsum("pq,ia,wiq,wpa->w", h1e, c1a, ghalfa, calg_a, optimize=True)
    e1_2_1 = 4.0 * einsum("w,wjb,wjb->w", h1g, c2g_dir, gova, optimize=True)
    e1_2_2 = -2.0 * einsum("w,wja,wja->w", h1g, c2g_cross, gova, optimize=True)
    e1_2_3 = -4.0 * einsum("pq,wpq->w", h1e, calg_c2g_dir_G, optimize=True)
    e1_2_4 = 2.0 * einsum("pq,wpq->w", h1e, calg_c2g_cross_G, optimize=True)
    e1 = e1_0 + e1_1_1 + e1_1_2 + e1_2_1 + e1_2_2 + e1_2_3 + e1_2_4

    e2_0_1 = 2.0 * einsum("wX,wX->w", chol_g, chol_g, optimize=True)
    e2_0_2 = -1.0 * einsum("wXsr,wXrs->w", cholg_mat, cholg_mat, optimize=True)
    e2_1_1 = 2.0 * c1gov * (e2_0_1 + e2_0_2)
    e2_1_2 = -4.0 * einsum("wX,wX->w", lc1calgG, chol_g, optimize=True)
    e2_1_3 = 2.0 * einsum("wXsq,qsX->w", cholg_c1gcalg, chol, optimize=True)
    c2g_dir_g = einsum("wjb,wjb->w", c2g_dir, gova, optimize=True)
    c2g_cross_g = einsum("wja,wja->w", c2g_cross, gova, optimize=True)
    e2_2_1 = (e2_0_1 + e2_0_2) * (2.0 * c2g_dir_g - c2g_cross_g)
    e2_2_2 = -4.0 * einsum("wX,wX->w", lc2gcalGG_c, chol_g, optimize=True)
    e2_2_3 = 2.0 * einsum("wXsr,wXsr->w", cholg_mat, lc2gcalGG_x, optimize=True)
    e2_2_4 = 2.0 * einsum("iajb,wXai,wXbj->w", c2aa, cholgTcalg, cholgTcalg, optimize=True)
    e2_2_5 = -1.0 * einsum("iajb,wXaj,wXbi->w", c2aa, cholgTcalg, cholgTcalg, optimize=True)
    e2 = e2_0_1 + e2_0_2 + e2_1_1 + e2_1_2 + e2_1_3 + e2_2_1 + e2_2_2 + e2_2_3 + e2_2_4 + e2_2_5

    return e1 / ovlp_ratio, e2 / ovlp_ratio


def local_energy_rcisd_batch(hamiltonian: GenericRealChol, walkers: UHFWalkers, trial: CISD):
    """Calculate local energy for RCISD trial wavefunction."""
    use_cuquantum = _use_cuquantum_einsum()
    chol = hamiltonian.chol.reshape((hamiltonian.nbasis, hamiltonian.nbasis, hamiltonian.nchol))
    nwalkers = walkers.nwalkers
    energy = xp.zeros((nwalkers, 3), dtype=xp.complex128)

    ovlp_ratio_all = trial.ovlp_ratio_cisd
    ovlp_is_walker_resolved = (
        hasattr(ovlp_ratio_all, "shape")
        and len(ovlp_ratio_all.shape) > 0
        and ovlp_ratio_all.shape[0] == nwalkers
    )

    for w_sls in _get_walker_chunks(nwalkers, hamiltonian.nchol, hamiltonian.nbasis):
        ghalfa_chunk = walkers.ghalfa[w_sls]
        if ovlp_is_walker_resolved:
            ovlp_chunk = ovlp_ratio_all[w_sls]
        else:
            ovlp_chunk = ovlp_ratio_all

        e1b_chunk, e2b_chunk = _local_energy_rcisd_kernel(
            hamiltonian.h1e[0],
            chol,
            ghalfa_chunk,
            trial.rh1,
            trial.c1a,
            trial.c2aa,
            trial.rchol,
            ovlp_chunk,
            use_cuquantum=use_cuquantum,
        )

        energy[w_sls, 0] = e1b_chunk + e2b_chunk + hamiltonian.ecore
        energy[w_sls, 1] = e1b_chunk + hamiltonian.ecore
        energy[w_sls, 2] = e2b_chunk
    return energy

def _get_walker_chunks(nwalkers: int, nchol: int, nbasis: int):
    """Return walker-index slices for chunked local-energy evaluation."""
    # Conservative memory model for dominant tensor intermediates per walker.
    mem_per_walker_gb = nchol * nbasis**2 * 16 * 4 / 1024**3.0

    max_mem_gb = None

    # Prefer CUDA memory info if available.
    if hasattr(xp, "cuda"):
        try:
            free_bytes = xp.cuda.Device().mem_info[0]
            free_gb = free_bytes / 1024**3.0
            max_mem_gb = max(4.0, 0.7 * free_gb)
        except Exception:
            max_mem_gb = None

    # Fallback: estimate from host available memory for numpy/cpu backends.
    if max_mem_gb is None:
        try:
            pages = os.sysconf("SC_AVPHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            free_gb = (pages * page_size) / 1024**3.0
            max_mem_gb = max(4.0, 0.5 * free_gb)
        except Exception:
            # Last-resort safety budget to avoid giant one-shot allocations.
            max_mem_gb = 4.0

    num_chunks = max(1, ceil((nwalkers * mem_per_walker_gb) / max_mem_gb))

    chunk_size = ceil(nwalkers / num_chunks)
    return [slice(i, min(i + chunk_size, nwalkers)) for i in range(0, nwalkers, chunk_size)]


def _local_energy_ucisd_kernel(
    h1ea,
    h1eb,
    chol,
    cholb,
    ghalfa,
    ghalfb,
    rh1a,
    rh1b,
    c1a,
    c1b,
    c2aa,
    c2ab,
    c2bb,
    rchola,
    rcholb,
    ovlp_ratio,
    c2_antisymm,
    use_cuquantum=None,
):
    """Array-only UCISD local-energy kernel for a walker chunk."""
    if use_cuquantum is None:
        use_cuquantum = _use_cuquantum_einsum()

    cuq_options = _get_cuquantum_options() if use_cuquantum else None

    def einsum(subscripts, *operands, optimize=True):
        return _einsum_dispatch(
            subscripts,
            *operands,
            optimize=optimize,
            use_cuquantum=use_cuquantum,
            cuq_options=cuq_options,
        )

    nalpha = c1a.shape[0]
    nbeta = c1b.shape[0]
    nbasis = h1ea.shape[0]
    gova = ghalfa[:, :, nalpha:]
    govb = ghalfb[:, :, nbeta:]
    nvira = nbasis - nalpha
    nvirb = nbasis - nbeta
    eye_a = -xp.broadcast_to(xp.eye(nvira)[None, :, :], (gova.shape[0], nvira, nvira))
    eye_b = -xp.broadcast_to(xp.eye(nvirb)[None, :, :], (govb.shape[0], nvirb, nvirb))
    calg_a = xp.concatenate([gova, eye_a], axis=1)
    calg_b = xp.concatenate([govb, eye_b], axis=1)

    h1g_a = einsum("kq,wkq->w", rh1a, ghalfa, optimize=True)
    h1g_b = einsum("kq,wkq->w", rh1b, ghalfb, optimize=True)
    hg = h1g_a + h1g_b
    c1g_a = einsum("ia,wia->w", c1a, gova, optimize=True)
    c1g_b = einsum("ia,wia->w", c1b, govb, optimize=True)
    c1g = c1g_a + c1g_b
    c1calg_a = einsum("ia,wpa->wpi", c1a, calg_a, optimize=True)
    c1calg_b = einsum("ia,wpa->wpi", c1b, calg_b, optimize=True)
    c1calgG_a = einsum("wpi,wiq->wpq", c1calg_a, ghalfa, optimize=True)
    c1calgG_b = einsum("wpi,wiq->wpq", c1calg_b, ghalfb, optimize=True)
    chol_g_a = einsum("kqX,wkq->wX", rchola, ghalfa, optimize=True)
    chol_g_b = einsum("kqX,wkq->wX", rcholb, ghalfb, optimize=True)
    chol_g = chol_g_a + chol_g_b
    cholg_mat_a = einsum("krX,wks->wXsr", rchola, ghalfa, optimize=True)
    cholg_mat_b = einsum("krX,wks->wXsr", rcholb, ghalfb, optimize=True)
    lc1calgG_a = einsum("pqX,wpq->wX", chol, c1calgG_a, optimize=True)
    lc1calgG_b = einsum("pqX,wpq->wX", cholb, c1calgG_b, optimize=True)
    c1g_mat_a = einsum("ia,wir->wra", c1a, ghalfa, optimize=True)
    c1g_mat_b = einsum("ia,wir->wra", c1b, ghalfb, optimize=True)
    c1g_matcalg_a = einsum("wra,wqa->wrq", c1g_mat_a, calg_a, optimize=True)
    c1g_matcalg_b = einsum("wra,wqa->wrq", c1g_mat_b, calg_b, optimize=True)
    cholg_c1gcalg_a = einsum("wXsr,wrq->wXsq", cholg_mat_a, c1g_matcalg_a, optimize=True)
    cholg_c1gcalg_b = einsum("wXsr,wrq->wXsq", cholg_mat_b, c1g_matcalg_b, optimize=True)
    gl_a = einsum("wpr,rqX->wpqX", ghalfa, chol, optimize=True)
    gl_b = einsum("wpr,rqX->wpqX", ghalfb, cholb, optimize=True)
    cholgT_a = einsum("prX,wir->wXpi", chol, ghalfa, optimize=True)
    cholgT_b = einsum("prX,wir->wXpi", cholb, ghalfb, optimize=True)
    cholgTcalg_a = einsum("wXpi,wpa->wXai", cholgT_a, calg_a, optimize=True)
    cholgTcalg_b = einsum("wXpi,wpa->wXai", cholgT_b, calg_b, optimize=True)

    if c2_antisymm:
        c2g_aa = einsum("iajb,wia->wjb", c2aa, gova, optimize=True)
        c2g_bb = einsum("iajb,wia->wjb", c2bb, govb, optimize=True)
        c2g_ab_a = einsum("iajb,wjb->wia", c2ab, govb, optimize=True)
        c2g_ab_b = einsum("iajb,wia->wjb", c2ab, gova, optimize=True)
        gci2g_aa_c = 0.5 * einsum("wia,wia->w", c2g_aa, gova, optimize=True)
        gci2g_bb_c = 0.5 * einsum("wia,wia->w", c2g_bb, govb, optimize=True)
        gci2g_aa_x = xp.zeros_like(gci2g_aa_c)
        gci2g_bb_x = xp.zeros_like(gci2g_bb_c)
        gci2g_ab = einsum("wia,wia->w", c2g_ab_a, gova, optimize=True)
        gci2g = gci2g_aa_c + gci2g_bb_c + gci2g_ab
        c2g_aa_G = einsum("wia,wiq->wqa", c2g_aa, ghalfa, optimize=True)
        c2g_bb_G = einsum("wia,wiq->wqa", c2g_bb, ghalfb, optimize=True)
        c2g_ab_a_G = einsum("wia,wiq->wqa", c2g_ab_a, ghalfa, optimize=True)
        c2g_ab_b_G = einsum("wia,wiq->wqa", c2g_ab_b, ghalfb, optimize=True)
        calg_c2g_aa_G = einsum("wpa,wqa->wpq", calg_a, c2g_aa_G, optimize=True)
        calg_c2g_bb_G = einsum("wpa,wqa->wpq", calg_b, c2g_bb_G, optimize=True)
        calg_c2g_ab_a_G = einsum("wpa,wqa->wpq", calg_a, c2g_ab_a_G, optimize=True)
        calg_c2g_ab_b_G = einsum("wpa,wqa->wpq", calg_b, c2g_ab_b_G, optimize=True)
        e2_2_4_aa = 0.5 * einsum("iajb,wXai,wXbj->w", c2aa, cholgTcalg_a, cholgTcalg_a, optimize=True)
        e2_2_4_bb = 0.5 * einsum("iajb,wXai,wXbj->w", c2bb, cholgTcalg_b, cholgTcalg_b, optimize=True)
        e2_2_5_aa = xp.zeros_like(e2_2_4_aa)
        e2_2_5_bb = xp.zeros_like(e2_2_4_bb)
        e1_2_1 = hg * gci2g
        e1_2_2 = xp.zeros_like(e1_2_1)
        e1_2_3_aa_c = -einsum("pq,wpq->w", h1ea, calg_c2g_aa_G, optimize=True)
        e1_2_3_bb_c = -einsum("pq,wpq->w", h1eb, calg_c2g_bb_G, optimize=True)
        e1_2_3_ab_a = -einsum("pq,wpq->w", h1ea, calg_c2g_ab_a_G, optimize=True)
        e1_2_3_ab_b = -einsum("pq,wpq->w", h1eb, calg_c2g_ab_b_G, optimize=True)
        e1_2_4_aa_x = xp.zeros_like(e1_2_1)
        e1_2_4_bb_x = xp.zeros_like(e1_2_1)
    else:
        c2g_aa_c = einsum("iajb,wia->wjb", c2aa, gova, optimize=True)
        c2g_bb_c = einsum("iajb,wia->wjb", c2bb, govb, optimize=True)
        c2g_aa_x = einsum("iajb,wib->wja", c2aa, gova, optimize=True)
        c2g_bb_x = einsum("iajb,wib->wja", c2bb, govb, optimize=True)
        c2g_ab_a = einsum("iajb,wjb->wia", c2ab, govb, optimize=True)
        c2g_ab_b = einsum("iajb,wia->wjb", c2ab, gova, optimize=True)
        gci2g_aa_c = 0.5 * einsum("wia,wia->w", c2g_aa_c, gova, optimize=True)
        gci2g_aa_x = -0.5 * einsum("wia,wia->w", c2g_aa_x, gova, optimize=True)
        gci2g_bb_c = 0.5 * einsum("wia,wia->w", c2g_bb_c, govb, optimize=True)
        gci2g_bb_x = -0.5 * einsum("wia,wia->w", c2g_bb_x, govb, optimize=True)
        gci2g_ab = einsum("wia,wia->w", c2g_ab_a, gova, optimize=True)
        gci2g = gci2g_aa_c + gci2g_aa_x + gci2g_bb_c + gci2g_bb_x + gci2g_ab
        c2g_aa_G_c = 0.5 * einsum("wia,wiq->wqa", c2g_aa_c, ghalfa, optimize=True)
        c2g_aa_G_x = -0.5 * einsum("wja,wjq->wqa", c2g_aa_x, ghalfa, optimize=True)
        c2g_bb_G_c = 0.5 * einsum("wia,wiq->wqa", c2g_bb_c, ghalfb, optimize=True)
        c2g_bb_G_x = -0.5 * einsum("wja,wjq->wqa", c2g_bb_x, ghalfb, optimize=True)
        c2g_ab_a_G = einsum("wia,wiq->wqa", c2g_ab_a, ghalfa, optimize=True)
        c2g_ab_b_G = einsum("wia,wiq->wqa", c2g_ab_b, ghalfb, optimize=True)
        calg_c2g_aa_G_c = einsum("wpa,wqa->wpq", calg_a, c2g_aa_G_c, optimize=True)
        calg_c2g_aa_G_x = einsum("wpa,wqa->wpq", calg_a, c2g_aa_G_x, optimize=True)
        calg_c2g_bb_G_c = einsum("wpa,wqa->wpq", calg_b, c2g_bb_G_c, optimize=True)
        calg_c2g_bb_G_x = einsum("wpa,wqa->wpq", calg_b, c2g_bb_G_x, optimize=True)
        calg_c2g_ab_a_G = einsum("wpa,wqa->wpq", calg_a, c2g_ab_a_G, optimize=True)
        calg_c2g_ab_b_G = einsum("wpa,wqa->wpq", calg_b, c2g_ab_b_G, optimize=True)
        e2_2_4_aa = 0.5 * einsum("iajb,wXai,wXbj->w", c2aa, cholgTcalg_a, cholgTcalg_a, optimize=True)
        e2_2_5_aa = -0.5 * einsum("iajb,wXaj,wXbi->w", c2aa, cholgTcalg_a, cholgTcalg_a, optimize=True)
        e2_2_4_bb = 0.5 * einsum("iajb,wXai,wXbj->w", c2bb, cholgTcalg_b, cholgTcalg_b, optimize=True)
        e2_2_5_bb = -0.5 * einsum("iajb,wXaj,wXbi->w", c2bb, cholgTcalg_b, cholgTcalg_b, optimize=True)
        e1_2_1 = hg * (gci2g_aa_c + gci2g_bb_c + gci2g_ab)
        e1_2_2 = hg * (gci2g_aa_x + gci2g_bb_x)
        e1_2_3_aa_c = -einsum("pq,wpa,wia,wiq->w", h1ea, calg_a, c2g_aa_c, ghalfa, optimize=True)
        e1_2_3_bb_c = -einsum("pq,wpa,wia,wiq->w", h1eb, calg_b, c2g_bb_c, ghalfb, optimize=True)
        e1_2_3_ab_a = -einsum("pq,wpq->w", h1ea, calg_c2g_ab_a_G, optimize=True)
        e1_2_3_ab_b = -einsum("pq,wpq->w", h1eb, calg_c2g_ab_b_G, optimize=True)
        e1_2_4_aa_x = einsum("pq,wpa,wja,wjq->w", h1ea, calg_a, c2g_aa_x, ghalfa, optimize=True)
        e1_2_4_bb_x = einsum("pq,wpa,wja,wjq->w", h1eb, calg_b, c2g_bb_x, ghalfb, optimize=True)

    e1_0 = hg
    e1_1_1 = c1g * hg
    e1_1_2 = -einsum("pq,wpq->w", h1ea, c1calgG_a, optimize=True)
    e1_1_2 -= einsum("pq,wpq->w", h1eb, c1calgG_b, optimize=True)
    e1_2_3 = e1_2_3_aa_c + e1_2_3_bb_c + e1_2_3_ab_a + e1_2_3_ab_b
    e1_2_4 = e1_2_4_aa_x + e1_2_4_bb_x
    e2_2_4_ab = einsum("iajb,wXai,wXbj->w", c2ab, cholgTcalg_a, cholgTcalg_b, optimize=True)
    e2_0_1 = 0.5 * einsum("wX,wX->w", chol_g, chol_g, optimize=True)
    e2_0_2 = -0.5 * (
        einsum("wXsr,wXrs->w", cholg_mat_a, cholg_mat_a, optimize=True)
        + einsum("wXsr,wXrs->w", cholg_mat_b, cholg_mat_b, optimize=True)
    )
    e2_1_1 = (e2_0_1 + e2_0_2) * c1g
    e2_1_2 = -einsum("wX,wX->w", lc1calgG_a + lc1calgG_b, chol_g, optimize=True)
    e2_1_3 = einsum("wXsq,qsX->w", cholg_c1gcalg_a, chol, optimize=True)
    e2_1_3 += einsum("wXsq,qsX->w", cholg_c1gcalg_b, cholb, optimize=True)
    e2_2_1 = (e2_0_1 + e2_0_2) * gci2g

    if c2_antisymm:
        e2_2_2 = -einsum(
            "wX,wX->w",
            einsum("pqX,wpq->wX", chol, calg_c2g_aa_G + calg_c2g_ab_a_G, optimize=True),
            chol_g,
            optimize=True,
        )
        e2_2_2 -= einsum(
            "wX,wX->w",
            einsum("pqX,wpq->wX", cholb, calg_c2g_bb_G + calg_c2g_ab_b_G, optimize=True),
            chol_g,
            optimize=True,
        )
        e2_2_3 = einsum("wpqX,prX,wqr->w", gl_a, rchola, calg_c2g_aa_G + calg_c2g_ab_a_G, optimize=True)
        e2_2_3 += einsum("wpqX,prX,wqr->w", gl_b, rcholb, calg_c2g_bb_G + calg_c2g_ab_b_G, optimize=True)
    else:
        # UHF-spin equivalent of RHF lc2gcalGG construction.
        c2g_aa_c_calg = einsum("wpa,wia->wip", calg_a, c2g_aa_c, optimize=True)
        c2g_aa_x_calg = einsum("wpa,wja->wjp", calg_a, c2g_aa_x, optimize=True)
        c2g_ab_a_calg = einsum("wpa,wia->wip", calg_a, c2g_ab_a, optimize=True)
        c2gcalG_a = c2g_aa_c_calg - c2g_aa_x_calg + c2g_ab_a_calg
        c2gcalGG_a = einsum("wip,wir->wpr", c2gcalG_a, ghalfa, optimize=True)
        lc2gcalGG_c_a = einsum("prX,wpr->wX", chol, c2gcalGG_a, optimize=True)
        lc2gcalGG_x_a = einsum("sqX,wqr->wXsr", chol, c2gcalGG_a, optimize=True)

        c2g_bb_c_calg = einsum("wpa,wia->wip", calg_b, c2g_bb_c, optimize=True)
        c2g_bb_x_calg = einsum("wpa,wja->wjp", calg_b, c2g_bb_x, optimize=True)
        c2g_ab_b_calg = einsum("wpa,wia->wip", calg_b, c2g_ab_b, optimize=True)
        c2gcalG_b = c2g_bb_c_calg - c2g_bb_x_calg + c2g_ab_b_calg
        c2gcalGG_b = einsum("wip,wir->wpr", c2gcalG_b, ghalfb, optimize=True)
        lc2gcalGG_c_b = einsum("prX,wpr->wX", cholb, c2gcalGG_b, optimize=True)
        lc2gcalGG_x_b = einsum("sqX,wqr->wXsr", cholb, c2gcalGG_b, optimize=True)

        e2_2_2 = -einsum("wX,wX->w", lc2gcalGG_c_a + lc2gcalGG_c_b, chol_g, optimize=True)
        e2_2_3 = einsum("wXsr,wXsr->w", cholg_mat_a, lc2gcalGG_x_a, optimize=True)
        e2_2_3 += einsum("wXsr,wXsr->w", cholg_mat_b, lc2gcalGG_x_b, optimize=True)

    e2_2_4 = e2_2_4_aa + e2_2_4_bb + e2_2_4_ab
    e2_2_5 = e2_2_5_aa + e2_2_5_bb
    e1 = e1_0 + e1_1_1 + e1_1_2 + e1_2_1 + e1_2_2 + e1_2_3 + e1_2_4
    e2 = e2_0_1 + e2_0_2 + e2_1_1 + e2_1_2 + e2_1_3 + e2_2_1 + e2_2_2 + e2_2_3 + e2_2_4 + e2_2_5
    return e1 / ovlp_ratio, e2 / ovlp_ratio


def local_energy_ucisd_batch(hamiltonian: GenericRealChol, walkers: UHFWalkers, trial: CISD):
    """Calculate local energy for UCISD trial wavefunction.

    Parameters
    ----------
    hamiltonian : object
        Hamiltonian object.
    walkers : object
        WalkerBatch object (this stores some intermediates for the particular trial wfn).
    trial : object
        Trial wavefunction object.

    Returns
    -------
    eloc : float / complex
        Local energy.
    """
    use_cuquantum = _use_cuquantum_einsum()
    if hamiltonian.h1eb is None or hamiltonian.cholb is None:
        hamiltonian.construct_beta_integrals(trial.mo_coeffb)
    h1eb = hamiltonian.h1eb
    chol = hamiltonian.chol.reshape((hamiltonian.nbasis, hamiltonian.nbasis, hamiltonian.nchol))
    cholb = hamiltonian.cholb

    nwalkers = walkers.nwalkers
    energy = xp.zeros((nwalkers, 3), dtype=xp.complex128)

    ovlp_ratio_all = trial.ovlp_ratio_cisd
    ovlp_is_walker_resolved = (
        hasattr(ovlp_ratio_all, "shape")
        and len(ovlp_ratio_all.shape) > 0
        and ovlp_ratio_all.shape[0] == nwalkers
    )

    for w_sls in _get_walker_chunks(nwalkers, hamiltonian.nchol, hamiltonian.nbasis):
        ghalfa_chunk = walkers.ghalfa[w_sls]
        ghalfb_chunk = walkers.ghalfb[w_sls]
        if ovlp_is_walker_resolved:
            ovlp_chunk = ovlp_ratio_all[w_sls]
        else:
            ovlp_chunk = ovlp_ratio_all

        e1b_chunk, e2b_chunk = _local_energy_ucisd_kernel(
            hamiltonian.h1e[0],
            h1eb,
            chol,
            cholb,
            ghalfa_chunk,
            ghalfb_chunk,
            trial.rh1a,
            trial.rh1b,
            trial.c1a,
            trial.c1b,
            trial.c2aa,
            trial.c2ab,
            trial.c2bb,
            trial.rchola,
            trial.rcholb,
            ovlp_chunk,
            trial.c2_antisymm,
            use_cuquantum=use_cuquantum,
        )

        energy[w_sls, 0] = e1b_chunk + e2b_chunk + hamiltonian.ecore
        energy[w_sls, 1] = e1b_chunk + hamiltonian.ecore
        energy[w_sls, 2] = e2b_chunk
    return energy

def local_energy_cisd_batch(system, hamiltonian: GenericRealChol, walkers: UHFWalkers, trial: CISD):
    """Calculate local energy for CISD trial wavefunction.

    Parameters
    ----------
    hamiltonian : object
        Hamiltonian object.
    walkers : object
        WalkerBatch object (this stores some intermediates for the particular trial wfn).
    trial : object
        Trial wavefunction object.

    Returns
    -------
    eloc : float / complex
        Local energy.
    """
    if walkers.rhf:
        return local_energy_rcisd_batch(hamiltonian, walkers, trial)
    else:
        return local_energy_ucisd_batch(hamiltonian, walkers, trial)