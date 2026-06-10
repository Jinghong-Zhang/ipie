from math import ceil
from ipie.utils.backend import arraylib as xp

def slice_gf_kpq_k_given_q(gf, iq, kpq_mat):
    """
    slice the Green's function G^{w}_{pk+q, rk} to g^{w}_{kpr} for a given q
    """
    nw = gf.shape[0]
    nk = gf.shape[1]
    nocc = gf.shape[2]
    nbsf = gf.shape[-1]
    kpq = kpq_mat[iq][None, :, None, None]
    w = xp.arange(nw)[:, None, None, None]
    k = xp.arange(nk)[None, :, None, None]
    i = xp.arange(nocc)[None, None, :, None]
    p = xp.arange(nbsf)[None, None, None, :]
    gf_kpq = gf[w, kpq, i, k, p]
    return gf_kpq

def slice_gf_k_kpq_given_q(gf, iq, kpq_mat):
    """
    slice the Green's function G^{w}_{pk, rk+q} to g^{w}_{kpr} for a given q
    """
    nw = gf.shape[0]
    nk = gf.shape[1]
    nocc = gf.shape[2]
    nbsf = gf.shape[-1]
    kpq = kpq_mat[iq][None, :, None, None]
    w = xp.arange(nw)[:, None, None, None]
    k = xp.arange(nk)[None, :, None, None]
    i = xp.arange(nocc)[None, None, :, None]
    p = xp.arange(nbsf)[None, None, None, :]
    gf_kpq = gf[w, k, i, kpq, p]
    return gf_kpq

def slice_gf_kpq_k_qlis(gf, iq_lis, kpq_mat):
    """
    slice the Green's function G^{w}_{pk+q, rk} to g^{w}_{kpr} for a list of q
    Returns:
        gf_kpq_k: the sliced Green's function G[q, k, w, p, r]
    """
    nk = gf.shape[1]
    ik_q = xp.repeat(xp.arange(nk), len(iq_lis)).reshape(nk, len(iq_lis)).T
    kpq = kpq_mat[iq_lis]
    gf_kpq_lis = gf[:, kpq, :, ik_q, :]
    return gf_kpq_lis

def slice_gf_k_kpq_qlis(gf, iq_lis, kpq_mat):
    """
    slice the Green's function G^{w}_{pk, rk+q} to g^{w}_{kpr} for a list of q
    Returns:
        gf_k_kpq: the sliced Green's function G[q, k, w, p, r]
    """
    nk = gf.shape[1]
    ik_q = xp.repeat(xp.arange(nk), len(iq_lis)).reshape(nk, len(iq_lis)).T
    kpq = kpq_mat[iq_lis]
    gf_k_kpq_lis = gf[:, ik_q, :, kpq, :]
    return gf_k_kpq_lis

def slice_cgto_kpq(cgto, kpq_mat, iq_lis):
    nq = len(iq_lis)
    nk = cgto.shape[0]
    q_id = iq_lis[:, None]
    k_id = xp.arange(nk)[None, :]
    kpq = kpq_mat[q_id, k_id]
    cgto_kpq = cgto[kpq]
    return cgto_kpq

def contract_gf_cgto_kpq_k(gf_kpq, cgto, cgto_kpq, iq_real):
    """
    perform the contraction: psi^{k+q}_{pP}.conj(), psi^{k}_{rP}, G^{w}_{pk+q, rk} -> X^w_{Pq}
    """
    out_q = xp.einsum("kPp, kPr, wkpr -> wP", cgto_kpq.conj(), cgto, gf_kpq, optimize=True)
    return out_q

def contract_gf_cgto12_kpq_k(gf_kpq, cgto1_kpq, cgto2, iq_real):
    """
    perform the contraction: psi^{k+q}_{pP}.conj(), psi^{k}_{rP}, G^{w}_{pk+q, rk} -> X^w_{Pq}
    """
    out_q = xp.einsum("kPp, kPr, wkpr -> wP", cgto1_kpq.conj(), cgto2, gf_kpq, optimize=True)
    return out_q


def contract_gf_cgto12_kmq_k(gf_kmq, cgto1_kmq, cgto2, iq_real):
    """
    perform the contraction: psi^{k+q}_{pP}.conj(), psi^{k}_{rP}, G^{w}_{pk+q, rk} -> X^w_{Pq}
    """
    out_q = xp.einsum("kPp, kPr, wkpr -> wP", cgto1_kmq.conj(), cgto2, gf_kmq, optimize=True)
    return out_q

def contract_gf_cgto_k_kpq(gf_kpq, cgto, cgto_kpq, iq_real):
    """
    perform the contraction: psi^{k}_{pP}.conj(), psi^{k+q}_{rP}, G^{w}_{pk, rk+q} -> X^w_{Pq}
    """
    out_q = xp.einsum("kPp, kPr, wkpr -> wP", cgto.conj(), cgto_kpq, gf_kpq, optimize=True)
    return out_q

def contract_gf_cgto12_k_kpq(gf_kpq, cgto1, cgto2_kpq, iq_real):
    """
    perform the contraction: psi1^{k}_{pP}.conj(), psi2^{k+q}_{rP}, G^{w}_{pk, rk+q} -> X^w_{Pq}
    """
    out_q = xp.einsum("kPp, kPr, wkpr -> wP", cgto1.conj(), cgto2_kpq, gf_kpq, optimize=True)
    return out_q


def contract_gf_cgto12_k_kmq(gf_kmq, cgto1, cgto2_kmq, iq_real):
    """
    perform the contraction: psi1^{k}_{pP}.conj(), psi2^{k-q}_{rP}, G^{w}_{pk, rk-q} -> X^w_{Pq}
    """
    out_q = xp.einsum("kPp, kPr, wkpr -> wP", cgto1.conj(), cgto2_kmq, gf_kmq, optimize=True)
    return out_q

def contract_cgto_gf_batch(rcgtoa_kmq, halfrot_cgto, ga_kmq):
    nk, nisdf, nocc = rcgtoa_kmq.shape
    nq = ga_kmq.shape[0]
    nw = ga_kmq.shape[2]
    nbsf = halfrot_cgto.shape[-1] 
    intermediate_mem = nisdf * nocc * nw * nq * nk * 16 /1024**3
    max_mem = 2.0
    # slice the walker dimension to fit in memory
    nw_slice = ceil(intermediate_mem / max_mem)
    num_chunks = ceil(nw / nw_slice)
    nw_left = nw
    slices_w = []
    for i_chunk in range(num_chunks):
        if nw_left == 0:
            break
        nw_chunk = min(nw_left, nw_slice)
        nw_left -= nw_chunk
        slices_w.append(slice(i_chunk * nw_slice, i_chunk * nw_slice + nw_chunk))

    buff = xp.empty(nw_slice * nisdf * nbsf * nq * nk, dtype=xp.complex128)
    halfrot_cgto = halfrot_cgto.transpose(1, 0, 2).reshape(nisdf, nk * nocc)  # k, P, r -> P, kr
    Y_qwP = xp.empty((nq, nw, nisdf), dtype=xp.complex128)
    for slice_w in slices_w:
        g_slice = ga_kmq[:, :, slice_w, :, :].copy()
        nw_chunk = g_slice.shape[2]
        size_req = nw_chunk * nisdf * nocc * nq * nk
        cgto_g = buff[:size_req].reshape(nq, nk, nisdf, nw_chunk* nocc)
        g_slice = g_slice.transpose(0, 1, 3, 2, 4).reshape(nq, nk, nbsf, nw_chunk * nocc)  # qkwpr -> qkpwr
        xp.matmul(rcgtoa_kmq.conj(), g_slice, out=cgto_g)  # qkPp, qkpwr -> q,k,P,wr , kPr ->qwP
        cgto_g = cgto_g.reshape(nq, nk, nisdf, nw_chunk, nocc).transpose(0, 3, 2, 1, 4).reshape(nq, nw_chunk, nisdf, nk* nocc) #q, w, P, kr
        Y_qwP[:, slice_w, :] = xp.sum(cgto_g * halfrot_cgto[xp.newaxis, xp.newaxis, :, :], axis=-1)

    return Y_qwP
