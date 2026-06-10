"""Frozen pre-optimization KptISDF local-energy kernels (HEAD 7cb863c).

Copied verbatim from ipie/estimators/local_energy_kpt_sd.py before the
local-energy optimization commits, so benchmarks can compare old vs new
end to end. ``patch_old_energy()`` monkeypatches them into the live module.
"""

from math import ceil

import numpy

from ipie.utils.backend import arraylib as xp
from ipie.utils.contract_gf_cgto import (
    contract_gf_cgto12_k_kpq,
    contract_gf_cgto12_kpq_k,
    slice_gf_k_kpq_given_q,
    slice_gf_kpq_k_given_q,
)

def kpt_isdf_ecoul_kernel_gpu(MPQ, halfrot_cgtoa, halfrot_cgtob, cgto, Ghalfa_batch, Ghalfb_batch, kpq_mat, Sset, Qplus):
    nk = cgto.shape[0]
    nwalkers = Ghalfa_batch.shape[0]
    ecoul = xp.zeros(nwalkers, dtype=numpy.complex128)
    for iq in range(len(Sset)):
        iq_real = Sset[iq]
        MPQ_iq = MPQ[iq]
        ikpq = kpq_mat[iq_real]
        cgto_kpq = cgto[ikpq]
        rcgtoa_kpq = halfrot_cgtoa[ikpq]
        rcgtob_kpq = halfrot_cgtob[ikpq]
        ga_k_kpq = slice_gf_k_kpq_given_q(Ghalfa_batch, iq_real, kpq_mat)
        gb_k_kpq = slice_gf_k_kpq_given_q(Ghalfb_batch, iq_real, kpq_mat)
        v1_wP = contract_gf_cgto12_k_kpq(ga_k_kpq, halfrot_cgtoa, cgto_kpq, iq_real) + contract_gf_cgto12_k_kpq(gb_k_kpq, halfrot_cgtob, cgto_kpq, iq_real)
        del ga_k_kpq
        del gb_k_kpq
        ga_kpq_k = slice_gf_kpq_k_given_q(Ghalfa_batch, iq_real, kpq_mat)
        gb_kpq_k = slice_gf_kpq_k_given_q(Ghalfb_batch, iq_real, kpq_mat)
        # v2_wP = contract_gf_cgto12_kpq_k(Ghalfa_batch, halfrot_cgtoa, cgto, iq_real, kpq_mat) + contract_gf_cgto12_kpq_k(Ghalfb_batch, halfrot_cgtob, cgto, iq_real, kpq_mat)
        v2_wP = contract_gf_cgto12_kpq_k(ga_kpq_k, rcgtoa_kpq, cgto, iq_real) + contract_gf_cgto12_kpq_k(gb_kpq_k, rcgtob_kpq, cgto, iq_real)
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
        v1_wP = contract_gf_cgto12_k_kpq(ga_k_kpq, halfrot_cgtoa, cgto_kpq, iq_real) + contract_gf_cgto12_k_kpq(gb_k_kpq, halfrot_cgtob, cgto_kpq, iq_real)
        del ga_k_kpq
        del gb_k_kpq
        ga_kpq_k = slice_gf_kpq_k_given_q(Ghalfa_batch, iq_real, kpq_mat)
        gb_kpq_k = slice_gf_kpq_k_given_q(Ghalfb_batch, iq_real, kpq_mat)
        v2_wP = contract_gf_cgto12_kpq_k(ga_kpq_k, rcgtoa_kpq, cgto, iq_real) + contract_gf_cgto12_kpq_k(gb_kpq_k, rcgtob_kpq, cgto, iq_real)
        del ga_kpq_k
        del gb_kpq_k
        ecoul += 2. * xp.sum((v1_wP @ MPQ_iq) * v2_wP, axis=1)
    return 0.5 * ecoul / nk

def kpt_isdf_ecoul_rhf_kernel_gpu(MPQ, halfrot_cgtoa, cgto, Ghalfa_batch, kpq_mat, Sset, Qplus):
    nk = cgto.shape[0]
    nwalkers = Ghalfa_batch.shape[0]
    ecoul = xp.zeros(nwalkers, dtype=numpy.complex128)
    for iq in range(len(Sset)):
        iq_real = Sset[iq]
        MPQ_iq = MPQ[iq]
        ikpq = kpq_mat[iq_real]
        cgto_kpq = cgto[ikpq]
        rcgtoa_kpq = halfrot_cgtoa[ikpq]
        ga_k_kpq = slice_gf_k_kpq_given_q(Ghalfa_batch, iq_real, kpq_mat)
        v1_wP = contract_gf_cgto12_k_kpq(ga_k_kpq, halfrot_cgtoa, cgto_kpq, iq_real)
        del ga_k_kpq
        ga_kpq_k = slice_gf_kpq_k_given_q(Ghalfa_batch, iq_real, kpq_mat)
        v2_wP = contract_gf_cgto12_kpq_k(ga_kpq_k, rcgtoa_kpq, cgto, iq_real)
        del ga_kpq_k
        ecoul += xp.sum((v1_wP @ MPQ_iq) * v2_wP, axis=1)

    for iq in range(len(Sset), len(Sset) + len(Qplus)):
        iq_real = Qplus[iq - len(Sset)]
        MPQ_iq = MPQ[iq]
        ikpq = kpq_mat[iq_real]
        cgto_kpq = cgto[ikpq]
        rcgtoa_kpq = halfrot_cgtoa[ikpq]
        ga_k_kpq = slice_gf_k_kpq_given_q(Ghalfa_batch, iq_real, kpq_mat)
        v1_wP = contract_gf_cgto12_k_kpq(ga_k_kpq, halfrot_cgtoa, cgto_kpq, iq_real)
        del ga_k_kpq
        ga_kpq_k = slice_gf_kpq_k_given_q(Ghalfa_batch, iq_real, kpq_mat)
        v2_wP = contract_gf_cgto12_kpq_k(ga_kpq_k, rcgtoa_kpq, cgto, iq_real)
        del ga_kpq_k
        ecoul += 2. * xp.sum((v1_wP @ MPQ_iq) * v2_wP, axis=1)
    return 2. * ecoul / nk

def contract_psi_G_psi(psiiP_slice, psiqQ_slice, G, buff, nP, nQ):
    nw, nk, nocc, nbsf = G.shape[0], G.shape[1], G.shape[2], G.shape[4]
    G = G.transpose(3, 0, 1, 2, 4).reshape(nk, nw * nk * nocc, nbsf)  # k', wki, q
    size_Gpsi = nw * nocc * nk**2 * nQ
    Gpsi = buff[:size_Gpsi].reshape(nk, nw * nocc * nk, nQ)
    xp.matmul(G, psiqQ_slice, out=Gpsi)  # k', wki, Q
    Gpsi = Gpsi.reshape(nk, nw, nk, nocc, nQ).transpose(2, 1, 4, 0, 3).reshape(nk, nw * nQ * nk, nocc) # k, wQk', i
    size_TPQ = nk**2 * nw * nP * nQ
    TPQ = buff[:size_TPQ].reshape(nk, nw * nQ * nk, nP)
    xp.matmul(Gpsi, psiiP_slice, out=TPQ)
    return TPQ

def X_contract_cupy_lowk(halfrot_cgtoa, phikr_kpq, M_PQ_iq, phiki_kpq, cgto, Ga_chunk, G_kpq_kprimepq_chunk, buff1, buff2, slices_isdf):
    # low k algorithm
    nw = Ga_chunk.shape[0]
    nk = cgto.shape[0]
    exx = xp.zeros((nw,), dtype=xp.complex128)

    for ia, P in enumerate(slices_isdf): # ia is the chunk index
        psi_iP_k = halfrot_cgtoa[:, P, :].transpose(0, 2, 1).conj()  # k, i, P
        phip_kpq_P = phikr_kpq[:, P, :].transpose(0, 2, 1) # k+q, p, P
        nP = P.stop - P.start
        # for ib, Q in enumerate(slices_isdf[ia:], start=ia): # only compute upper triangle
        for ib, Q in enumerate(slices_isdf):
            nQ = Q.stop - Q.start
            psi_iQ_kp = cgto[:, Q, :].transpose(0, 2, 1)  # k', q, Q
            phij_kpq_Q = phiki_kpq[:, Q, :].transpose(0, 2, 1).conj() # k'+q, j, Q
            TPQ = contract_psi_G_psi(psi_iP_k, psi_iQ_kp, Ga_chunk, buff1, nP, nQ)
            TQP_kpq = contract_psi_G_psi(phij_kpq_Q, phip_kpq_P, G_kpq_kprimepq_chunk, buff2, nQ, nP)
            TPQ = TPQ.reshape(nk, nw, nQ, nk, nP).transpose(1, 0, 3, 4, 2).reshape(nw, nk * nk, nP, nQ)  # wkk', P, Q
            TQP_kpq = TQP_kpq.reshape(nk, nw, nP, nk, nQ).transpose(1, 3, 0, 2, 4).reshape(nw, nk * nk, nP, nQ)  # wkk', P, Q
            Tsq = xp.sum(TPQ * TQP_kpq, axis=1) # w, P, Q
            M_PQ_iq_sliced = M_PQ_iq[P, Q].astype(xp.complex128, copy=False)  # P, Q
            exx += (Tsq.reshape(nw, nP * nQ) @ M_PQ_iq_sliced.ravel())
    return exx

def contraction_exx(halfrot_cgtoa, phikr_kpq, M_PQ_iq, phiki_kpq, cgto, Ga_chunk, G_kpq_kprimepq_chunk, nk, nbsf, nisdf, nocc, nw, max_mem = 4.0):
    # slice over Q in the outside loop
    exx = 0.0 + 0.0j
    intermediate_mem = (nbsf * nisdf * nk**2 * nw + nbsf * nisdf * nk * nocc * nw) * 16 / 1024**3  # GB
    num_chunks = ceil(intermediate_mem / max_mem)
    nisdf_per_chunk = ceil(nisdf / num_chunks)
    nisdf_left = nisdf
    slices_isdf = []
    for i_chunk in range(num_chunks):
        if nisdf_left == 0:
            break
        nisdf_chunk = min(nisdf_left, nisdf_per_chunk)
        nisdf_left -= nisdf_chunk
        slices_isdf.append(slice(i_chunk * nisdf_per_chunk, i_chunk * nisdf_per_chunk + nisdf_chunk))
    buff1 = xp.empty(nk * nocc * nbsf * nisdf_per_chunk, dtype=xp.complex128)
    max_dim = max(nisdf_per_chunk, nocc)
    buff2 = xp.empty(max_dim * nbsf * nk**2 * nw, dtype=xp.complex128)
    # 1. psi^k_iP psi^k+[q]_pP -> rho^{k, [q]}_{ipP}
    mem_rhokPip = nk * nocc * nisdf * nbsf * 16 / 1024**3  # GB
    if mem_rhokPip < max_mem:
        rho_kpq_ipP = halfrot_cgtoa.conj()[:, :, :, xp.newaxis] * phikr_kpq[:, :, xp.newaxis, :] # k, P, i, p
        rho_kpq_ipP = rho_kpq_ipP.transpose(0, 2, 3, 1).reshape(nk * nocc * nbsf, nisdf)
        G_kpq_kprimepq_chunk = G_kpq_kprimepq_chunk.transpose(1, 2, 0, 3, 4).reshape(nk, nocc, nw * nk * nbsf) # k', j, wkp
        Ga_chunk = Ga_chunk.reshape(nw, -1)
        # 2. rho^{k, [q]}_{ipP} M^{P, Q}_{iq} -> rho^{k, [q]}_{iQp}
        for i_sls in slices_isdf:
            nisdf_chunk = i_sls.stop - i_sls.start
            M_PQ_iq_sliced = M_PQ_iq[:, i_sls]
            size_A = nk * nocc * nisdf_chunk * nbsf
            A_kipQ = buff1[:size_A].reshape(nk * nocc * nbsf, nisdf_chunk)
            xp.matmul(rho_kpq_ipP, M_PQ_iq_sliced, out=A_kipQ)  # kip, Q
            psi_kQj = phiki_kpq.conj()[:, i_sls, :] # k', Q, j
            size_B = nisdf_chunk * nbsf * nk**2 * nw
            B_kpQwkp = buff2[:size_B].reshape(nk, nisdf_chunk, nw * nk * nbsf)
            xp.matmul(psi_kQj, G_kpq_kprimepq_chunk, out=B_kpQwkp)  # k', Q, wkp
            B_kpQwkp = B_kpQwkp.reshape(nk, nisdf_chunk, nw, nk, nbsf).transpose(3, 1, 2, 0, 4).reshape(nk * nisdf_chunk, nw*nk, nbsf) # kQ, wk', p
            A_kipQ = A_kipQ.reshape(nk, nocc, nbsf, nisdf_chunk).transpose(0, 3, 2, 1).reshape(nk * nisdf_chunk, nbsf, nocc) # kQ, p, i
            size_C = nk**2 * nisdf_chunk * nw * nocc
            C_kQwkpi = buff2[:size_C].reshape(nk * nisdf_chunk, nw * nk, nocc)
            xp.matmul(B_kpQwkp, A_kipQ, out=C_kQwkpi)  # kQ, wk', i
            C_kQwkpi = C_kQwkpi.reshape(nk, nisdf_chunk, nw, nk, nocc).transpose(3, 0, 2, 4, 1).reshape(nk, nk * nw * nocc, nisdf_chunk) #k', kwi, Q
            size_D = nw * nk**2 * nocc * nbsf
            D_kpkwiq = buff2[:size_D].reshape(nk, nk * nw * nocc, nbsf)
            xp.matmul(C_kQwkpi, cgto[:, i_sls, :], out=D_kpkwiq) # k', kwi, q
            D_kpkwiq = D_kpkwiq.reshape(nk, nk, nw, nocc, nbsf).transpose(2, 1, 3, 0, 4).reshape(nw, -1)
            exx += xp.sum(D_kpkwiq * Ga_chunk, axis=-1)
    else:
        G_kpq_kprimepq_chunk = G_kpq_kprimepq_chunk.transpose(1, 2, 0, 3, 4).reshape(nk, nocc, nw * nk * nbsf) # k', j, wkp
        Ga_chunk = Ga_chunk.reshape(nw, -1)
        # 2. rho^{k, [q]}_{ipP} M^{P, Q}_{iq} -> rho^{k, [q]}_{iQp}
        for i_sls in slices_isdf:
            nisdf_chunk = i_sls.stop - i_sls.start
            A_kipQ = xp.zeros((nk * nocc * nbsf, nisdf_chunk), dtype=xp.complex128)
            for j_sls in slices_isdf:
                nisdf_chunk_P = j_sls.stop - j_sls.start
                rho_kpq_ipP = halfrot_cgtoa.conj()[:, j_sls, :, xp.newaxis] * phikr_kpq[:, j_sls, xp.newaxis, :] # k, P, i, p
                rho_kpq_ipP = rho_kpq_ipP.transpose(0, 2, 3, 1).reshape(nk * nocc * nbsf, nisdf_chunk_P)
                M_PQ_iq_sliced = M_PQ_iq[j_sls, i_sls]
                size_A = nk * nocc * nisdf_chunk * nbsf
                temp = buff1[:size_A].reshape(nk * nocc * nbsf, nisdf_chunk)
                xp.matmul(rho_kpq_ipP, M_PQ_iq_sliced, out=temp)  # kip, Q
                A_kipQ += temp
            psi_kQj = phiki_kpq.conj()[:, i_sls, :] # k', Q, j
            size_B = nisdf_chunk * nbsf * nk**2 * nw
            B_kpQwkp = buff2[:size_B].reshape(nk, nisdf_chunk, nw * nk * nbsf)
            xp.matmul(psi_kQj, G_kpq_kprimepq_chunk, out=B_kpQwkp)  # k', Q, wkp
            B_kpQwkp = B_kpQwkp.reshape(nk, nisdf_chunk, nw, nk, nbsf).transpose(3, 1, 2, 0, 4).reshape(nk * nisdf_chunk, nw*nk, nbsf) # kQ, wk', p
            A_kipQ = A_kipQ.reshape(nk, nocc, nbsf, nisdf_chunk).transpose(0, 3, 2, 1).reshape(nk * nisdf_chunk, nbsf, nocc) # kQ, p, i
            size_C = nk**2 * nisdf_chunk * nw * nocc
            C_kQwkpi = buff2[:size_C].reshape(nk * nisdf_chunk, nw * nk, nocc)
            xp.matmul(B_kpQwkp, A_kipQ, out=C_kQwkpi)  # kQ, wk', i
            C_kQwkpi = C_kQwkpi.reshape(nk, nisdf_chunk, nw, nk, nocc).transpose(3, 0, 2, 4, 1).reshape(nk, nk * nw * nocc, nisdf_chunk) #k', kwi, Q
            size_D = nw * nk**2 * nocc * nbsf
            D_kpkwiq = buff2[:size_D].reshape(nk, nk * nw * nocc, nbsf)
            xp.matmul(C_kQwkpi, cgto[:, i_sls, :], out=D_kpkwiq) # k', kwi, q
            D_kpkwiq = D_kpkwiq.reshape(nk, nk, nw, nocc, nbsf).transpose(2, 1, 3, 0, 4).reshape(nw, -1)
            exx += xp.sum(D_kpkwiq * Ga_chunk, axis=-1)
    return exx

def kpt_isdf_exx_kernel_gpu(MPQ, halfrot_cgtoa, cgto, Ghalfa_batch, kpq_mat, Sset, Qplus):
    nwalker, nk, nocc, _, nbsf = Ghalfa_batch.shape
    nisdf = MPQ.shape[-1]
    if nbsf > 8 * nk:
        # lowk algo
        exx = xp.zeros((nwalker,), dtype=xp.complex128)
        nisdf = MPQ.shape[-1]

        w_idx = xp.arange(nwalker)[:, None, None, None, None]  # shape (W,1,1,1,1)
        k_idx = xp.arange(nk)[None, :, None, None, None]  # shape (1,nk,1,1,1)
        i_idx = xp.arange(nocc)[None, None, :, None, None]  # shape (1,1,nocc,1,1)
        kprime_idx = xp.arange(nk)[None, None, None, :, None]  # shape (1,1,1,nk,1)
        p_idx = xp.arange(nbsf)[None, None, None, None, :] # shape (1,1,1,1,nbsf)
   
        intermediate_mem = nisdf * nisdf * nk**2 * nwalker * 3 * 16 / 1024**3
        max_mem = 8.0
        num_chunks = ceil(intermediate_mem / max_mem)
        num_chunk_per_dim = ceil(num_chunks ** 0.5)
        nisdf_per_chunk = ceil(nisdf / num_chunk_per_dim)
        nisdf_left = nisdf
        slices_isdf = []
        for i_chunk in range(num_chunks):
            if nisdf_left == 0:
                break
            nisdf_chunk = min(nisdf_left, nisdf_per_chunk)
            nisdf_left -= nisdf_chunk
            slices_isdf.append(slice(i_chunk * nisdf_per_chunk, i_chunk * nisdf_per_chunk + nisdf_chunk))
        max_dim = max(nisdf_per_chunk, nocc)
        buff1 = xp.empty(nwalker * nk**2 * max_dim**2, dtype=xp.complex128)
        buff2 = xp.empty(nwalker * nk**2 * max_dim**2, dtype=xp.complex128)
        for iq in range(len(Sset)):
            iq_real = Sset[iq]
            ikpq = kpq_mat[iq_real]
            phikr_kpq = cgto[ikpq]
            phiki_kpq = halfrot_cgtoa[ikpq]
            kpq_idx = kpq_mat[k_idx, iq_real]
            kprimepq_idx = kpq_mat[kprime_idx, iq_real]
            G_kpq_kprimepq_chunk = Ghalfa_batch[w_idx, kpq_idx, i_idx, kprimepq_idx, p_idx]
            MPQ_iq = MPQ[iq]
            exx_iq = X_contract_cupy_lowk(halfrot_cgtoa, phikr_kpq, MPQ_iq, phiki_kpq, cgto, Ghalfa_batch, G_kpq_kprimepq_chunk, buff1, buff2, slices_isdf)
            exx -= exx_iq

        for iq in range(len(Sset), len(Sset) + len(Qplus)):
            iq_real = Qplus[iq - len(Sset)]
            ikpq = kpq_mat[iq_real]
            phikr_kpq = cgto[ikpq]
            phiki_kpq = halfrot_cgtoa[ikpq]
            kpq_idx = kpq_mat[k_idx, iq_real]
            kprimepq_idx = kpq_mat[kprime_idx, iq_real]
            G_kpq_kprimepq_chunk = Ghalfa_batch[w_idx, kpq_idx, i_idx, kprimepq_idx, p_idx]
            MPQ_iq = MPQ[iq]
            exx_iq = X_contract_cupy_lowk(halfrot_cgtoa, phikr_kpq, MPQ_iq, phiki_kpq, cgto, Ghalfa_batch, G_kpq_kprimepq_chunk, buff1, buff2, slices_isdf)
            exx -= 2. * exx_iq

    else:
        # large k algo
        w_idx = xp.arange(nwalker)[:, None, None, None, None]  # shape (W,1,1,1,1)
        k_idx = xp.arange(nk)[None, :, None, None, None]  # shape (1,nk,1,1,1)
        i_idx = xp.arange(nocc)[None, None, :, None, None]  # shape (1,1,nocc,1,1)
        kprime_idx = xp.arange(nk)[None, None, None, :, None]  # shape (1,1,1,nk,1)
        p_idx = xp.arange(nbsf)[None, None, None, None, :] # shape (1,1,1,1,nbsf)

        exx = xp.zeros(nwalker, dtype=numpy.complex128)

        if nk < 64:
            intermediate_mem = nwalker * nisdf * nk * nk * nbsf * 6 * 16 / 1024 ** 3
        else:
            intermediate_mem = nwalker * nisdf * nk * nbsf * 6 * 16 / 1024 ** 3
        free_bytes = xp.cuda.Device().mem_info[0]
        free_gb = free_bytes / 1024**3.0
        max_mem = .7 * free_gb
        num_chunks = max(1, ceil(intermediate_mem / max_mem))
        chunk_size = ceil(nwalker / num_chunks)
        nw_left = nwalker
        for i_chunk in range(num_chunks):
            if nw_left == 0:
                break
            n_chunk = min(nw_left, chunk_size)
            nw_left -= n_chunk
            w_sls = xp.arange(nwalker)[i_chunk * chunk_size: i_chunk * chunk_size + n_chunk]
            Ga_chunk = Ghalfa_batch[w_sls]
            w_chunk_idx = xp.arange(n_chunk)[:, None, None, None, None]  # shape (W_chunk,1,1,1,1)

            for iq in range(len(Sset)):
                iq_real = Sset[iq]
                ikpq = kpq_mat[iq_real]
                phikr_kpq = cgto[ikpq]
                phiki_kpq = halfrot_cgtoa[ikpq]
                kpq_idx = kpq_mat[k_idx, iq_real]
                kprimepq_idx = kpq_mat[kprime_idx, iq_real]
                G_kpq_kprimepq_chunk = Ga_chunk[w_chunk_idx, kpq_idx, i_idx, kprimepq_idx, p_idx]
                MPQ_iq = MPQ[iq]
                # exx[w_sls] -= contract('kPi, kPp, PQ, KQj, KQq, wkiKq, wKjkp -> w', halfrot_cgtoa.conj(), phikr_kpq, MPQ_iq, phiki_kpq.conj(), cgto, Ga_chunk, G_kpq_kprimepq_chunk, options=network_opts)
                exx[w_sls] -= contraction_exx(halfrot_cgtoa, phikr_kpq, MPQ_iq, phiki_kpq, cgto, Ga_chunk, G_kpq_kprimepq_chunk, nk, nbsf, nisdf, nocc, n_chunk)
                xp.cuda.get_current_stream().synchronize()
                del G_kpq_kprimepq_chunk

            for iq in range(len(Sset), len(Sset) + len(Qplus)):
                iq_real = Qplus[iq - len(Sset)]
                ikpq = kpq_mat[iq_real]
                phikr_kpq = cgto[ikpq]
                phiki_kpq = halfrot_cgtoa[ikpq]
                kpq_idx = kpq_mat[k_idx, iq_real]
                kprimepq_idx = kpq_mat[kprime_idx, iq_real]
                G_kpq_kprimepq_chunk = Ga_chunk[w_chunk_idx, kpq_idx, i_idx, kprimepq_idx, p_idx]
                MPQ_iq = MPQ[iq]
                exx[w_sls] -= 2. * contraction_exx(halfrot_cgtoa, phikr_kpq, MPQ_iq, phiki_kpq, cgto, Ga_chunk, G_kpq_kprimepq_chunk, nk, nbsf, nisdf, nocc, n_chunk)
                xp.cuda.get_current_stream().synchronize()
                del G_kpq_kprimepq_chunk

    return 0.5 * exx / nk
def local_energy_kpt_single_det_uhf_isdf_gpu(system, hamiltonian, walkers, trial):
    """Compute local energy for walker batch (all walkers at once).

    Single determinant RHF case.

    Parameters
    ----------
    system : system object
        System being studied.
    hamiltonian : hamiltonian object
        Hamiltonian being studied.
    walkers : WalkerBatch
        Walkers object.
    trial : trial object
        Trial wavefunctioni.

    Returns
    -------
    local_energy : np.ndarray
        Total, one-body and two-body energies.
    """
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
        e1b = 2. * diagGhalfa.dot(trial._rH1a.ravel())
        e1b /= nk
        e1b += hamiltonian.ecore

        ecoul = kpt_isdf_ecoul_rhf_kernel_gpu(hamiltonian.MPQ, trial._rcgtoa, hamiltonian.cgto, ghalfa, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus)

        exxa = 2.0 * kpt_isdf_exx_kernel_gpu(hamiltonian.MPQ, trial._rcgtoa, hamiltonian.cgto, ghalfa, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus)

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

        ecoul = kpt_isdf_ecoul_kernel_gpu(hamiltonian.MPQ, trial._rcgtoa, trial._rcgtob, hamiltonian.cgto, ghalfa, ghalfb, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus)

        exxa = kpt_isdf_exx_kernel_gpu(hamiltonian.MPQ, trial._rcgtoa, hamiltonian.cgto, ghalfa, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus)
        exxb = kpt_isdf_exx_kernel_gpu(hamiltonian.MPQ, trial._rcgtob, hamiltonian.cgto, ghalfb, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus)

        e2b = ecoul + exxa + exxb

    energy = xp.zeros((nwalkers, 3), dtype=numpy.complex128)
    energy[:, 0] = e1b + e2b
    energy[:, 1] = e1b
    energy[:, 2] = e2b

    xp._default_memory_pool.free_all_blocks()
    return energy

def patch_old_energy():
    """Route the live estimator module through the frozen kernels."""
    import ipie.estimators.local_energy_kpt_sd as le

    le.kpt_isdf_ecoul_kernel_gpu = kpt_isdf_ecoul_kernel_gpu
    le.kpt_isdf_ecoul_rhf_kernel_gpu = kpt_isdf_ecoul_rhf_kernel_gpu
    le.kpt_isdf_exx_kernel_gpu = kpt_isdf_exx_kernel_gpu
    le.local_energy_kpt_single_det_uhf_isdf_gpu = local_energy_kpt_single_det_uhf_isdf_gpu

# Same frozen kernel but with the low-k/large-k branch forced by the caller
# instead of the nbsf > 8*nk heuristic (used by the algorithm-regime sweep).
def kpt_isdf_exx_kernel_gpu_forced(MPQ, halfrot_cgtoa, cgto, Ghalfa_batch, kpq_mat, Sset, Qplus, algo):
    nwalker, nk, nocc, _, nbsf = Ghalfa_batch.shape
    nisdf = MPQ.shape[-1]
    if algo == "lowk":
        # lowk algo
        exx = xp.zeros((nwalker,), dtype=xp.complex128)
        nisdf = MPQ.shape[-1]

        w_idx = xp.arange(nwalker)[:, None, None, None, None]  # shape (W,1,1,1,1)
        k_idx = xp.arange(nk)[None, :, None, None, None]  # shape (1,nk,1,1,1)
        i_idx = xp.arange(nocc)[None, None, :, None, None]  # shape (1,1,nocc,1,1)
        kprime_idx = xp.arange(nk)[None, None, None, :, None]  # shape (1,1,1,nk,1)
        p_idx = xp.arange(nbsf)[None, None, None, None, :] # shape (1,1,1,1,nbsf)
   
        intermediate_mem = nisdf * nisdf * nk**2 * nwalker * 3 * 16 / 1024**3
        max_mem = 8.0
        num_chunks = ceil(intermediate_mem / max_mem)
        num_chunk_per_dim = ceil(num_chunks ** 0.5)
        nisdf_per_chunk = ceil(nisdf / num_chunk_per_dim)
        nisdf_left = nisdf
        slices_isdf = []
        for i_chunk in range(num_chunks):
            if nisdf_left == 0:
                break
            nisdf_chunk = min(nisdf_left, nisdf_per_chunk)
            nisdf_left -= nisdf_chunk
            slices_isdf.append(slice(i_chunk * nisdf_per_chunk, i_chunk * nisdf_per_chunk + nisdf_chunk))
        max_dim = max(nisdf_per_chunk, nocc)
        buff1 = xp.empty(nwalker * nk**2 * max_dim**2, dtype=xp.complex128)
        buff2 = xp.empty(nwalker * nk**2 * max_dim**2, dtype=xp.complex128)
        for iq in range(len(Sset)):
            iq_real = Sset[iq]
            ikpq = kpq_mat[iq_real]
            phikr_kpq = cgto[ikpq]
            phiki_kpq = halfrot_cgtoa[ikpq]
            kpq_idx = kpq_mat[k_idx, iq_real]
            kprimepq_idx = kpq_mat[kprime_idx, iq_real]
            G_kpq_kprimepq_chunk = Ghalfa_batch[w_idx, kpq_idx, i_idx, kprimepq_idx, p_idx]
            MPQ_iq = MPQ[iq]
            exx_iq = X_contract_cupy_lowk(halfrot_cgtoa, phikr_kpq, MPQ_iq, phiki_kpq, cgto, Ghalfa_batch, G_kpq_kprimepq_chunk, buff1, buff2, slices_isdf)
            exx -= exx_iq

        for iq in range(len(Sset), len(Sset) + len(Qplus)):
            iq_real = Qplus[iq - len(Sset)]
            ikpq = kpq_mat[iq_real]
            phikr_kpq = cgto[ikpq]
            phiki_kpq = halfrot_cgtoa[ikpq]
            kpq_idx = kpq_mat[k_idx, iq_real]
            kprimepq_idx = kpq_mat[kprime_idx, iq_real]
            G_kpq_kprimepq_chunk = Ghalfa_batch[w_idx, kpq_idx, i_idx, kprimepq_idx, p_idx]
            MPQ_iq = MPQ[iq]
            exx_iq = X_contract_cupy_lowk(halfrot_cgtoa, phikr_kpq, MPQ_iq, phiki_kpq, cgto, Ghalfa_batch, G_kpq_kprimepq_chunk, buff1, buff2, slices_isdf)
            exx -= 2. * exx_iq

    else:
        # large k algo
        w_idx = xp.arange(nwalker)[:, None, None, None, None]  # shape (W,1,1,1,1)
        k_idx = xp.arange(nk)[None, :, None, None, None]  # shape (1,nk,1,1,1)
        i_idx = xp.arange(nocc)[None, None, :, None, None]  # shape (1,1,nocc,1,1)
        kprime_idx = xp.arange(nk)[None, None, None, :, None]  # shape (1,1,1,nk,1)
        p_idx = xp.arange(nbsf)[None, None, None, None, :] # shape (1,1,1,1,nbsf)

        exx = xp.zeros(nwalker, dtype=numpy.complex128)

        if nk < 64:
            intermediate_mem = nwalker * nisdf * nk * nk * nbsf * 6 * 16 / 1024 ** 3
        else:
            intermediate_mem = nwalker * nisdf * nk * nbsf * 6 * 16 / 1024 ** 3
        free_bytes = xp.cuda.Device().mem_info[0]
        free_gb = free_bytes / 1024**3.0
        max_mem = .7 * free_gb
        num_chunks = max(1, ceil(intermediate_mem / max_mem))
        chunk_size = ceil(nwalker / num_chunks)
        nw_left = nwalker
        for i_chunk in range(num_chunks):
            if nw_left == 0:
                break
            n_chunk = min(nw_left, chunk_size)
            nw_left -= n_chunk
            w_sls = xp.arange(nwalker)[i_chunk * chunk_size: i_chunk * chunk_size + n_chunk]
            Ga_chunk = Ghalfa_batch[w_sls]
            w_chunk_idx = xp.arange(n_chunk)[:, None, None, None, None]  # shape (W_chunk,1,1,1,1)

            for iq in range(len(Sset)):
                iq_real = Sset[iq]
                ikpq = kpq_mat[iq_real]
                phikr_kpq = cgto[ikpq]
                phiki_kpq = halfrot_cgtoa[ikpq]
                kpq_idx = kpq_mat[k_idx, iq_real]
                kprimepq_idx = kpq_mat[kprime_idx, iq_real]
                G_kpq_kprimepq_chunk = Ga_chunk[w_chunk_idx, kpq_idx, i_idx, kprimepq_idx, p_idx]
                MPQ_iq = MPQ[iq]
                # exx[w_sls] -= contract('kPi, kPp, PQ, KQj, KQq, wkiKq, wKjkp -> w', halfrot_cgtoa.conj(), phikr_kpq, MPQ_iq, phiki_kpq.conj(), cgto, Ga_chunk, G_kpq_kprimepq_chunk, options=network_opts)
                exx[w_sls] -= contraction_exx(halfrot_cgtoa, phikr_kpq, MPQ_iq, phiki_kpq, cgto, Ga_chunk, G_kpq_kprimepq_chunk, nk, nbsf, nisdf, nocc, n_chunk)
                xp.cuda.get_current_stream().synchronize()
                del G_kpq_kprimepq_chunk

            for iq in range(len(Sset), len(Sset) + len(Qplus)):
                iq_real = Qplus[iq - len(Sset)]
                ikpq = kpq_mat[iq_real]
                phikr_kpq = cgto[ikpq]
                phiki_kpq = halfrot_cgtoa[ikpq]
                kpq_idx = kpq_mat[k_idx, iq_real]
                kprimepq_idx = kpq_mat[kprime_idx, iq_real]
                G_kpq_kprimepq_chunk = Ga_chunk[w_chunk_idx, kpq_idx, i_idx, kprimepq_idx, p_idx]
                MPQ_iq = MPQ[iq]
                exx[w_sls] -= 2. * contraction_exx(halfrot_cgtoa, phikr_kpq, MPQ_iq, phiki_kpq, cgto, Ga_chunk, G_kpq_kprimepq_chunk, nk, nbsf, nisdf, nocc, n_chunk)
                xp.cuda.get_current_stream().synchronize()
                del G_kpq_kprimepq_chunk

    return 0.5 * exx / nk
