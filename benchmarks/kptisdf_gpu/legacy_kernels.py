"""Reference implementations of the KptISDF VHS path as of commit cbaa246.

These are verbatim copies of the code before commit e3487cc ("Harden
GEMM-based KptISDF propagation kernels") so the benchmark can measure the
effect of that commit in isolation:
  - phi transpose copy done inside the inner kernel, once per outer ISDF chunk
  - chunking estimate counting 2 buffers with a hardcoded 4 GB budget
  - alpha and beta propagated in separate Taylor expansions
"""

import math
import time
from math import ceil

from ipie.config import config
from ipie.propagation.operations import apply_exponential_batch
from ipie.propagation.phaseless_kpt import construct_full_l_batch_for_gemm, construct_VHS_batch
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import synchronize


def contract_lowmem_vhs_walkers_from_l_batch_old(
    l_batch, cgto_slice, phi_reshape, nw, nk, nisdf, nbsf, nocc
):
    intermediate_mem = nw * nk**2 * nisdf * nocc * 16 * 2 / 1024**3
    max_mem = 4.0
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
    max_dim = max(nisdf_per_chunk, nbsf)
    buff = xp.empty(2 * nw * nk**2 * nocc * max_dim, dtype=xp.complex128)
    phi_reshape = phi_reshape.transpose(1, 2, 0, 3, 4).reshape(nk, nbsf, nw * nk * nocc)
    result = xp.zeros((nw, nk, nbsf, nk, nocc), dtype=xp.complex128)
    for i_sls in slices_isdf:
        nisdf_chunk = i_sls.stop - i_sls.start
        cgto_slice_P = cgto_slice[:, i_sls, :]
        size_cgtophi = nk**2 * nisdf_chunk * nw * nocc
        cgtophi = buff[:size_cgtophi].reshape(nk, nisdf_chunk, nw * nk * nocc)
        xp.matmul(cgto_slice_P, phi_reshape, out=cgtophi)
        l_batch_slice = l_batch[:, i_sls].reshape(nw * nisdf_chunk, nk, nk)
        cgtophi = cgtophi.reshape(nk, nisdf_chunk, nw, nk, nocc).transpose(2, 1, 0, 3, 4).reshape(nw * nisdf_chunk, nk, nk * nocc)
        size_Lxcgtophi = nk**2 * nisdf_chunk * nw * nocc
        Lx_cgtophi = buff[size_cgtophi:size_cgtophi + size_Lxcgtophi].reshape(nw * nisdf_chunk, nk, nk * nocc)
        xp.matmul(l_batch_slice, cgtophi, out=Lx_cgtophi)
        Lx_cgtophi = Lx_cgtophi.reshape(nw, nisdf_chunk, nk, nk, nocc).transpose(2, 0, 3, 4, 1).reshape(nk, nw * nk * nocc, nisdf_chunk)
        size_result = nw * nk**2 * nocc * nbsf
        temp = buff[:size_result].reshape(nk, nw * nk * nocc, nbsf)
        xp.matmul(Lx_cgtophi, cgto_slice_P.conj(), out=temp)
        result += temp.reshape(nk, nw, nk, nocc, nbsf).transpose(1, 0, 4, 2, 3)
    return result


def apply_VHS_to_phi_batch_old(cgto, Lx, Lconjx, phi, kpq_mat, kmq_mat, unique_qs):
    nwalkers = Lx.shape[0]
    nk = kpq_mat.shape[1]
    nbsf = cgto.shape[2]
    nisdf = Lx.shape[-1]
    nknocc = phi.shape[-1]
    nocc = nknocc // nk
    outphi = xp.zeros_like(phi)
    mem_cost = nwalkers * nisdf * nisdf * nknocc * 16 * 4 / 1024**3
    max_mem = 0.3 * xp.cuda.Device().mem_info[0] / 1024**3
    num_nisdf_chunks = max(1, math.ceil(mem_cost / max_mem))
    nisdf_chunk_size = math.ceil(nisdf / num_nisdf_chunks)
    nisdf_left = nisdf
    for i in range(num_nisdf_chunks):
        if nisdf_left == 0:
            break
        nisdf_chunk = min(nisdf_chunk_size, nisdf_left)
        nisdf_left -= nisdf_chunk
        Lx_chunk = Lx[:, :, i * nisdf_chunk_size: i * nisdf_chunk_size + nisdf_chunk]
        Lconjx_chunk = Lconjx[:, :, i * nisdf_chunk_size: i * nisdf_chunk_size + nisdf_chunk]
        l_batch = construct_full_l_batch_for_gemm(
            Lx_chunk, Lconjx_chunk, kpq_mat, kmq_mat, unique_qs
        )
        phi_reshape = phi.reshape(nwalkers, nk, nbsf, nk, -1)
        cgto_slice = cgto[:, i * nisdf_chunk_size: i * nisdf_chunk_size + nisdf_chunk, :]
        out = contract_lowmem_vhs_walkers_from_l_batch_old(
            l_batch, cgto_slice, phi_reshape, nwalkers, nk, nisdf_chunk, nbsf, nocc
        )
        del l_batch, cgto_slice, phi_reshape
        outphi += out.reshape(nwalkers, nk * nbsf, -1)
        del out
    xp._default_memory_pool.free_all_blocks()
    return outphi


def apply_VHS_old(self, walkers, hamiltonian, xshifted):
    """PhaselessKptISDF.apply_VHS as of cbaa246 (spin-separate Taylor loops)."""
    if config.get_option("use_gpu"):
        nbsf = hamiltonian.nbasis
        nk = hamiltonian.nk
        nwalkers = walkers.nwalkers
        occ_ratio = walkers.nup / walkers.nbasis if walkers.rhf else (walkers.nup + walkers.ndown) / (walkers.nbasis)
        mem_vhs = 2 * nwalkers * nbsf**2 * nk**2 * 16
        if mem_vhs > 0.45 * xp.cuda.Device().mem_info[0] or occ_ratio < 0.2:
            start_time = time.time()
            Lx, Lconjx = self.contract_cholM_xshifted(hamiltonian, xshifted)
            self.timer.tvhs += time.time() - start_time
            assert len(Lx.shape) == 3

            start_time = time.time()
            Temp = xp.zeros(walkers.phia.shape, dtype=walkers.phia.dtype)
            xp.copyto(Temp, walkers.phia)
            for n in range(1, self.exp_nmax + 1):
                Temp = apply_VHS_to_phi_batch_old(hamiltonian.cgto, Lx, Lconjx, Temp, hamiltonian.ikpq_mat, hamiltonian.ikmq_mat, hamiltonian.unique_k) / n
                walkers.phia += Temp
            del Temp
            if walkers.ndown > 0 and not walkers.rhf:
                Temp = xp.zeros(walkers.phib.shape, dtype=walkers.phib.dtype)
                xp.copyto(Temp, walkers.phib)
                for n in range(1, self.exp_nmax + 1):
                    Temp = apply_VHS_to_phi_batch_old(hamiltonian.cgto, Lx, Lconjx, Temp, hamiltonian.ikpq_mat, hamiltonian.ikmq_mat, hamiltonian.unique_k) / n
                    walkers.phib += Temp
                del Temp
        else:
            start_time = time.time()
            Lx, Lconjx = self.contract_cholM_xshifted(hamiltonian, xshifted)
            VHS = construct_VHS_batch(hamiltonian.cgto, Lx, Lconjx, hamiltonian.ikpq_mat, hamiltonian.ikmq_mat, hamiltonian.unique_k)
            synchronize()
            self.timer.tvhs += time.time() - start_time
            assert len(Lx.shape) == 3

            start_time = time.time()
            walkers.phia = apply_exponential_batch(walkers.phia, VHS, self.exp_nmax)
            if walkers.ndown > 0 and not walkers.rhf:
                walkers.phib = apply_exponential_batch(walkers.phib, VHS, self.exp_nmax)
        synchronize()
        self.timer.tgemm += time.time() - start_time
    else:
        raise NotImplementedError


def patch_old_propagation():
    """Monkeypatch the cbaa246 VHS path into the running ipie modules."""
    import ipie.propagation.phaseless_kpt as pk

    pk.apply_VHS_to_phi_batch = apply_VHS_to_phi_batch_old
    pk.PhaselessKptISDF.apply_VHS = apply_VHS_old
