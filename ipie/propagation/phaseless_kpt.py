import math
import time

import numpy

import plum

from ipie.config import config
from ipie.hamiltonians.kpt_hamiltonian import KptComplexChol, KptComplexCholSymm, KptISDF
from ipie.hamiltonians.kpt_chunked import KptComplexCholChunked
from ipie.hamiltonians.generic_base import GenericBase
from ipie.propagation.operations import apply_exponential, apply_exponential_batch
from ipie.propagation.phaseless_kpt_base import PhaselessKptBase
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import synchronize
from ipie.walkers.uhf_walkers import UHFWalkers
from numba import jit
from ipie.utils.backend import get_device_memory
from ipie.propagation.kernels import call_kernel_VHS_construction1, call_kernel_VHS_construction2
from math import ceil

@jit(nopython=True, fastmath=True)
def construct_VHS_kernel_symm(chol, sqrt_dt, xshifted, nk, nbasis, nwalkers, ikpq_mat, Sset, Qplus):

    VHS = numpy.zeros((nk, nk, nwalkers, nbasis * nbasis), dtype=numpy.complex128)
    for iq in range(len(Sset)):
        iq_real = Sset[iq]
        for ik in range(nk):
            ikpq = ikpq_mat[iq_real, ik]
            x_iq = .5 * (1j * xshifted[0, :, :, iq] + xshifted[1, :, :, iq])
            xconj_iq = .5 * (1j * xshifted[0, :, :, iq] - xshifted[1, :, :, iq])
            cholkq = chol[:, ik, :, iq, :].copy()
            cholkq = cholkq.reshape(-1, nbasis*nbasis)
            VHS[ik, ikpq] += sqrt_dt * x_iq @ cholkq
            XL = sqrt_dt * xconj_iq @ cholkq.conj()
            XL = XL.reshape(nwalkers, nbasis, nbasis).transpose(0, 2, 1).copy()
            VHS[ikpq, ik] += XL.reshape(nwalkers, nbasis * nbasis)

    for iq in range(len(Sset), len(Sset) + len(Qplus)):
        iq_real = Qplus[iq - len(Sset)]
        for ik in range(nk):
            ikpq = ikpq_mat[iq_real, ik]
            x_iq = .5 * (1j * xshifted[0, :, :, iq] + xshifted[1, :, :, iq])
            xconj_iq = .5 * (1j * xshifted[0, :, :, iq] - xshifted[1, :, :, iq])
            cholkq = chol[:, ik, :, iq, :].copy()
            cholkq = cholkq.reshape(-1, nbasis*nbasis)
            VHS[ik, ikpq] += math.sqrt(2) * sqrt_dt * x_iq @ cholkq
            XL = sqrt_dt * xconj_iq @ cholkq.conj()
            XL = XL.reshape(nwalkers, nbasis, nbasis).transpose(0, 2, 1).copy()
            VHS[ikpq, ik] += math.sqrt(2) * XL.reshape(nwalkers, nbasis * nbasis)
    VHS = VHS.reshape(nk, nk, nwalkers, nbasis, nbasis).transpose(2, 0, 3, 1, 4).copy()
    VHS = VHS.reshape(nwalkers, nk * nbasis, nk * nbasis)
    return VHS

def construct_VHS_symm_gpu(chol, sqrt_dt, xshifted, nk, nbasis, nwalkers, ikpq_mat, Sset, Qplus):
    VHS = xp.zeros((nwalkers, nk, nbasis, nk, nbasis), dtype=xp.complex128)
    x= .5 * (1j * xshifted[0] + xshifted[1])
    xconj = .5 * (1j * xshifted[0] - xshifted[1])
    unique_qs = xp.concatenate((Sset, Qplus))
    # print("ikpq_S", ikpq_S)
    idx_lenS = xp.arange(len(Sset))
    idx_lenQ = xp.arange(len(Qplus)) + len(Sset)

    xS = sqrt_dt * x[:, :, idx_lenS]
    xQ = xp.sqrt(2) * sqrt_dt * x[:, :, idx_lenQ]
    xconjS = sqrt_dt * xconj[:, :, idx_lenS]
    xconjQ = xp.sqrt(2) * sqrt_dt * xconj[:, :, idx_lenQ]

    xtot = xp.concatenate((xS, xQ), axis=-1)
    xconjtot = xp.concatenate((xconjS, xconjQ), axis=-1)

    kpq_mat = ikpq_mat[unique_qs]

    naux = chol.shape[0]

    call_kernel_VHS_construction1(chol, xtot, naux, nk, nbasis, nwalkers, kpq_mat, VHS)
    call_kernel_VHS_construction2(chol, xconjtot, naux, nk, nbasis, nwalkers, kpq_mat, VHS)

    VHS = VHS.reshape(nwalkers, nk * nbasis, nk * nbasis)
    return VHS

def construct_VHS_cuquantum(chol, sqrt_dt, xshifted, nk, nbasis, nwalkers, ikpq_mat, Sset, Qplus):
    raise NotImplementedError("CuQuantum VHS construction is not available on AMD GPUs.")

class PhaselessKptChol(PhaselessKptBase):
    """A class for performing phaseless propagation with k-point Hamiltonian."""

    def __init__(self, time_step, ebound_const = 2.0, fbbound = 1.0, exp_nmax=6, verbose=False):
        super().__init__(time_step, ebound_const = ebound_const, fbbound = fbbound, verbose=verbose)
        self.exp_nmax = exp_nmax

    @plum.dispatch
    def apply_VHS(self, walkers: UHFWalkers, hamiltonian: GenericBase, xshifted: xp.ndarray):
        start_time = time.time()
        VHS = self.construct_VHS(hamiltonian, xshifted)
        synchronize()
        self.timer.tvhs += time.time() - start_time
        assert len(VHS.shape) == 3  # shape = nwalkers, nk * nbasis, nk * nbasis
        start_time = time.time()
        if config.get_option("use_gpu"):
            walkers.phia = apply_exponential_batch(walkers.phia, VHS, self.exp_nmax)
            if walkers.ndown > 0 and not walkers.rhf:
                walkers.phib = apply_exponential_batch(walkers.phib, VHS, self.exp_nmax)

        else:
            for iw in range(walkers.nwalkers):
                # 2.b Apply two-body
                walkers.phia[iw] = apply_exponential(walkers.phia[iw], VHS[iw], self.exp_nmax)
                if walkers.ndown > 0 and not walkers.rhf:
                    walkers.phib[iw] = apply_exponential(walkers.phib[iw], VHS[iw], self.exp_nmax)
        synchronize()
        self.timer.tgemm += time.time() - start_time

    @plum.dispatch.abstract
    def construct_VHS(self, hamiltonian: GenericBase, xshifted: xp.ndarray) -> xp.ndarray:
        print("JOONHO here abstract function for construct VHS")
        "abstract function for construct VHS"

    # Any class inherited from PhaselessGeneric should override this method.
    @plum.dispatch
    def construct_VHS(self, hamiltonian: KptComplexChol, xshifted: xp.ndarray) -> xp.ndarray:
        """
        Construct the VHS matrix for phaseless propagation.
        
        xshifted: [2, nwalkers, naux, nk]
        """
        nwalkers = xshifted.shape[1]
        VHS = numpy.zeros((nwalkers, hamiltonian.nk, hamiltonian.nbasis, hamiltonian.nk, hamiltonian.nbasis), dtype=numpy.complex128)

        for iq in range(hamiltonian.nk):
            for ik in range(hamiltonian.nk):
                ikpq = hamiltonian.ikpq_mat[ik, iq]
                imq = hamiltonian.imq_vec[iq]
                xtildepiq = xshifted[0, :, :, iq] + xshifted[0, :, :, imq]
                xtildemiq = xshifted[1, :, :, iq] - xshifted[1, :, :, imq]
                xvhsiq = (1j * xtildepiq + xtildemiq) / 2
                VHS[:, ik, :, ikpq, :] = self.sqrt_dt * numpy.einsum('wx, xpr -> wpr', xvhsiq, hamiltonian.chol[:, ik, :, iq, :])
        VHS = VHS.reshape(nwalkers, hamiltonian.nk * hamiltonian.nbasis, hamiltonian.nk * hamiltonian.nbasis)
        if config.get_option("use_gpu"):
            raise NotImplementedError
        return VHS
    
    @plum.dispatch
    def construct_VHS(self, hamiltonian: KptComplexCholSymm, xshifted: xp.ndarray) -> xp.ndarray:
        """
        Construct the VHS matrix for phaseless propagation.
        
        xshifted: [2, nwalkers, naux, unique_nk]
        """
        nwalkers = xshifted.shape[1]
        if config.get_option("use_gpu"):
            raise NotImplementedError
        else:
            VHS = construct_VHS_kernel_symm(hamiltonian.chol, self.sqrt_dt, xshifted, hamiltonian.nk, hamiltonian.nbasis, nwalkers, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus)
        
        return VHS

class PhaselessKptCholChunked(PhaselessKptChol):
    """A class for performing phaseless propagation with complex hamiltonian with k point symmetry."""

    def __init__(self, time_step, ebound_const = 2.0, fbbound = 1.0, exp_nmax=6, verbose=False):
        super().__init__(time_step, ebound_const = ebound_const, fbbound = fbbound, verbose=verbose)

    def build(self, hamiltonian, trial=None, walkers=None, mpi_handler=None, verbose=False):
        super().build(hamiltonian, trial, walkers, mpi_handler, verbose)
        self.mpi_handler = mpi_handler

    @plum.dispatch
    def construct_VHS(
        self, hamiltonian: KptComplexCholChunked, xshifted: xp.ndarray
    ) -> xp.ndarray:
        assert hamiltonian.chunked
        nwalkers = xshifted.shape[1]
        
        xshifted_send = xshifted.copy()
        xshifted_recv = xp.zeros_like(xshifted)

        idxs = hamiltonian.chol_idxs_chunk
        chol_chunk = hamiltonian.chol_chunk.reshape(-1, hamiltonian.nk, hamiltonian.nbasis, hamiltonian.unique_nk, hamiltonian.nbasis)
        if config.get_option("use_gpu"):
            if hamiltonian.nk > 256:
                # nk^2 > 65536, use cuquantum instead
                VHS_send = construct_VHS_cuquantum(chol_chunk, self.sqrt_dt, xshifted[:, :, idxs, :], hamiltonian.nk, hamiltonian.nbasis, nwalkers, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus)
            else:
                VHS_send = construct_VHS_symm_gpu(chol_chunk, self.sqrt_dt, xshifted[:, :, idxs, :], hamiltonian.nk, hamiltonian.nbasis, nwalkers, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus)
        else:
            VHS_send = construct_VHS_kernel_symm(chol_chunk, self.sqrt_dt, xshifted[:, :, idxs, :], hamiltonian.nk, hamiltonian.nbasis, nwalkers, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus)
        VHS_recv = xp.zeros_like(VHS_send)

        srank = self.mpi_handler.scomm.rank
        sender = numpy.where(self.mpi_handler.receivers == srank)[0]

        for _ in range(self.mpi_handler.ssize - 1):
            synchronize()
            self.mpi_handler.scomm.Isend(
                xshifted_send, dest=self.mpi_handler.receivers[srank], tag=1
            )
            self.mpi_handler.scomm.Isend(VHS_send, dest=self.mpi_handler.receivers[srank], tag=2)

            req1 = self.mpi_handler.scomm.Irecv(xshifted_recv, source=sender, tag=1)
            req2 = self.mpi_handler.scomm.Irecv(VHS_recv, source=sender, tag=2)
            req1.wait()
            req2.wait()

            self.mpi_handler.scomm.barrier()
            if config.get_option("use_gpu"):
                if hamiltonian.nk > 256:
                    VHS_send = construct_VHS_cuquantum(chol_chunk, self.sqrt_dt, xshifted_recv[:, :, idxs, :], hamiltonian.nk, hamiltonian.nbasis, nwalkers, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus)
                else:
                    VHS_send = construct_VHS_symm_gpu(chol_chunk, self.sqrt_dt, xshifted_recv[:, :, idxs, :], hamiltonian.nk, hamiltonian.nbasis, nwalkers, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus)
            else:
                VHS_send = construct_VHS_kernel_symm(chol_chunk, self.sqrt_dt, xshifted_recv[:, :, idxs, :], hamiltonian.nk, hamiltonian.nbasis, nwalkers, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus)
            VHS_send += VHS_recv

            xshifted_send = xshifted_recv.copy()

        synchronize()
        self.mpi_handler.scomm.Isend(VHS_send, dest=self.mpi_handler.receivers[srank], tag=1)
        req = self.mpi_handler.scomm.Irecv(VHS_recv, source=sender, tag=1)
        req.wait()
        self.mpi_handler.scomm.barrier()

        synchronize()
        # if config.get_option("use_gpu"):
        #     xp._default_memory_pool.free_all_blocks()
        return VHS_recv

class PhaselessKptISDF(PhaselessKptBase):
    """A class for performing phaseless propagation with k-point Hamiltonian with ERI approximated by ISDF. Here we do not save VHS to save memory."""

    def __init__(self, time_step, ebound_const = 2.0, fbbound = 1.0, exp_nmax=6, verbose=False):
        super().__init__(time_step, ebound_const = ebound_const, fbbound = fbbound, verbose=verbose)
        self.exp_nmax = exp_nmax

    @plum.dispatch
    def apply_VHS(self, walkers: UHFWalkers, hamiltonian: GenericBase, xshifted: xp.ndarray):
        if config.get_option("use_gpu"):
            #determine whether to use bigmem algo or not
            nbsf = hamiltonian.nbasis
            nk = hamiltonian.nk
            nwalkers = walkers.nwalkers
            occ_ratio = walkers.nup / walkers.nbasis if walkers.rhf else (walkers.nup + walkers.ndown) / (walkers.nbasis)
            mem_vhs = 2* nwalkers * nbsf**2 * nk**2 * 16
            if mem_vhs > 0.45 * xp.cuda.Device().mem_info[0] or occ_ratio < 0.2:
                start_time = time.time()
                Lx, Lconjx = self.contract_cholM_xshifted(hamiltonian, xshifted)
                self.timer.tvhs += time.time() - start_time
                assert len(Lx.shape) == 3 # nwalkers, nq, nisdf

                start_time = time.time()
                if walkers.ndown > 0 and not walkers.rhf:
                    # the contraction is independent along the occupied index, so both
                    # spins can share one Taylor expansion with larger GEMMs
                    nocca = walkers.phia.shape[-1] // nk
                    noccb = walkers.phib.shape[-1] // nk
                    Temp = xp.concatenate(
                        (
                            walkers.phia.reshape(nwalkers, nk * nbsf, nk, nocca),
                            walkers.phib.reshape(nwalkers, nk * nbsf, nk, noccb),
                        ),
                        axis=3,
                    ).reshape(nwalkers, nk * nbsf, nk * (nocca + noccb))
                    for n in range(1, self.exp_nmax + 1):
                        Temp = apply_VHS_to_phi_batch(hamiltonian.cgto, Lx, Lconjx, Temp, hamiltonian.ikpq_mat, hamiltonian.ikmq_mat, hamiltonian.unique_k) / n  # matmul use much less GPU memory than einsum
                        Temp_split = Temp.reshape(nwalkers, nk * nbsf, nk, nocca + noccb)
                        walkers.phia += Temp_split[:, :, :, :nocca].reshape(nwalkers, nk * nbsf, nk * nocca)
                        walkers.phib += Temp_split[:, :, :, nocca:].reshape(nwalkers, nk * nbsf, nk * noccb)
                    del Temp, Temp_split
                else:
                    Temp = xp.zeros(walkers.phia.shape, dtype=walkers.phia.dtype)
                    xp.copyto(Temp, walkers.phia)
                    for n in range(1, self.exp_nmax + 1):
                        Temp = apply_VHS_to_phi_batch(hamiltonian.cgto, Lx, Lconjx, Temp, hamiltonian.ikpq_mat, hamiltonian.ikmq_mat, hamiltonian.unique_k) / n  # matmul use much less GPU memory than einsum
                        walkers.phia += Temp
                    del Temp
            else:
                start_time = time.time()
                Lx, Lconjx = self.contract_cholM_xshifted(hamiltonian, xshifted)
                VHS = construct_VHS_batch(hamiltonian.cgto, Lx, Lconjx, hamiltonian.ikpq_mat, hamiltonian.ikmq_mat, hamiltonian.unique_k)
                synchronize()
                self.timer.tvhs += time.time() - start_time
                assert len(Lx.shape) == 3 # nwalkers, nq, nisdf

                start_time = time.time()
                walkers.phia = apply_exponential_batch(walkers.phia, VHS, self.exp_nmax)
                if walkers.ndown > 0 and not walkers.rhf:
                    walkers.phib = apply_exponential_batch(walkers.phib, VHS, self.exp_nmax)
            synchronize()
            self.timer.tgemm += time.time() - start_time                
        else:
            raise NotImplementedError
        

    def contract_cholM_xshifted(self, hamiltonian, xshifted):
        cholM = hamiltonian.cholM # q, P, gamma
        x = .5 * (1j * xshifted[0] + xshifted[1]) # w, gamma, q
        xconj = .5 * (1j * xshifted[0] - xshifted[1]) # w, gamma, q
        unique_qs = xp.concatenate((hamiltonian.Sset, hamiltonian.Qplus))
        # print("ikpq_S", ikpq_S)
        idx_lenS = xp.arange(len(hamiltonian.Sset))
        idx_lenQ = xp.arange(len(hamiltonian.Qplus)) + len(hamiltonian.Sset)

        xS = self.sqrt_dt * x[:, :, idx_lenS]
        xQ = xp.sqrt(2) * self.sqrt_dt * x[:, :, idx_lenQ]
        xconjS = self.sqrt_dt * xconj[:, :, idx_lenS]
        xconjQ = xp.sqrt(2) * self.sqrt_dt * xconj[:, :, idx_lenQ]

        xtot = xp.concatenate((xS, xQ), axis=-1)
        xconjtot = xp.concatenate((xconjS, xconjQ), axis=-1)
        # cholMx = contract('qPg, wgq -> wqP', cholM, xtot, options=network_opts)
        xtot = xtot.transpose(2, 1, 0) # q, g, w
        cholMx = xp.matmul(cholM, xtot) # q, P, w
        cholMx = cholMx.transpose(2, 0, 1) # w, q, P
        # cholMxconj = contract('qPg, wgq -> wqP', cholM.conj(), xconjtot, options=network_opts)
        xconjtot = xconjtot.transpose(2, 1, 0).conj()
        cholMxconj = xp.conj(xp.matmul(cholM, xconjtot))
        cholMxconj = cholMxconj.transpose(2, 0, 1)
        return cholMx, cholMxconj

def construct_full_Lx_batch(Lx, kpq_mat, unique_qs):
    """
    Construct full Lx from Lx using advanced indexing. Lx: [nwalkers, nk, nisdf]. 
    """
    nk = kpq_mat.shape[1]
    nq = len(unique_qs)
    nwalker = Lx.shape[0]
    nisdf = Lx.shape[-1]  # Lx: (nwalker, nq, nisdf)

    fullLconjx = xp.zeros((nwalker, nk, nk, nisdf), dtype=xp.complex128)
    q_idx = unique_qs[:, None]          # shape (nq, 1)
    ik_idx = xp.arange(nk)[None, :] # shape (1, nk)

    row_idx = kpq_mat[q_idx, ik_idx]
    row_idx = xp.broadcast_to(row_idx, (nq, nk))
    col_idx = xp.broadcast_to(ik_idx, (nq, nk))
    data = Lx[:, :, None, :]  # shape (nw, nq, 1, nisdf)
    data = xp.broadcast_to(data, (nwalker, nq, nk, nisdf))
    fullLconjx[:, row_idx, col_idx, :] = data

    return fullLconjx


def construct_full_l_batch_for_gemm(Lx, Lconjx, kpq_mat, kmq_mat, unique_qs):
    """
    Construct the dense k-point coupling matrix in the layout consumed by GEMM.
    """
    nwalkers = Lx.shape[0]
    nk = kpq_mat.shape[1]
    pchunk = Lx.shape[-1]
    full_l = xp.zeros((nwalkers, pchunk, nk, nk), dtype=Lx.dtype)

    q_idx = unique_qs[:, None]
    k_idx = xp.arange(nk)[None, :]
    row_idx = xp.broadcast_to(k_idx, (len(unique_qs), nk))
    data = xp.broadcast_to(
        Lx.transpose(0, 2, 1)[:, :, :, None], (nwalkers, pchunk, len(unique_qs), nk)
    )
    full_l[:, :, row_idx, kpq_mat[q_idx, k_idx]] = data

    data_conj = xp.broadcast_to(
        Lconjx.transpose(0, 2, 1)[:, :, :, None],
        (nwalkers, pchunk, len(unique_qs), nk),
    )
    full_l[:, :, row_idx, kmq_mat[q_idx, k_idx]] += data_conj
    return full_l


def contract_lowmem_vhs_walkers_from_l_batch(
    l_batch, cgto_slice, phi_for_cgto, nw, nk, nisdf, nbsf, nocc, max_mem=4.0
):
    # besides the two buff regions, each iteration materializes two transpose
    # copies of the same chunk size, so the peak is ~4 chunk-sized buffers
    intermediate_mem = nw * nk**2 * nisdf * nocc * 16 * 4 /1024**3
    num_chunks = max(1, ceil(intermediate_mem / max_mem))
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
    result = xp.zeros((nw, nk, nbsf, nk, nocc), dtype=xp.complex128)
    for i_sls in slices_isdf:
        nisdf_chunk = i_sls.stop - i_sls.start
        cgto_slice_P = cgto_slice[:, i_sls, :]
        size_cgtophi = nk**2 * nisdf_chunk * nw * nocc
        cgtophi = buff[:size_cgtophi].reshape(nk, nisdf_chunk, nw * nk * nocc)
        xp.matmul(cgto_slice_P, phi_for_cgto, out=cgtophi)
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
        
def contract_lowmem_vhs_walkers(fullLpLconjx, cgto_slice, phi_reshape, nw, nk, nisdf, nbsf, nocc):
    # we need to slice over the P index
    intermediate_mem = nw * nk**2 * nisdf * nocc * 16 * 2 /1024**3
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
    phi_reshape = phi_reshape.transpose(1, 2, 0, 3, 4).reshape(nk, nbsf, nw * nk * nocc) # K, r, wQi
    result = xp.zeros((nw, nk, nbsf, nk, nocc), dtype=xp.complex128)  # w, k, p, Q, i
    for i_sls in slices_isdf:
        nisdf_chunk = i_sls.stop - i_sls.start
        cgto_slice_P = cgto_slice[:, i_sls, :]
        size_cgtophi = nk**2 * nisdf_chunk * nw * nocc
        cgtophi = buff[:size_cgtophi].reshape(nk, nisdf_chunk, nw * nk * nocc)
        xp.matmul(cgto_slice_P, phi_reshape, out=cgtophi)  # K, P, r -> K, P, wQi
        fullLpLconjx_slice = fullLpLconjx[:, :, :, i_sls]  # w, K, k, P
        fullLpLconjx_slice = fullLpLconjx_slice.transpose(0, 3, 2, 1).reshape(nw * nisdf_chunk, nk, nk) # wP, k, K
        cgtophi = cgtophi.reshape(nk, nisdf_chunk, nw, nk, nocc).transpose(2, 1, 0, 3, 4).reshape(nw * nisdf_chunk, nk, nk * nocc) # wP, K, Qi
        size_Lxcgtophi = nk**2 * nisdf_chunk * nw * nocc
        Lx_cgtophi = buff[size_cgtophi:size_cgtophi + size_Lxcgtophi].reshape(nw * nisdf_chunk, nk, nk * nocc)
        xp.matmul(fullLpLconjx_slice, cgtophi, out=Lx_cgtophi) # wP, k, Qi
        Lx_cgtophi = Lx_cgtophi.reshape(nw, nisdf_chunk, nk, nk, nocc).transpose(2, 0, 3, 4, 1).reshape(nk, nw * nk * nocc, nisdf_chunk)  # k, wQi, P
        size_result = nw * nk**2 * nocc * nbsf
        temp = buff[:size_result].reshape(nk, nw * nk * nocc, nbsf)
        xp.matmul(Lx_cgtophi, cgto_slice_P.conj(), out=temp)
        result += temp.reshape(nk, nw, nk, nocc, nbsf).transpose(1, 0, 4, 2, 3)  # w, k, p, Q, i
    return result

def build_VHS_lowmem(fullLpLconjx, cgto_slice, nw, nk, nisdf, nbsf, max_mem = 4.0):
    intermediate_mem = nw * nk**2 * nisdf * nbsf * 16 * 2 /1024**3
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
    vhs = xp.zeros((nw, nk, nbsf, nk, nbsf), dtype=xp.complex128)  # w, k, p, Q
    buff = xp.empty((nk, nw * nbsf * nk, nbsf), dtype=xp.complex128) 
    
    for i_sls in slices_isdf:
        nisdf_chunk = i_sls.stop - i_sls.start
        fullLpLconjx_slice = fullLpLconjx[:, :, :, i_sls] # wKkP
        # K, w,k,P; k, p, P -> K, w, k, p, P
        # 1. k, wK, P; k, P, p -> k, wK, P, p
        # 2. k, wK, P, p; K, P, r -> k, wK, p, r
        # K, wkp, P; K, P, r -> K, wkp, r
        fullLpLconjx_slice = fullLpLconjx_slice.transpose(1, 0, 2, 3) # K, w, k, P
        cgto_slice_P = cgto_slice[:, i_sls, :] #k, P, p
        cgto_slice_T = cgto_slice_P.transpose(0, 2, 1)
        LpLconjx_cgto = fullLpLconjx_slice[:, :, :, xp.newaxis, :] * cgto_slice_T[xp.newaxis, xp.newaxis, :, :, :].conj()  # K, w, k, p, P
        LpLconjx_cgto = LpLconjx_cgto.reshape(nk, nw * nk * nbsf, nisdf_chunk)
        buff = buff.reshape(nk, nw * nbsf * nk, nbsf)
        xp.matmul(LpLconjx_cgto, cgto_slice_P, out=buff)  # K, wkp, r
        buff = buff.reshape(nk, nw, nk, nbsf, nbsf).transpose(1, 2, 3, 0, 4)  # w, k, p, K, r
        vhs += buff  # w, k, p, K, r

    return vhs

def apply_VHS_to_phi_batch(cgto, Lx, Lconjx, phi, kpq_mat, kmq_mat, unique_qs):
    """
    Apply VHS to phi in batch.
    """
    nwalkers = Lx.shape[0]
    nk = kpq_mat.shape[1]
    nbsf = cgto.shape[2]
    nisdf = Lx.shape[-1]
    nknocc = phi.shape[-1]
    nocc = nknocc // nk
    outphi = xp.zeros_like(phi)
    # the outer loop only materializes l_batch, O(nw*nk^2) per ISDF point; the
    # inner kernel bounds its own intermediates, so give l_batch a quarter of
    # the memory allowance and keep outer chunks as large as possible
    max_mem_bytes = int(0.3 * xp.cuda.Device().mem_info[0])
    bytes_per_isdf = nwalkers * nk * nk * 16
    nisdf_chunk_size = max(1, min(nisdf, max_mem_bytes // (4 * bytes_per_isdf)))
    num_nisdf_chunks = math.ceil(nisdf / nisdf_chunk_size)
    # let the inner kernel use real headroom instead of a fixed 4 GB so its
    # chunks stay large when nocc and nk^2 grow
    inner_max_mem = max(4.0, 0.1 * xp.cuda.Device().mem_info[0] / 1024**3)
    nisdf_left = nisdf
    phi_for_cgto = phi.reshape(nwalkers, nk, nbsf, nk, nocc).transpose(1, 2, 0, 3, 4).reshape(nk, nbsf, nwalkers * nk * nocc)
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
        cgto_slice = cgto[:, i * nisdf_chunk_size: i * nisdf_chunk_size + nisdf_chunk, :]
        # out = contract('wKkP, kPp, KPr, wKrQi -> wkpQi', fullLpLconjx, cgto_slice.conj(), cgto_slice, phi_reshape, options=network_opts)
        out = contract_lowmem_vhs_walkers_from_l_batch(l_batch, cgto_slice, phi_for_cgto, nwalkers, nk, nisdf_chunk, nbsf, nocc, max_mem=inner_max_mem)

        del l_batch, cgto_slice
        outphi += out.reshape(nwalkers, nk * nbsf, -1)
        del out
    del phi_for_cgto
    xp._default_memory_pool.free_all_blocks()
    return outphi

def construct_VHS_batch(cgto, Lx, Lconjx, kpq_mat, kmq_mat, unique_qs):
    """
    Apply VHS to phi in batch.
    """
    nwalkers = Lx.shape[0]
    nk = kpq_mat.shape[1]
    nbsf = cgto.shape[2]
    nisdf = Lx.shape[-1]
    outvhs = xp.zeros((nwalkers, nk * nbsf, nk * nbsf), dtype=xp.complex128)
    # calculate intermediate array memory
    mem_cost = nwalkers * nk * nk * nisdf * nbsf * 16 * 4/ 1024**3
    # max_mem = 80 percent of available memory
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
        full_Lx = construct_full_Lx_batch(Lx_chunk, kpq_mat, unique_qs)
        full_Lconjx = construct_full_Lx_batch(Lconjx_chunk, kmq_mat, unique_qs)
        fullLpLconjx = full_Lx + full_Lconjx
        cgto_slice = cgto[:, i * nisdf_chunk_size: i * nisdf_chunk_size + nisdf_chunk, :]
        # out = contract('wKkP, kPp, KPr -> wkpKr', fullLpLconjx, cgto_slice.conj(), cgto_slice, options=network_opts)
        out = build_VHS_lowmem(fullLpLconjx, cgto_slice, nwalkers, nk, nisdf_chunk, nbsf)
        del full_Lx, full_Lconjx, fullLpLconjx, cgto_slice
        outvhs += out.reshape(nwalkers, nk * nbsf, nk * nbsf)
        del out
    xp._default_memory_pool.free_all_blocks()
    return outvhs

Phaseless = {"cholesky": PhaselessKptChol, "isdf": PhaselessKptISDF, "cholchunked": PhaselessKptCholChunked}
