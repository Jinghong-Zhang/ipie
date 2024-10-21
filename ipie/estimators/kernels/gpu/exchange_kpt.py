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

try:
    # pylint: disable=import-error
    import cupy as cp
    import numba
    from numba import cuda, complex128
except ModuleNotFoundError:
    pass

_block_size = 512  # can be optimized
TPB = 16

@cuda.jit("void(complex128[:, :, :, :, :], complex128[:, :, :, :, :], complex128[:, :, :, :, :], int64[:], int64[:, :], complex128[:, :, :, :, :], complex128[:, :, :, :, :])")
# rchol[q, k, i, X, p], rcholbar[q, k, p, X, i], Ghalf[k1, k2, w, i, p], kpq_mat
def rchol_ghalf_to_T12(rchol, rcholbar, Ghalf, kcubelist, kpq_mat, T1, T2):

    # kcubelist: batched index I
    # J: nocc * naux
    # L: nbasis
    # K: nwalkers * nocc
    
    nq = rchol.shape[0]
    nk = rchol.shape[1]
    naux = rchol.shape[3]
    nocc = rchol.shape[2]
    nbasis = rchol.shape[-1]
    nwalker = Ghalf.shape[2]
    J = naux * nocc
    L = nbasis
    K = nocc * nwalker
    nkcube = len(kcubelist)

    batch_id = cuda.blockIdx.z # I
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y # J
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x # K

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y


    ikcube_real = kcubelist[batch_id]
    # expand ikcube to q, k, k'
    iq = ikcube_real // (nk * nk)
    ik = (ikcube_real % (nk * nk)) // nk
    ik_pr = (ikcube_real % (nk * nk)) % nk
    ikpq = kpq_mat[iq, ik]
    ikpr_pq = kpq_mat[iq, ik_pr]
    
   
    # Allocate shared memory for tiles of chol and Ghalf
    schol = cuda.shared.array(shape=(TPB, TPB), dtype=complex128)
    sG1 = cuda.shared.array(shape=(TPB, TPB), dtype=complex128)
    # Initialize the accumulator
    tmp1 = complex128(0.0)
    # \sum_p rchol[aX, p] * Ghalf[wb, p] -> T1[a,X,w,b]
    a = row // naux
    X = row % naux
    w = col // nocc
    b = col % nocc
    for l in range(0, L, TPB):
        if row < J and (l + tx) < L:
            schol[ty, tx] = rchol[iq, ik, a, X, l + tx]
        else:
            schol[ty, tx] = complex128(0.0)
        if (l + ty) < L and col < K:
            sG1[ty, tx] = Ghalf[ikpr_pq, ikpq, w, b, l + ty]
        else:
            sG1[ty, tx] = complex128(0.0)
        cuda.syncthreads()
        for k in range(TPB):
            tmp1 += schol[ty, k] * sG1[k, tx]
        cuda.syncthreads()

    if row < J and col < K:
        T1[batch_id, a, X, w, b] = tmp1

    # Allocate shared memory for tiles of cholbar and Ghalf
    scholbar = cuda.shared.array(shape=(TPB, TPB), dtype=complex128)
    sG2 = cuda.shared.array(shape=(TPB, TPB), dtype=complex128)
    # \sum_p rcholbar[p, Xb] * Ghalf[wa, p] -> T2[X,b,w,a]
    X = row // nocc
    b = row % nocc
    w = col // nocc
    a = col % nocc
    tmp2 = complex128(0.0)
    
    for l in range(0, L, TPB):
        if row < J and (l + tx) < L:
            scholbar[ty, tx] = rcholbar[iq, ik_pr, l + tx, X, b]
        else:
            scholbar[ty, tx] = complex128(0.0)
        if (l + ty) < L and col < K:
            sG2[ty, tx] = Ghalf[ik, ik_pr, w, a, l + ty]
        else:
            sG2[ty, tx] = complex128(0.0)
        cuda.syncthreads()
        for k in range(TPB):
            tmp2 += scholbar[ty, k] * sG2[k, tx]
        cuda.syncthreads()
    
    if row < J and col < K:
        T2[batch_id, X, b, w, a] = tmp2

@cuda.jit("void(complex128[:,:,:,:,:], complex128[:,:,:,:,:], complex128[:])")
def kernel_exchange_reduction(T1, T2, exx_w):
    # T1[batch_id, a, X, w, b], T2[batch_id, X, b, w, a] -> exx_w[w]
    nbatch = T1.shape[0]
    naux = T1.shape[2]
    nocc = T1.shape[1]
    nwalker = T1.shape[3]
    nocc_sq = nocc * nocc
    thread_ix = cuda.threadIdx.x
    block_ix = cuda.blockIdx.x
    if naux < nbatch:
        if block_ix > nwalker * nocc * nocc * nbatch:
            return
        walker = block_ix // (nocc_sq * nbatch)
        batch_id = (block_ix % (nocc_sq * nbatch)) // (nocc_sq)
        a = (block_ix % (nocc_sq * nbatch)) % (nocc_sq) // nocc
        b = (block_ix % (nocc_sq * nbatch)) % (nocc_sq) % nocc
        shared_array = cuda.shared.array(shape=(_block_size,), dtype=numba.complex128)
        block_size = cuda.blockDim.x
        shared_array[thread_ix] = 0.0
        for x in range(thread_ix, naux, block_size):
            shared_array[thread_ix] += T1[batch_id, a, x, walker, b] * T2[batch_id, x, b, walker, a]
        # pylint: disable=no-value-for-parameter
        cuda.syncthreads()
        nreduce = block_size // 2
        indx = nreduce
        for _ in range(0, nreduce):
            if indx == 0:
                break
            if thread_ix < indx:
                shared_array[thread_ix] += shared_array[thread_ix + indx]
            # pylint: disable=no-value-for-parameter
            cuda.syncthreads()
            indx = indx // 2
        if thread_ix == 0:
            cuda.atomic.add(exx_w.real, walker, shared_array[0].real)
            cuda.atomic.add(exx_w.imag, walker, shared_array[0].imag)
    else:
        if block_ix > nwalker * nocc * nocc * naux:
            return
        walker = block_ix // (nocc_sq * naux)
        x = (block_ix % (nocc_sq * naux)) // nocc_sq
        a = (block_ix % (nocc_sq * naux)) % nocc_sq // nocc
        b = (block_ix % (nocc_sq * naux)) % nocc_sq % nocc
        shared_array = cuda.shared.array(shape=(_block_size,), dtype=numba.complex128)
        block_size = cuda.blockDim.x
        shared_array[thread_ix] = 0.0
        for batch_id in range(thread_ix, nbatch, block_size):
            shared_array[thread_ix] += T1[batch_id, a, x, walker, b] * T2[batch_id, x, b, walker, a]
        # pylint: disable=no-value-for-parameter
        cuda.syncthreads()
        nreduce = block_size // 2
        indx = nreduce
        for _ in range(0, nreduce):
            if indx == 0:
                break
            if thread_ix < indx:
                shared_array[thread_ix] += shared_array[thread_ix + indx]
            # pylint: disable=no-value-for-parameter
            cuda.syncthreads()
            indx = indx // 2
        if thread_ix == 0:
            cuda.atomic.add(exx_w.real, walker, shared_array[0].real)
            cuda.atomic.add(exx_w.imag, walker, shared_array[0].imag)



def exx_kpt_kernel(rchol, rcholbar, Ghalf, kcubelist, kpq_mat):
    """Calculate the exchange energy for each walker.
    """
    nwalkers = Ghalf.shape[2]
    nocc = rchol.shape[2]
    naux = rchol.shape[3]
    nbasis = rchol.shape[-1]

    T1 = cp.zeros((len(kcubelist), nocc, naux, nwalkers, nocc), dtype=cp.complex128)
    T2 = cp.zeros((len(kcubelist), naux, nocc, nwalkers, nocc), dtype=cp.complex128)

    threadsperblock = (TPB, TPB, 1)
    blockspergrid_x = (nwalkers * nocc + TPB - 1) // TPB
    blockspergrid_y = (nocc * naux + TPB - 1) // TPB
    blockspergrid_z = len(kcubelist)

    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

    rchol_ghalf_to_T12[blockspergrid, threadsperblock](rchol, rcholbar, Ghalf, kcubelist, kpq_mat, T1, T2)
    cp.cuda.stream.get_current_stream().synchronize()

    exx_w = cp.zeros(nwalkers, dtype=cp.complex128)

    if naux > len(kcubelist):
        kernel_exchange_reduction[nwalkers * nocc * nocc * naux, _block_size](T1, T2, exx_w)
    else:
        kernel_exchange_reduction[nwalkers * nocc * nocc * len(kcubelist), _block_size](T1, T2, exx_w)

    cp.cuda.stream.get_current_stream().synchronize()
    return exx_w


# @cuda.jit("void(complex128[:,:,:,:], complex128[:,:,:,:], complex128[:])")
# def kernel_exchange_reduction(T1, T2, exx_w):
#     naux = T1.shape[1]
#     nocc = T1.shape[0]
#     nwalker = T1.shape[3]
#     nocc_sq = nocc * nocc
#     thread_ix = cuda.threadIdx.x
#     block_ix = cuda.blockIdx.x
#     if block_ix > nwalker * nocc * nocc:
#         return
#     walker = block_ix // nocc_sq
#     a = (block_ix % nocc_sq) // nocc
#     b = (block_ix % nocc_sq) % nocc
#     shared_array = cuda.shared.array(shape=(_block_size,), dtype=numba.complex128)
#     block_size = cuda.blockDim.x
#     shared_array[thread_ix] = 0.0
#     for x in range(thread_ix, naux, block_size):
#         shared_array[thread_ix] += T1[a, x, b, walker] * T2[walker, a, x, b]
#     # pylint: disable=no-value-for-parameter
#     cuda.syncthreads()
#     nreduce = block_size // 2
#     indx = nreduce
#     for _ in range(0, nreduce):
#         if indx == 0:
#             break
#         if thread_ix < indx:
#             shared_array[thread_ix] += shared_array[thread_ix + indx]
#         # pylint: disable=no-value-for-parameter
#         cuda.syncthreads()
#         indx = indx // 2
#     if thread_ix == 0:
#         cuda.atomic.add(exx_w.real, walker, shared_array[0].real)
#         cuda.atomic.add(exx_w.imag, walker, shared_array[0].imag)

        
# def exchange_reduction_kpt(T1, T2, exx_walker):
#     """Reduce intermediate with itself.

#     equivalent to einsum('xijw,wixj->w', T1, T2)

#     Parameters
#     ---------
#     Txiwj : np.ndarray
#         Intemediate tensor of dimension (naux, nocca/b, nwalker, nocca/b).
#     exx_walker : np.ndarray
#         Exchange contribution for all walkers in batch.
#     """
#     nwalkers = T1.shape[-1]
#     nocc = T1.shape[0]
#     blocks_per_grid = nwalkers * nocc * nocc
#     # todo add constants to config
#     # do blocks_per_grid dot products + reductions
#     # look into optimizations.
#     kernel_exchange_reduction[blocks_per_grid, _block_size](T1, T2, exx_walker)
#     cp.cuda.stream.get_current_stream().synchronize()

    
    

    
