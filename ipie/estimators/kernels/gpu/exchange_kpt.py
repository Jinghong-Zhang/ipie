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
    from numba import cuda
except ModuleNotFoundError:
    pass

_block_size = 512  #

@cuda.jit("void(complex128[:,:,:,:], complex128[:,:,:,:], complex128[:])")
def kernel_exchange_reduction(T1, T2, exx_w):
    naux = T1.shape[1]
    nocc = T1.shape[0]
    nwalker = T1.shape[3]
    nocc_sq = nocc * nocc
    thread_ix = cuda.threadIdx.x
    block_ix = cuda.blockIdx.x
    if block_ix > nwalker * nocc * nocc:
        return
    walker = block_ix // nocc_sq
    a = (block_ix % nocc_sq) // nocc
    b = (block_ix % nocc_sq) % nocc
    shared_array = cuda.shared.array(shape=(_block_size,), dtype=numba.complex128)
    block_size = cuda.blockDim.x
    shared_array[thread_ix] = 0.0
    for x in range(thread_ix, naux, block_size):
        shared_array[thread_ix] += T1[a, x, b, walker] * T2[walker, a, x, b]
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

        
def exchange_reduction_kpt(T1, T2, exx_walker):
    """Reduce intermediate with itself.

    equivalent to einsum('xijw,wixj->w', T1, T2)

    Parameters
    ---------
    Txiwj : np.ndarray
        Intemediate tensor of dimension (naux, nocca/b, nwalker, nocca/b).
    exx_walker : np.ndarray
        Exchange contribution for all walkers in batch.
    """
    nwalkers = T1.shape[-1]
    nocc = T1.shape[0]
    blocks_per_grid = nwalkers * nocc * nocc
    # todo add constants to config
    # do blocks_per_grid dot products + reductions
    # look into optimizations.
    kernel_exchange_reduction[blocks_per_grid, _block_size](T1, T2, exx_walker)
    cp.cuda.stream.get_current_stream().synchronize()

    
    

    
