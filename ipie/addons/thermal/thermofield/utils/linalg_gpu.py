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
"""GPU kernel for the Householder QR of a stack of small square matrices."""

import numpy

try:
    # pylint: disable=import-error
    import cupy as cp
except ModuleNotFoundError:
    pass

_block_size = 64

# One thread block per matrix (a power-of-two number of threads). The column
# below the diagonal is loaded in parallel and scaled by its largest entry, so
# that columns far below the leading ones neither underflow nor lose their
# direction; its norm is a block reduction, and each thread reflects one
# trailing column at a time. Q is accumulated backwards from the identity.
# Only cupy/complex.cuh is used, so the source also compiles with hiprtc on
# ROCm.
_kernel_code = r"""
#include <cupy/complex.cuh>

__device__ void block_reduce(double* red, const bool take_max)
{
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            red[threadIdx.x] = take_max ? fmax(red[threadIdx.x], red[threadIdx.x + s])
                                        : red[threadIdx.x] + red[threadIdx.x + s];
        }
        __syncthreads();
    }
}

extern "C" __global__
void householder_qr(complex<double>* R, complex<double>* Q, complex<double>* U, const int batch,
                    const int m)
{
    const int b = blockIdx.x;
    if (b >= batch) {
        return;
    }
    const int tid = threadIdx.x, nt = blockDim.x;
    const long long mm = (long long)m * m;
    complex<double>* A = R + mm * b;
    complex<double>* V = U + mm * b;
    complex<double>* P = Q + mm * b;
    extern __shared__ double2 shared[];
    complex<double>* u = reinterpret_cast<complex<double>*>(shared);
    double* red = reinterpret_cast<double*>(u + m);
    __shared__ double scale, vnorm;
    for (int k = 0; k < m - 1; ++k) {
        double local = 0.0;
        for (int i = k + tid; i < m; i += nt) {
            u[i] = A[i * m + k];
            local = fmax(local, abs(u[i]));
        }
        red[tid] = local;
        __syncthreads();
        block_reduce(red, true);
        if (tid == 0) {
            scale = red[0];
        }
        __syncthreads();
        if (scale > 0.0) {
            local = 0.0;
            for (int i = k + tid; i < m; i += nt) {
                u[i] /= scale;
                local += u[i].real() * u[i].real() + u[i].imag() * u[i].imag();
            }
            red[tid] = local;
            __syncthreads();
            block_reduce(red, false);
            if (tid == 0) {
                const double norm = sqrt(red[0]);
                const complex<double> alpha = u[k];
                const double absa = abs(alpha);
                const complex<double> phase =
                    absa > 0.0 ? alpha / absa : complex<double>(1.0, 0.0);
                // v = x - beta e_1 with beta = -phase |x|, so |v|^2 = 2 |x| (|x| + |alpha|).
                u[k] = alpha + phase * norm;
                vnorm = sqrt(2.0 * norm * (norm + absa));
            }
            __syncthreads();
            for (int i = k + tid; i < m; i += nt) {
                u[i] /= vnorm;
                V[i * m + k] = u[i];
            }
            __syncthreads();
            for (int j = k + tid; j < m; j += nt) {
                complex<double> w(0.0, 0.0);
                for (int i = k; i < m; ++i) {
                    w += conj(u[i]) * A[i * m + j];
                }
                w *= 2.0;
                if (j == k) {
                    A[k * m + k] -= u[k] * w;
                    for (int i = k + 1; i < m; ++i) {
                        A[i * m + k] = 0.0;
                    }
                } else {
                    for (int i = k; i < m; ++i) {
                        A[i * m + j] -= u[i] * w;
                    }
                }
            }
        } else {
            for (int i = k + tid; i < m; i += nt) {
                V[i * m + k] = 0.0;
            }
        }
        __syncthreads();
    }
    for (long long e = tid; e < mm; e += nt) {
        P[e] = e / m == e % m ? 1.0 : 0.0;
    }
    for (int k = m - 2; k >= 0; --k) {
        for (int i = k + tid; i < m; i += nt) {
            u[i] = V[i * m + k];
        }
        __syncthreads();
        for (int j = k + tid; j < m; j += nt) {
            complex<double> w(0.0, 0.0);
            for (int i = k; i < m; ++i) {
                w += conj(u[i]) * P[i * m + j];
            }
            w *= 2.0;
            for (int i = k; i < m; ++i) {
                P[i * m + j] -= u[i] * w;
            }
        }
        __syncthreads();
    }
}
"""


_kernel = None


def householder_qr(A):
    """Householder QR `A = Q R` of every square matrix in a stack on the GPU.

    Parameters
    ----------
    A : :class:`cupy.ndarray`
        Stack of square matrices, shape (..., m, m).

    Returns
    -------
    Q : :class:`cupy.ndarray`
        Unitary factors.
    R : :class:`cupy.ndarray`
        Upper triangular factors.
    """
    global _kernel  # pylint: disable=global-statement
    if _kernel is None:
        _kernel = cp.RawKernel(_kernel_code, "householder_qr")
    shape = A.shape
    m = shape[-1]
    R = cp.array(A, dtype=cp.complex128, order="C").reshape(-1, m, m)
    batch = R.shape[0]
    Q = cp.empty_like(R)
    U = cp.empty_like(R)
    if batch > 0:
        _kernel(
            (batch,),
            (_block_size,),
            (R, Q, U, numpy.int32(batch), numpy.int32(m)),
            shared_mem=16 * m + 8 * _block_size,
        )
    return Q.reshape(shape), R.reshape(shape)
