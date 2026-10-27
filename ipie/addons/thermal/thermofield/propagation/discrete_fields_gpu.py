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
"""GPU kernel for the sequential heat-bath sampling of discrete fields."""

import numpy

try:
    # pylint: disable=import-error
    import cupy as cp
except ModuleNotFoundError:
    pass

_block_size = 128

# One thread block per walker. Thread 0 samples the field of every interaction
# term in turn and all threads apply the Sherman-Morrison update of the Green's
# function, leg by leg. A field acts on two legs (spin, orbital) with factors
# 1 + delta[field, x, leg]. Only cupy/complex.cuh is used, so the source also
# compiles with hiprtc on ROCm.
_kernel_code = r"""
#include <cupy/complex.cuh>

extern "C" __global__
void sample_discrete_fields(complex<double>* green, const double* uniforms, const int* leg_spin,
                            const int* leg_orb, const complex<double>* delta,
                            const complex<double>* auxf, const complex<double>* log_scalar,
                            bool* live, double* log_weight, complex<double>* log_s,
                            complex<double>* factors, double* record, long long* counts,
                            const int nwalkers, const int nbasis, const int nfields,
                            const double real_tolerance, const int record_fields)
{
    const int w = blockIdx.x;
    if (w >= nwalkers) {
        return;
    }
    extern __shared__ double2 shared[];
    complex<double>* column = reinterpret_cast<complex<double>*>(shared);
    complex<double>* row = column + nbasis;
    __shared__ int alive, chosen, bad;
    const long long nb2 = (long long)nbasis * nbasis;
    complex<double>* G = green + 2 * nb2 * w;
    // Invalid, steps, kills and forbidden values; kept by thread 0.
    long long n_invalid = 0, n_steps = 0, n_kill = 0, n_forbidden = 0;
    double lw = 0.0;
    complex<double> ls(0.0, 0.0);
    if (threadIdx.x == 0) {
        alive = live[w];
        lw = log_weight[w];
        ls = log_s[w];
    }
    __syncthreads();
    // Only thread 0 writes alive; the others read it after a barrier.
    for (int f = 0; f < nfields; ++f) {
        if (threadIdx.x == 0 && alive) {
            const int sa = leg_spin[2 * f], ia = leg_orb[2 * f];
            const int sb = leg_spin[2 * f + 1], ib = leg_orb[2 * f + 1];
            const complex<double> ga = G[sa * nb2 + ia * nbasis + ia];
            const complex<double> gb = G[sb * nb2 + ib * nbasis + ib];
            complex<double> p[2];
            bool good = true;
            for (int x = 0; x < 2; ++x) {
                const complex<double> da = delta[4 * f + 2 * x];
                const complex<double> db = delta[4 * f + 2 * x + 1];
                complex<double> ratio = (1.0 + da * ga) * (1.0 + db * gb);
                if (sa == sb) {
                    ratio -= (da * db) * (G[sa * nb2 + ia * nbasis + ib]
                                          * G[sa * nb2 + ib * nbasis + ia]);
                }
                p[x] = 0.5 * exp(log_scalar[2 * f + x]) * ratio;
                good = good && isfinite(p[x].real()) && isfinite(p[x].imag())
                       && fabs(p[x].imag()) <= real_tolerance * fmax(1.0, fabs(p[x].real()));
            }
            if (!good) {
                n_invalid += 1;
                alive = 0;
            } else {
                n_steps += 1;
                const double p0 = p[0].real(), p1 = p[1].real();
                n_forbidden += (p0 <= 0.0) + (p1 <= 0.0);
                const double q0 = fmax(p0, 0.0), q1 = fmax(p1, 0.0);
                const double norm = q0 + q1;
                if (!isfinite(norm)) {
                    n_invalid += 1;
                    alive = 0;
                } else if (norm == 0.0) {
                    n_kill += 1;
                    alive = 0;
                } else {
                    const int c = uniforms[(long long)nfields * w + f] >= q0 / norm;
                    chosen = c;
                    lw += log(norm);
                    ls += (log(2.0) + log(c ? p1 : p0)) - log_scalar[2 * f + c];
                    factors[(2 * w + sa) * nbasis + ia] *= auxf[4 * f + 2 * c];
                    factors[(2 * w + sb) * nbasis + ib] *= auxf[4 * f + 2 * c + 1];
                    if (record_fields) {
                        record[(long long)nfields * w + f] = 1 - 2 * c;
                    }
                }
            }
            bad = 0;
        }
        __syncthreads();
        if (!alive) {
            break;
        }
        for (int leg = 0; leg < 2; ++leg) {
            const int s = leg_spin[2 * f + leg], i = leg_orb[2 * f + leg];
            complex<double>* Gs = G + s * nb2;
            for (int k = threadIdx.x; k < nbasis; k += blockDim.x) {
                const complex<double> g = -Gs[k * nbasis + i];
                column[k] = k == i ? g + 1.0 : g;
                row[k] = Gs[i * nbasis + k];
            }
            __syncthreads();
            const complex<double> d = delta[4 * f + 2 * chosen + leg];
            const complex<double> coef = d / (1.0 + d * row[i]);
            for (long long e = threadIdx.x; e < nb2; e += blockDim.x) {
                const complex<double> g = Gs[e] + (coef * column[e / nbasis]) * row[e % nbasis];
                Gs[e] = g;
                if (!(isfinite(g.real()) && isfinite(g.imag()))) {
                    bad = 1;
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0 && bad) {
            n_invalid += 1;
            alive = 0;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        live[w] = alive;
        log_weight[w] = lw;
        log_s[w] = ls;
        counts[4 * w] = n_invalid;
        counts[4 * w + 1] = n_steps;
        counts[4 * w + 2] = n_kill;
        counts[4 * w + 3] = n_forbidden;
    }
}
"""

_kernel = None


def sample_discrete_fields(
    green, uniforms, legs, live, log_weight, log_s, factors, record, real_tolerance
):
    """Sample every field of one slice for a stack of walkers on the GPU.

    The walkers are independent; per walker this is the sequential heat-bath
    loop of :meth:`ThermofieldHubbardDiscrete._slice_fields`. A walker that
    dies stops; its remaining fields are left unsampled.

    Parameters
    ----------
    green : :class:`cupy.ndarray`
        Green's functions at the interaction cut, shape (n, 2, nbasis,
        nbasis); updated in place.
    uniforms : :class:`cupy.ndarray`
        Uniform deviates, shape (n, nfields).
    legs : tuple
        Device arrays (leg_spin, leg_orb, delta, auxf, log_scalar) of shapes
        (nfields, 2), (nfields, 2), (nfields, 2, 2), (nfields, 2, 2) and
        (nfields, 2); delta and auxf are indexed [field, value, leg].
    live : :class:`cupy.ndarray`
        Live walkers, shape (n,); updated in place.
    log_weight : :class:`cupy.ndarray`
        Log weight increments, shape (n,); updated in place.
    log_s : :class:`cupy.ndarray`
        Complex log guide overlaps, shape (n,); updated in place.
    factors : :class:`cupy.ndarray`
        Diagonal field factors, shape (n, 2, nbasis); updated in place.
    record : :class:`cupy.ndarray`
        Sampled fields, shape (n, nfields), or None.
    real_tolerance : float
        Relative imaginary part allowed in a probability.

    Returns
    -------
    counts : :class:`cupy.ndarray`
        Invalid, step, kill and forbidden counts, shape (4,).
    """
    global _kernel  # pylint: disable=global-statement
    if _kernel is None:
        _kernel = cp.RawKernel(_kernel_code, "sample_discrete_fields")
    n, _, nbasis, _ = green.shape
    nfields = uniforms.shape[1]
    leg_spin, leg_orb, delta, auxf, log_scalar = legs
    arrays = (green, live, log_weight, log_s, factors) + tuple(legs)
    if record is not None:
        arrays += (record,)
    if not all(a.flags.c_contiguous for a in arrays):
        raise ValueError("The discrete-field kernel needs C-contiguous arrays.")
    counts = cp.zeros((n, 4), dtype=cp.int64)
    record_fields = record is not None
    if not record_fields:
        record = cp.zeros(1)
    if n > 0:
        _kernel(
            (n,),
            (_block_size,),
            (
                green,
                cp.ascontiguousarray(uniforms, dtype=cp.float64),
                leg_spin,
                leg_orb,
                delta,
                auxf,
                log_scalar,
                live,
                log_weight,
                log_s,
                factors,
                record,
                counts,
                numpy.int32(n),
                numpy.int32(nbasis),
                numpy.int32(nfields),
                numpy.float64(real_tolerance),
                numpy.int32(record_fields),
            ),
            shared_mem=2 * nbasis * 16,
        )
    return counts.sum(axis=0)
