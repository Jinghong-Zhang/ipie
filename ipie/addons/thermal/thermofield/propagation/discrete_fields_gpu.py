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
# 1 + delta[field, x, leg]. The updates are delayed: each spin keeps up to kmax
# rank-1 terms U W in shared memory, the entries needed are G0 + U W, and the
# terms are added to G0 in one pass when kmax are pending, which divides the
# traffic of G0 by kmax; kmax = 0 updates G0 at once (faster for small nbasis).
# With COMPLETION the importance function is the completion guide, whose Green's
# function receives the same updates; otherwise it is the guide. The variants
# are compiled separately (DELAYED = 0 or 1, COMPLETION = 0 or 1). Only
# cupy/complex.cuh is used, so the source also compiles with hiprtc on ROCm.
_kernel_code = r"""
#include <cupy/complex.cuh>

// Green's functions per walker: the guide, and the completion guide.
#define NGREEN (COMPLETION ? 2 : 1)

// Entry (r, c) of the current Green's function G0 + U W of one spin.
__device__ complex<double> current(const complex<double>* G0, const complex<double>* U,
                                   const complex<double>* W, const int pending, const int nbasis,
                                   const int kmax, const int r, const int c)
{
    complex<double> g = G0[(long long)r * nbasis + c];
#if DELAYED
    for (int t = 0; t < pending; ++t) {
        g += U[r * kmax + t] * W[t * nbasis + c];
    }
#endif
    return g;
}

// G0 += U W for one spin; returns whether every entry is finite.
__device__ bool flush(complex<double>* G0, const complex<double>* U, const complex<double>* W,
                      const int pending, const int nbasis, const int kmax)
{
    bool finite = true;
    const long long nb2 = (long long)nbasis * nbasis;
    for (long long e = threadIdx.x; e < nb2; e += blockDim.x) {
        const int r = e / nbasis, c = e % nbasis;
        complex<double> g = G0[e];
        for (int t = 0; t < pending; ++t) {
            g += U[r * kmax + t] * W[t * nbasis + c];
        }
        G0[e] = g;
        finite = finite && isfinite(g.real()) && isfinite(g.imag());
    }
    return finite;
}

__device__ bool is_finite(const complex<double> z)
{
    return isfinite(z.real()) && isfinite(z.imag());
}

extern "C" __global__
void sample_discrete_fields(complex<double>* green, complex<double>* green_imp,
                            const double* uniforms, const int* leg_spin, const int* leg_orb,
                            const complex<double>* delta, const complex<double>* auxf,
                            const complex<double>* log_scalar, bool* live, double* log_weight,
                            complex<double>* log_s, complex<double>* log_s_imp,
                            complex<double>* factors, double* record, long long* counts,
                            const int nwalkers, const int nbasis, const int nfields,
                            const double real_tolerance, const int record_fields, const int kmax)
{
    const int w = blockIdx.x;
    if (w >= nwalkers) {
        return;
    }
    extern __shared__ double2 shared[];
    complex<double>* column = reinterpret_cast<complex<double>*>(shared);
    complex<double>* row = column + nbasis;
    // Pending terms of Green's function g, spin s: U (nbasis x kmax) and W (kmax x nbasis).
    complex<double>* Us[NGREEN][2];
    complex<double>* Ws[NGREEN][2];
    complex<double>* Gs[NGREEN][2];
    const long long nb2 = (long long)nbasis * nbasis;
    for (int g = 0; g < NGREEN; ++g) {
        complex<double>* terms = row + nbasis + 4 * g * nbasis * kmax;
        Us[g][0] = terms;
        Us[g][1] = terms + nbasis * kmax;
        Ws[g][0] = terms + 2 * nbasis * kmax;
        Ws[g][1] = terms + 3 * nbasis * kmax;
        complex<double>* G = (g == 0 ? green : green_imp) + 2 * nb2 * w;
        Gs[g][0] = G;
        Gs[g][1] = G + nb2;
    }
    __shared__ int alive, chosen, bad;
    int pending[2] = {0, 0};
    // Invalid, steps, kills and forbidden values; kept by thread 0.
    long long n_invalid = 0, n_steps = 0, n_kill = 0, n_forbidden = 0;
    double lw = 0.0;
    complex<double> ls(0.0, 0.0), lsi(0.0, 0.0);
    if (threadIdx.x == 0) {
        alive = live[w];
        lw = log_weight[w];
        ls = log_s[w];
#if COMPLETION
        lsi = log_s_imp[w];
#endif
        bad = 0;
    }
    __syncthreads();
    // Only thread 0 writes alive; the others read it after a barrier.
    for (int f = 0; f < nfields; ++f) {
        if (threadIdx.x == 0 && alive) {
            const int sa = leg_spin[2 * f], ia = leg_orb[2 * f];
            const int sb = leg_spin[2 * f + 1], ib = leg_orb[2 * f + 1];
            // Probabilities from the guide (g = 0) and the completion guide (g = 1).
            complex<double> p[NGREEN][2];
            bool good = true;
            for (int g = 0; g < NGREEN; ++g) {
                const complex<double> ga =
                    current(Gs[g][sa], Us[g][sa], Ws[g][sa], pending[sa], nbasis, kmax, ia, ia);
                const complex<double> gb =
                    current(Gs[g][sb], Us[g][sb], Ws[g][sb], pending[sb], nbasis, kmax, ib, ib);
                complex<double> exchange(0.0, 0.0);
                if (sa == sb) {
                    exchange =
                        current(Gs[g][sa], Us[g][sa], Ws[g][sa], pending[sa], nbasis, kmax, ia, ib)
                        * current(Gs[g][sa], Us[g][sa], Ws[g][sa], pending[sa], nbasis, kmax, ib, ia);
                }
                for (int x = 0; x < 2; ++x) {
                    const complex<double> da = delta[4 * f + 2 * x];
                    const complex<double> db = delta[4 * f + 2 * x + 1];
                    complex<double> ratio = (1.0 + da * ga) * (1.0 + db * gb);
                    if (sa == sb) {
                        ratio -= (da * db) * exchange;
                    }
                    p[g][x] = 0.5 * exp(log_scalar[2 * f + x]) * ratio;
                    good = good && is_finite(p[g][x])
                           && fabs(p[g][x].imag()) <= real_tolerance * fmax(1.0, fabs(p[g][x].real()));
                }
            }
            if (!good) {
                n_invalid += 1;
                alive = 0;
            } else {
                n_steps += 1;
                const double p0 = p[0][0].real(), p1 = p[0][1].real();
                n_forbidden += (p0 <= 0.0) + (p1 <= 0.0);
#if COMPLETION
                // Values forbidden by the guide get zero probability; the
                // allowed ones are drawn with the completion-guide magnitude.
                const double q0 = p0 > 0.0 ? fabs(p[1][0].real()) : 0.0;
                const double q1 = p1 > 0.0 ? fabs(p[1][1].real()) : 0.0;
#else
                const double q0 = fmax(p0, 0.0), q1 = fmax(p1, 0.0);
#endif
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
#if COMPLETION
                    lsi += (log(2.0) + log(c ? q1 : q0)) - log_scalar[2 * f + c].real();
#endif
                    factors[(2 * w + sa) * nbasis + ia] *= auxf[4 * f + 2 * c];
                    factors[(2 * w + sb) * nbasis + ib] *= auxf[4 * f + 2 * c + 1];
                    if (record_fields) {
                        record[(long long)nfields * w + f] = 1 - 2 * c;
                    }
                }
            }
        }
        __syncthreads();
        if (!alive) {
            break;
        }
        for (int leg = 0; leg < 2; ++leg) {
            const int s = leg_spin[2 * f + leg], i = leg_orb[2 * f + leg];
            const int t = pending[s];
            const complex<double> d = delta[4 * f + 2 * chosen + leg];
            for (int g = 0; g < NGREEN; ++g) {
                complex<double>* G0 = Gs[g][s];
                complex<double>* U = Us[g][s];
                complex<double>* W = Ws[g][s];
                for (int k = threadIdx.x; k < nbasis; k += blockDim.x) {
                    const complex<double> gk = -current(G0, U, W, t, nbasis, kmax, k, i);
                    column[k] = k == i ? gk + 1.0 : gk;
                    row[k] = current(G0, U, W, t, nbasis, kmax, i, k);
                }
                __syncthreads();
                const complex<double> coef = d / (1.0 + d * row[i]);
#if DELAYED
                for (int k = threadIdx.x; k < nbasis; k += blockDim.x) {
                    const complex<double> uk = coef * column[k];
                    U[k * kmax + t] = uk;
                    W[t * nbasis + k] = row[k];
                    if (!(is_finite(uk) && is_finite(row[k]))) {
                        bad = 1;
                    }
                }
                if (t + 1 == kmax) {
                    __syncthreads();
                    if (!flush(G0, U, W, kmax, nbasis, kmax)) {
                        bad = 1;
                    }
                }
#else
                // Immediate rank-1 update of G0.
                for (long long e = threadIdx.x; e < nb2; e += blockDim.x) {
                    const complex<double> gnew = G0[e] + (coef * column[e / nbasis]) * row[e % nbasis];
                    G0[e] = gnew;
                    if (!is_finite(gnew)) {
                        bad = 1;
                    }
                }
#endif
                // Column and row are reused by the next Green's function.
                __syncthreads();
            }
#if DELAYED
            // Every thread keeps the same count.
            pending[s] = t + 1 == kmax ? 0 : t + 1;
#endif
        }
        if (threadIdx.x == 0 && bad) {
            n_invalid += 1;
            alive = 0;
        }
        __syncthreads();
    }
#if DELAYED
    if (alive) {
        for (int s = 0; s < 2; ++s) {
            for (int g = 0; g < NGREEN; ++g) {
                if (pending[s] > 0 && !flush(Gs[g][s], Us[g][s], Ws[g][s], pending[s], nbasis, kmax)) {
                    bad = 1;
                }
            }
        }
        __syncthreads();
        if (threadIdx.x == 0 && bad) {
            n_invalid += 1;
            alive = 0;
        }
    }
#endif
    if (threadIdx.x == 0) {
        live[w] = alive;
        log_weight[w] = lw;
        log_s[w] = ls;
#if COMPLETION
        log_s_imp[w] = lsi;
#endif
        counts[4 * w] = n_invalid;
        counts[4 * w + 1] = n_steps;
        counts[4 * w + 2] = n_kill;
        counts[4 * w + 3] = n_forbidden;
    }
}
"""

# Shared memory for the pending terms of both spins, column and row.
_shared_budget = 40 * 1024

_kernels = {}


def sample_discrete_fields(
    green,
    uniforms,
    legs,
    live,
    log_weight,
    log_s,
    factors,
    record,
    real_tolerance,
    kmax=None,
    completion=None,
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
    kmax : int
        Number of delayed updates per spin, 0 for immediate updates. Default
        None: chosen from nbasis.
    completion : tuple
        (green_imp, log_s_imp): Green's functions at the cut, shape (n, 2,
        nbasis, nbasis), and real log overlaps, shape (n,), of the completion
        guide, which draws the values the guide allows (importance_guide
        "completion"); updated in place. Default None.

    Returns
    -------
    counts : :class:`cupy.ndarray`
        Invalid, step, kill and forbidden counts, shape (4,).
    """
    n, _, nbasis, _ = green.shape
    nfields = uniforms.shape[1]
    leg_spin, leg_orb, delta, auxf, log_scalar = legs
    green_imp = log_s_imp = None
    if completion is not None:
        green_imp, log_s_imp = completion
    arrays = (green, live, log_weight, log_s, factors) + tuple(legs)
    arrays += tuple(a for a in (record, green_imp, log_s_imp) if a is not None)
    if not all(a.flags.c_contiguous for a in arrays):
        raise ValueError("The discrete-field kernel needs C-contiguous arrays.")
    two_guides = int(completion is not None)
    if kmax is None:
        # On an A100 immediate updates were fastest for nbasis 16 and about
        # nbasis / 8 delayed ones from nbasis 36 to 100, within the budget.
        budget = (_shared_budget // 16 - 2 * nbasis) // (4 * nbasis * (1 + two_guides))
        kmax = 0 if nbasis < 32 else int(max(1, min(budget, nbasis // 8, 16)))
    delayed = int(kmax > 0)
    if (delayed, two_guides) not in _kernels:
        _kernels[delayed, two_guides] = cp.RawKernel(
            _kernel_code,
            "sample_discrete_fields",
            options=(f"-DDELAYED={delayed}", f"-DCOMPLETION={two_guides}"),
        )
    counts = cp.zeros((n, 4), dtype=cp.int64)
    record_fields = record is not None
    # Unused pointers.
    unused = cp.zeros(1, dtype=cp.complex128)
    if n > 0:
        _kernels[delayed, two_guides](
            (n,),
            (_block_size,),
            (
                green,
                unused if green_imp is None else green_imp,
                cp.ascontiguousarray(uniforms, dtype=cp.float64),
                leg_spin,
                leg_orb,
                delta,
                auxf,
                log_scalar,
                live,
                log_weight,
                log_s,
                unused if log_s_imp is None else log_s_imp,
                factors,
                unused if record is None else record,
                counts,
                numpy.int32(n),
                numpy.int32(nbasis),
                numpy.int32(nfields),
                numpy.float64(real_tolerance),
                numpy.int32(record_fields),
                numpy.int32(kmax),
            ),
            shared_mem=16 * (2 * nbasis + 4 * (1 + two_guides) * nbasis * kmax),
        )
    return counts.sum(axis=0)
