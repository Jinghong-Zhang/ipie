# Copyright 2026 The ipie Developers. All Rights Reserved.
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
"""Thermofield Gaussian algebra.

An (unnormalized) open thermofield Gaussian over M physical modes c_i and M
tilde modes \tilde{c}_j is parameterized by an M x M complex matrix Delta,

    |Phi(Delta)> = exp(sum_{ij} c_i^dagger Delta_{ij} \tilde{c}_j^dagger) |0 \tilde{0}>.

For a left state |Phi(L)> and right state |Phi(Delta)>, with Lambda = L^dagger,

    S(Lambda, Delta) = <Phi(L)|Phi(Delta)> = det(I + Lambda Delta),
    G(Lambda, Delta) = Delta (I + Lambda Delta)^{-1} Lambda,

where G follows the physical one-body convention

    <c_i^dagger c_j> / <1> = G[j, i],    <c^dagger A c> / <1> = Tr(A G).

The ipie 1-RDM convention P[i, j] = <c_i^dagger c_j> is therefore P = G.T.

These functions are the single source of truth for mixed local energies,
replica local energies, and force-bias checks; higher-level code must reduce
to them.
"""

import numpy


def thermofield_overlap(Lambda, Delta):
    r"""Overlap S(Lambda, Delta) = det(I + Lambda Delta).

    Parameters
    ----------
    Lambda : :class:`numpy.ndarray`
        Left thermofield matrix (M, M).  For a guide D_T this is D_T^dagger.
    Delta : :class:`numpy.ndarray`
        Right thermofield matrix (M, M).

    Returns
    -------
    ovlp : complex
        det(I + Lambda Delta).
    """
    M = Delta.shape[-1]
    return numpy.linalg.det(numpy.eye(M) + Lambda @ Delta)


def thermofield_log_overlap(Lambda, Delta, log_scale=0.0):
    r"""Stable complex log of det(I + e^{log_scale} Lambda Delta).

    The optional real `log_scale` accounts for scalar gauge factors pulled
    out of Lambda and/or Delta during walker stabilization (the physical
    matrix is e^{log_scale} Lambda Delta).  For log_scale > 0 the determinant
    is evaluated in the branch det = e^{M * log_scale} det(e^{-log_scale} I +
    Lambda Delta) so that no overflowing intermediate is formed.

    Returns
    -------
    log_ovlp : complex
        log|S| + i arg(S).
    """
    M = Delta.shape[-1]
    A = (Lambda @ Delta).astype(numpy.complex128)
    if log_scale <= 0.0:
        sgn, logdet = numpy.linalg.slogdet(numpy.eye(M) + numpy.exp(log_scale) * A)
        shift = 0.0
    else:
        sgn, logdet = numpy.linalg.slogdet(numpy.exp(-log_scale) * numpy.eye(M) + A)
        shift = M * log_scale
    if sgn == 0:
        raise ZeroDivisionError("Vanishing thermofield overlap.")
    return shift + logdet + numpy.log(sgn)


def thermofield_greens_function(Lambda, Delta, log_scale=0.0):
    r"""Transition Green's function G = Delta (I + Lambda Delta)^{-1} Lambda.

    Convention: <c_i^dagger c_j> / <1> = G[j, i], i.e. Tr(A G) = <c^dagger A c>.

    As in :func:`thermofield_log_overlap`, `log_scale` restores scalar gauge
    factors: the physical pair is (e^{a} Lambda, e^{b} Delta) with
    a + b = log_scale, for which G is evaluated without overflow.

    Returns
    -------
    G : :class:`numpy.ndarray`
        (M, M) transition Green's function.
    """
    M = Delta.shape[-1]
    A = Lambda @ Delta
    if log_scale <= 0.0:
        scale = numpy.exp(log_scale)
        return scale * (Delta @ numpy.linalg.inv(numpy.eye(M) + scale * A) @ Lambda)
    return Delta @ numpy.linalg.inv(numpy.exp(-log_scale) * numpy.eye(M) + A) @ Lambda


def thermofield_one_rdm(Lambda, Delta, log_scale=0.0):
    r"""Transition 1-RDM in the ipie convention P[i, j] = <c_i^dagger c_j> / <1>.

    This is the transpose of :func:`thermofield_greens_function` and is the
    object consumed by `local_energy_generic_cholesky`.
    """
    return thermofield_greens_function(Lambda, Delta, log_scale=log_scale).T.copy()


def _log_complex_det(A):
    """Complex log det via slogdet (fail fast on singular input)."""
    sgn, logdet = numpy.linalg.slogdet(numpy.asarray(A, dtype=numpy.complex128))
    if sgn == 0:
        raise ZeroDivisionError("Vanishing thermofield overlap.")
    return logdet + numpy.log(sgn)


def principal_log_phase(log_z):
    """Reduce the imaginary part of a complex log to (-pi, pi].

    Sums of slogdet phases are only defined modulo 2 pi i; overlaps and
    ratios only ever use exp(log), so the winding is unphysical.
    """
    return log_z.real + 1j * (numpy.mod(log_z.imag + numpy.pi, 2.0 * numpy.pi) - numpy.pi)


def stabilized_inverse_one_plus(Q, log_d_left, X, log_d_right, V, Vinv=None, log_det_V=None):
    r"""Stable (log det, inverse) of I + A with A = Q e^{Dl} X e^{Dr} V.

    Q, X, V are bounded matrices; the diagonal scales enter only in the log
    domain.  Uses the standard two-sided big/small splitting of DQMC (each
    D = Dbar Dhat with Dbar = max(D, 1), Dhat = min(D, 1)):

        I + A = Q Dbar_l [Dbar_l^{-1} Q^{-1} V^{-1} Dbar_r^{-1}
                          + Dhat_l X Dhat_r] Dbar_r V,

    so the bracket has O(1) entries and the unbounded scales appear only as
    exact diagonal scalings.  This preserves the small scales that plain
    matrix products destroy (the FT-AFQMC stratification problem).

    Returns
    -------
    (log_det, inv) : tuple
        Complex log det(I + A) and the matrix (I + A)^{-1}.
    """
    dbar_l = numpy.maximum(log_d_left, 0.0)
    dhat_l = numpy.minimum(log_d_left, 0.0)
    dbar_r = numpy.maximum(log_d_right, 0.0)
    dhat_r = numpy.minimum(log_d_right, 0.0)

    Qinv = numpy.linalg.inv(Q)
    if Vinv is None:
        Vinv = numpy.linalg.inv(V)
    if log_det_V is None:
        log_det_V = _log_complex_det(V)

    bracket = (numpy.exp(-dbar_l)[:, None] * (Qinv @ Vinv)) * numpy.exp(-dbar_r)[None, :]
    bracket += (numpy.exp(dhat_l)[:, None] * X) * numpy.exp(dhat_r)[None, :]

    log_det = (
        _log_complex_det(Q)
        + numpy.sum(dbar_l)
        + _log_complex_det(bracket)
        + numpy.sum(dbar_r)
        + log_det_V
    )
    inv = (
        Vinv
        @ ((numpy.exp(-dbar_r)[:, None] * numpy.linalg.inv(bracket)) * numpy.exp(-dbar_l)[None, :])
        @ Qinv
    )
    return log_det, inv


def factored_pair_log_overlap(Q_left, log_d_left, T_left, Q_right, log_d_right, T_right):
    r"""log det(I + Delta_L^dagger Delta_R) for QDT-factored walkers.

    Each walker is stored as Delta = Q diag(e^{log_d}) T with bounded Q, T.
    By Sylvester's identity det(I + Delta_L^dag Delta_R) =
    det(I + Delta_R Delta_L^dag) with

        Delta_R Delta_L^dag = Q_R e^{D_R} (T_R T_L^dag) e^{D_L} Q_L^dag,

    which is evaluated with the two-sided stabilized splitting of
    :func:`stabilized_inverse_one_plus`.
    """
    log_det, _ = factored_pair_log_overlap_and_greens_function(
        Q_left, log_d_left, T_left, Q_right, log_d_right, T_right
    )
    return log_det


def factored_pair_log_overlap_and_greens_function(
    Q_left, log_d_left, T_left, Q_right, log_d_right, T_right
):
    r"""Log overlap and transition Green's function for a QDT-factored pair.

    Both quantities depend on the same stabilized inverse.  Computing them
    together avoids repeating the matrix products, inversions, and
    determinants needed by :func:`stabilized_inverse_one_plus`.
    """
    M = Q_right.shape[-1]
    log_det, inv = stabilized_inverse_one_plus(
        Q_right,
        log_d_right,
        T_right @ T_left.conj().T,
        log_d_left,
        Q_left.conj().T,
    )
    return principal_log_phase(log_det), numpy.eye(M) - inv


def factored_pair_greens_function(Q_left, log_d_left, T_left, Q_right, log_d_right, T_right):
    r"""Transition G for a QDT-factored pair, same convention as
    :func:`thermofield_greens_function` with Lambda = Delta_L^dagger.

    Uses the push-through identity

        G = Delta_R (I + Delta_L^dag Delta_R)^{-1} Delta_L^dag
          = I - (I + Delta_R Delta_L^dag)^{-1},

    with the inverse evaluated by :func:`stabilized_inverse_one_plus`.
    """
    _, G = factored_pair_log_overlap_and_greens_function(
        Q_left, log_d_left, T_left, Q_right, log_d_right, T_right
    )
    return G


def thermofield_wick_normal_ordered_square(L, G):
    r"""Normal-ordered square via Wick's theorem.

    < : (c^dagger L c)^2 : > / <1> = [Tr(L G)]^2 - Tr(L G L G),

    where :...: denotes normal ordering, i.e.
    :c_i^dagger c_k c_j^dagger c_l: = c_i^dagger c_j^dagger c_l c_k, and G is
    the transition Green's function from :func:`thermofield_greens_function`.
    Valid because the physical-sector contractions of a thermofield Gaussian
    have no anomalous <cc> / <c^dagger c^dagger> terms.
    """
    LG = L @ G
    tr = numpy.trace(LG)
    return tr * tr - numpy.trace(LG @ LG)
