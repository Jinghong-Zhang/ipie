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
"""Stabilized linear algebra for QDT-factored thermofield walkers."""

import numpy
import scipy.linalg

from ipie.config import config
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import to_host

# Smallest scale kept in a QDT factorization.  Dead or nearly dead strata are
# floored here so that 1/d and log(d) stay finite at large beta.
SCALE_FLOOR = 1e-290

# (nbasis, dtype.char) -> (geqp3, ungqr, lwork_geqp3, lwork_ungqr).  The lwork
# values are those scipy.linalg.qr obtains from its own lwork=-1 query and must
# be reused verbatim: LAPACK picks blocked or unblocked code paths from lwork,
# which changes the rounding of Q and R.
_pivoted_qr_cache = {}


def _pivoted_qr_funcs(nbasis, dtype):
    key = (nbasis, numpy.dtype(dtype).char)
    entry = _pivoted_qr_cache.get(key)
    if entry is None:
        probe = numpy.zeros((nbasis, nbasis), dtype=dtype, order="F")
        geqp3, ungqr = scipy.linalg.get_lapack_funcs(("geqp3", "orgqr"), (probe,))
        ret = geqp3(probe, lwork=-1, overwrite_a=False)
        lwork_qp3 = int(ret[-2][0].real)
        ret = ungqr(probe, numpy.zeros(nbasis, dtype=dtype), lwork=-1, overwrite_a=False)
        lwork_gqr = int(ret[-2][0].real)
        entry = (geqp3, ungqr, lwork_qp3, lwork_gqr)
        _pivoted_qr_cache[key] = entry
    return entry


def pivoted_qr(F):
    """Column-pivoted QR factorization `F[:, perm] = Q R` of a square matrix.

    Calls geqp3/ungqr directly with the workspace sizes scipy uses, so the
    result is bit-identical to ``scipy.linalg.qr(F, pivoting=True)`` without
    its per-call overhead.

    Parameters
    ----------
    F : :class:`numpy.ndarray`
        Matrix to factorize. It may be overwritten; pass a Fortran-ordered
        array to avoid a copy.

    Returns
    -------
    Q : :class:`numpy.ndarray`
        Unitary factor.
    R : :class:`numpy.ndarray`
        Upper triangular factor.
    perm : :class:`numpy.ndarray`
        Column permutation (zero based).
    """
    nbasis = F.shape[0]
    geqp3, ungqr, lwork_qp3, lwork_gqr = _pivoted_qr_funcs(nbasis, F.dtype)
    qr, jpvt, tau, _, info = geqp3(F, lwork=lwork_qp3, overwrite_a=True)
    if info < 0:
        raise RuntimeError(f"Illegal value in argument {-info} of LAPACK geqp3.")
    R = numpy.triu(qr)
    perm = jpvt - 1  # LAPACK pivots are one based.
    Q, _, info = ungqr(qr, tau, lwork=lwork_gqr, overwrite_a=True)
    if info < 0:
        raise RuntimeError(f"Illegal value in argument {-info} of LAPACK ungqr.")
    return Q, R, perm


def _log_complex_det(A):
    """Complex log det via slogdet; raises on a singular matrix."""
    sgn, logdet = numpy.linalg.slogdet(numpy.asarray(A, dtype=numpy.complex128))
    if sgn == 0:
        raise ZeroDivisionError("Vanishing thermofield overlap.")
    return logdet + numpy.log(sgn)


def principal_log_phase(log_z):
    """Reduce the imaginary part of a complex log to (-pi, pi].

    Sums of slogdet phases are only defined modulo 2 pi i, and overlaps only
    ever enter through exp(log), so the winding carries no information.
    """
    return log_z.real + 1j * (numpy.mod(log_z.imag + numpy.pi, 2.0 * numpy.pi) - numpy.pi)


def stabilized_inverse_one_plus(Q, log_d_left, X, log_d_right, V, Vinv=None, log_det_V=None):
    r"""Stable log determinant and inverse of :math:`I + A`, :math:`A = Q e^{D_l} X e^{D_r} V`.

    Q, X and V are bounded; the scales enter only in the log domain. With the
    two-sided splitting of DQMC, :math:`e^{D} = \bar{D} \hat{D}`,
    :math:`\bar{D} = \max(e^{D}, 1)`, :math:`\hat{D} = \min(e^{D}, 1)`,

    .. math::

        I + A = Q \bar{D}_l \left[\bar{D}_l^{-1} Q^{-1} V^{-1} \bar{D}_r^{-1}
                + \hat{D}_l X \hat{D}_r\right] \bar{D}_r V,

    the bracket has O(1) entries and the unbounded scales appear only as exact
    diagonal factors, which preserves the small scales that plain matrix
    products lose.

    Parameters
    ----------
    Q, X, V : :class:`numpy.ndarray`
        Bounded factors.
    log_d_left, log_d_right : :class:`numpy.ndarray`
        Log scales :math:`D_l` and :math:`D_r`.
    Vinv : :class:`numpy.ndarray`
        Inverse of V if already known.
    log_det_V : complex
        Log determinant of V if already known.

    Returns
    -------
    log_det : complex
        :math:`\log\det(I + A)`.
    inv : :class:`numpy.ndarray`
        :math:`(I + A)^{-1}`.
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


def inv_stack(A):
    """Inverse of every matrix in a stack, NaN for the singular ones.

    numpy.linalg.inv raises if any matrix in the stack is singular. Here a
    singular item gives NaN entries instead, so the caller can kill that walker
    while the rest of the batch proceeds.

    Parameters
    ----------
    A : :class:`numpy.ndarray`
        Stack of square matrices, shape (..., n, n).

    Returns
    -------
    Ainv : :class:`numpy.ndarray`
        Stack of inverses.
    """
    try:
        return xp.linalg.inv(A)
    except numpy.linalg.LinAlgError:
        # Invert item by item on the host.
        A_host = to_host(A)
        out = numpy.full_like(A_host, numpy.nan)
        flat_in = A_host.reshape(-1, A_host.shape[-2], A_host.shape[-1])
        flat_out = out.reshape(-1, A_host.shape[-2], A_host.shape[-1])
        for i in range(flat_in.shape[0]):
            try:
                flat_out[i] = numpy.linalg.inv(flat_in[i])
            except numpy.linalg.LinAlgError:
                pass
        return xp.asarray(out)


def stabilized_inverse_one_plus_batched(
    Q, log_d_left, X, log_d_right, Vinv, log_det_V, Qinv=None, log_det_Q=None, need_inverse=True
):
    r"""Stacked :func:`stabilized_inverse_one_plus` over leading batch axes.

    Same arithmetic as the scalar kernel, evaluated with stacked LAPACK (or
    cuBLAS) calls. A singular factor gives non-finite output for that batch
    item instead of raising, so callers must kill walkers whose log determinant
    or inverse is not finite.

    Parameters
    ----------
    Q, X : :class:`numpy.ndarray`
        Bounded factors, shape (..., n, n).
    log_d_left, log_d_right : :class:`numpy.ndarray`
        Log scales, shape (..., n).
    Vinv : :class:`numpy.ndarray`
        Inverse of V, shape (..., n, n) or (n, n).
    log_det_V : complex or :class:`numpy.ndarray`
        Log determinant of V, scalar or shape (...,).
    Qinv : :class:`numpy.ndarray`
        Inverse of Q if already known.
    log_det_Q : :class:`numpy.ndarray`
        Log determinant of Q if already known.
    need_inverse : bool
        Also return :math:`(I + A)^{-1}`. Default True.

    Returns
    -------
    log_det : :class:`numpy.ndarray`
        :math:`\log\det(I + A)`, shape (...,).
    inv : :class:`numpy.ndarray` or None
        :math:`(I + A)^{-1}`, or None when `need_inverse` is False.
    """
    Q = xp.asarray(Q, dtype=xp.complex128)
    X = xp.asarray(X)
    Vinv = xp.asarray(Vinv)
    ldl = xp.asarray(log_d_left, dtype=xp.float64)
    ldr = xp.asarray(log_d_right, dtype=xp.float64)
    dbar_l = xp.maximum(ldl, 0.0)
    dhat_l = xp.minimum(ldl, 0.0)
    dbar_r = xp.maximum(ldr, 0.0)
    dhat_r = xp.minimum(ldr, 0.0)

    if Qinv is None:
        Qinv = inv_stack(Q)
    QV = Qinv @ Vinv
    bracket = (
        xp.exp(-dbar_l)[..., :, None] * QV * xp.exp(-dbar_r)[..., None, :]
        + xp.exp(dhat_l)[..., :, None] * X * xp.exp(dhat_r)[..., None, :]
    )

    with numpy.errstate(divide="ignore", invalid="ignore"):
        if log_det_Q is None:
            sgn_q, log_q = xp.linalg.slogdet(Q)
            log_det_Q = log_q + xp.log(sgn_q)
        sgn_b, log_b = xp.linalg.slogdet(bracket)
        # Same association as the scalar kernel, so the two agree bit for bit.
        log_det_bracket = log_b + xp.log(sgn_b)
        log_det = (
            log_det_Q + dbar_l.sum(axis=-1) + log_det_bracket + dbar_r.sum(axis=-1) + log_det_V
        )

    if not need_inverse:
        return log_det, None
    binv = inv_stack(bracket)
    core = (xp.exp(-dbar_r)[..., :, None] * binv) * xp.exp(-dbar_l)[..., None, :]
    return log_det, Vinv @ core @ Qinv


def _inv_slogdet_stack(A):
    """Inverse, determinant sign and log |det| of every matrix in a stack.

    On the CPU these are the separate library calls, bit-identical to the
    scalar kernels. On the GPU one batched getrf serves both: the determinant
    is read off the LU diagonal and pivots, then getri reuses the factors.
    Singular items give non-finite entries instead of raising.
    """
    if not config.get_option("use_gpu"):
        with numpy.errstate(divide="ignore", invalid="ignore"):
            sgn, logabs = xp.linalg.slogdet(A)
        return inv_stack(A), sgn, logabs
    return _inv_slogdet_stack_gpu(A)


def _inv_slogdet_stack_gpu(A):
    from cupy.cuda import cublas, device  # pylint: disable=import-error

    shape = A.shape
    n = shape[-1]
    # getrf overwrites its input, so work on a C-ordered copy. cuBLAS reads it
    # column major, i.e. factors A^T, which has the same determinant and whose
    # inverse read back row major is A^{-1}.
    a = xp.array(A, dtype=xp.complex128, order="C").reshape(-1, n, n)
    batch = a.shape[0]
    handle = device.get_cublas_handle()
    step = n * n * a.itemsize
    a_ptrs = xp.arange(a.data.ptr, a.data.ptr + step * batch, step, dtype=xp.uintp)
    piv = xp.empty((batch, n), dtype=xp.int32)
    info = xp.empty((batch,), dtype=xp.int32)
    cublas.zgetrfBatched(handle, n, a_ptrs.data.ptr, n, piv.data.ptr, info.data.ptr, batch)
    diag = a.diagonal(axis1=-2, axis2=-1)
    parity = (piv != xp.arange(1, n + 1, dtype=xp.int32)).sum(axis=-1) % 2
    absd = xp.abs(diag)
    logabs = xp.log(absd).sum(axis=-1)
    sgn = xp.prod(diag / absd, axis=-1) * (1.0 - 2.0 * parity)
    c = xp.empty_like(a)
    c_ptrs = xp.arange(c.data.ptr, c.data.ptr + step * batch, step, dtype=xp.uintp)
    cublas.zgetriBatched(
        handle, n, a_ptrs.data.ptr, n, piv.data.ptr, c_ptrs.data.ptr, n, info.data.ptr, batch
    )
    bad = info != 0
    if bool(bad.any()):
        c[bad] = xp.nan
        sgn[bad] = 0.0
        logabs[bad] = -xp.inf
    return c.reshape(shape), sgn.reshape(shape[:-2]), logabs.reshape(shape[:-2])


def one_plus_qdy(Q, log_d, Y, need_inverse=True):
    r"""Log determinant and Green's function of :math:`I + A`, :math:`A = Q e^{D} Y`.

    Q and Y are bounded (Q orthonormal from the walker QR, Y = T Lambda); the
    walker scales D carry the exponential range. With :math:`D = \bar{D} +
    \hat{D}`, :math:`\bar{D} = \max(D, 0)`, :math:`\hat{D} = \min(D, 0)`,

    .. math::

        \det(I + Q e^{D} Y) &= e^{\mathrm{tr}\bar{D}}
            \det(e^{-\bar{D}} + Y Q e^{\hat{D}}) \\
        I - (I + Q e^{D} Y)^{-1} &= Q e^{\hat{D}}
            \left[e^{-\bar{D}} + Y Q e^{\hat{D}}\right]^{-1} Y

    (Sylvester and Woodbury identities). The bracket is bounded, so it is the
    only matrix factored: one LU per spin block instead of the three of
    :func:`stabilized_inverse_one_plus`. Same arithmetic as
    :func:`one_plus_qdy_batched`, so the two agree bit for bit.

    Parameters
    ----------
    Q, Y : :class:`numpy.ndarray`
        Bounded factors.
    log_d : :class:`numpy.ndarray`
        Walker log scales D.
    need_inverse : bool
        Also return the Green's function. Default True.

    Returns
    -------
    log_det : complex
        :math:`\log\det(I + A)`.
    G : :class:`numpy.ndarray` or None
        :math:`I - (I + A)^{-1}`, or None when `need_inverse` is False.
    """
    dbar = numpy.maximum(log_d, 0.0)
    dhat = numpy.minimum(log_d, 0.0)
    idx = numpy.arange(Q.shape[-1])
    bracket = (Y @ Q) * numpy.exp(dhat)[None, :]
    bracket[idx, idx] += numpy.exp(-dbar)
    with numpy.errstate(divide="ignore", invalid="ignore"):
        sgn, logabs = numpy.linalg.slogdet(bracket)
        log_det = dbar.sum() + (logabs + numpy.log(sgn))
    if not need_inverse:
        return log_det, None
    G = (Q * numpy.exp(dhat)[None, :]) @ inv_stack(bracket) @ Y
    return log_det, G


def one_plus_qdy_batched(Q, log_d, Y, need_inverse=True):
    r"""Stacked :func:`one_plus_qdy` over leading batch axes.

    A singular bracket gives non-finite entries for that batch item instead of
    raising; callers must kill those walkers.

    Parameters
    ----------
    Q : :class:`numpy.ndarray`
        Bounded factors, shape (..., n, n).
    log_d : :class:`numpy.ndarray`
        Walker log scales, shape (..., n).
    Y : :class:`numpy.ndarray`
        Bounded factors, shape (..., n, n) or (n, n) broadcast over the batch.
    need_inverse : bool
        Also return the Green's functions. Default True.

    Returns
    -------
    log_det : :class:`numpy.ndarray`
        :math:`\log\det(I + A)`, shape (...,).
    G : :class:`numpy.ndarray` or None
        :math:`I - (I + A)^{-1}`, or None when `need_inverse` is False.
    """
    Q = xp.asarray(Q, dtype=xp.complex128)
    Y = xp.asarray(Y)
    ld = xp.asarray(log_d, dtype=xp.float64)
    dbar = xp.maximum(ld, 0.0)
    dhat = xp.minimum(ld, 0.0)
    idx = xp.arange(Q.shape[-1])
    bracket = (Y @ Q) * xp.exp(dhat)[..., None, :]
    bracket[..., idx, idx] += xp.exp(-dbar)
    if not need_inverse:
        with numpy.errstate(divide="ignore", invalid="ignore"):
            sgn, logabs = xp.linalg.slogdet(bracket)
            return dbar.sum(axis=-1) + (logabs + xp.log(sgn)), None
    binv, sgn, logabs = _inv_slogdet_stack(bracket)
    with numpy.errstate(divide="ignore", invalid="ignore"):
        log_det = dbar.sum(axis=-1) + (logabs + xp.log(sgn))
    G = (Q * xp.exp(dhat)[..., None, :]) @ binv @ Y
    return log_det, G
