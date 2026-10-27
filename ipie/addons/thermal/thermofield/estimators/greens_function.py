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
import numpy


def thermofield_overlap(Lambda, Delta):
    r"""Overlap :math:`S(\Lambda, \Delta) = \det(I + \Lambda\Delta)` of two thermofield Gaussians.

    Parameters
    ----------
    Lambda : :class:`numpy.ndarray`
        Left matrix, shape (nbasis, nbasis); :math:`D_T^\dagger` for a guide :math:`D_T`.
    Delta : :class:`numpy.ndarray`
        Right matrix, shape (nbasis, nbasis).

    Returns
    -------
    ovlp : complex
        :math:`\det(I + \Lambda\Delta)`.
    """
    M = Delta.shape[-1]
    return numpy.linalg.det(numpy.eye(M) + Lambda @ Delta)


def thermofield_log_overlap(Lambda, Delta, log_scale=0.0):
    r"""Complex log of :math:`\det(I + e^{s} \Lambda\Delta)` without overflow.

    The real log scale s restores scalar factors taken out of the matrices
    during stabilization. For s > 0 the determinant is evaluated as
    :math:`e^{M s} \det(e^{-s} I + \Lambda\Delta)`.

    Parameters
    ----------
    Lambda, Delta : :class:`numpy.ndarray`
        Left and right matrices, shape (nbasis, nbasis).
    log_scale : float
        Log scale s. Default 0.0.

    Returns
    -------
    log_ovlp : complex
        :math:`\log|S| + i \arg S`.
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
    r"""Transition Green's function :math:`G = \Delta (I + \Lambda\Delta)^{-1} \Lambda`.

    :math:`\langle c_i^\dagger c_j \rangle / \langle 1 \rangle = G_{ji}`, so
    :math:`\mathrm{Tr}(A G) = \langle c^\dagger A c \rangle`. The log scale restores
    scalar factors as in :func:`thermofield_log_overlap`.

    Parameters
    ----------
    Lambda, Delta : :class:`numpy.ndarray`
        Left and right matrices, shape (nbasis, nbasis).
    log_scale : float
        Log scale. Default 0.0.

    Returns
    -------
    G : :class:`numpy.ndarray`
        Shape (nbasis, nbasis).
    """
    M = Delta.shape[-1]
    A = Lambda @ Delta
    if log_scale <= 0.0:
        scale = numpy.exp(log_scale)
        return scale * (Delta @ numpy.linalg.inv(numpy.eye(M) + scale * A) @ Lambda)
    return Delta @ numpy.linalg.inv(numpy.exp(-log_scale) * numpy.eye(M) + A) @ Lambda


def thermofield_one_rdm(Lambda, Delta, log_scale=0.0):
    r"""Transition 1RDM :math:`P_{ij} = \langle c_i^\dagger c_j \rangle / \langle 1 \rangle`.

    The transpose of :func:`thermofield_greens_function`, as consumed by
    `local_energy_generic_cholesky`.
    """
    return thermofield_greens_function(Lambda, Delta, log_scale=log_scale).T.copy()


def thermofield_wick_normal_ordered_square(L, G):
    r"""Normal-ordered square of a one-body operator by Wick's theorem.

    .. math::

        \langle :(c^\dagger L c)^2: \rangle / \langle 1 \rangle
            = [\mathrm{Tr}(L G)]^2 - \mathrm{Tr}(L G L G),

    with :math:`:c_i^\dagger c_k c_j^\dagger c_l: = c_i^\dagger c_j^\dagger c_l c_k` and G
    from :func:`thermofield_greens_function`. Valid because the physical
    contractions of a thermofield Gaussian have no anomalous terms.
    """
    LG = L @ G
    tr = numpy.trace(LG)
    return tr * tr - numpy.trace(LG @ LG)
