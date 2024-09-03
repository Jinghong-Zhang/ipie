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
# Authors: Joonho Lee <linusjoonho@gmail.com>
#          Fionn Malone <fmalone@google.com>
#

import numpy
import math

from numba import jit
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import synchronize


def construct_force_bias_batch(hamiltonian, walkers, trial, mpi_handler=None):
    """Compute optimal force bias.

    Uses rotated Green's function.

    Parameters
    ----------
    hamiltonian : class
        hamiltonian object.

    walkers : class
        walkers object.

    trial : class
        Trial wavefunction object.

    Returns
    -------
    xbar : :class:`numpy.ndarray`
        Force bias.
    """

    if walkers.name == "SingleDetWalkerBatch" and trial.name == "MultiSlater":
        if hamiltonian.chunked:
            return construct_force_bias_batch_single_det_chunked(
                hamiltonian, walkers, trial, mpi_handler
            )
        else:
            return construct_force_bias_batch_single_det(hamiltonian, walkers, trial)
    elif walkers.name == "MultiDetTrialWalkerBatch" and trial.name == "MultiSlater":
        return construct_force_bias_batch_multi_det_trial(hamiltonian, walkers, trial)


def construct_force_bias_batch_multi_det_trial(hamiltonian, walkers, trial):
    Ga = walkers.Ga.reshape(walkers.nwalkers, hamiltonian.nbasis**2)
    Gb = walkers.Gb.reshape(walkers.nwalkers, hamiltonian.nbasis**2)
    # Cholesky vectors. [M^2, nchol]
    # Why are there so many transposes here?
    if numpy.isrealobj(hamiltonian.chol):
        vbias_batch = numpy.empty((hamiltonian.nchol, walkers.nwalkers), dtype=numpy.complex128)
        vbias_batch.real = hamiltonian.chol.T.dot(Ga.T.real + Gb.T.real)
        vbias_batch.imag = hamiltonian.chol.T.dot(Ga.T.imag + Gb.T.imag)
        vbias_batch = vbias_batch.T.copy()
        return vbias_batch
    else:
        vbias_batch_tmp = hamiltonian.chol.T.dot(Ga.T + Gb.T)
        vbias_batch_tmp = vbias_batch_tmp.T.copy()
        return vbias_batch_tmp


# only implement real Hamiltonian
def construct_force_bias_batch_single_det(
    hamiltonian: "GenericRealChol", walkers: "UHFWalkers", trial: "SingleDetTrial"
):
    """Compute optimal force bias.

    Uses rotated Green's function.

    Parameters
    ----------
    hamiltonian : class
        hamiltonian object.

    walkers : class
        walkers object.

    trial : class
        Trial wavefunction object.

    Returns
    -------
    xbar : :class:`numpy.ndarray`
        Force bias.
    """
    if walkers.rhf:
        Ghalfa = walkers.Ghalfa.reshape(walkers.nwalkers, walkers.nup * hamiltonian.nbasis)
        vbias_batch_real = 2.0 * trial._rchola.dot(Ghalfa.T.real)
        vbias_batch_imag = 2.0 * trial._rchola.dot(Ghalfa.T.imag)
        vbias_batch = xp.empty((walkers.nwalkers, hamiltonian.nchol), dtype=Ghalfa.dtype)
        vbias_batch.real = vbias_batch_real.T.copy()
        vbias_batch.imag = vbias_batch_imag.T.copy()
        synchronize()

        return vbias_batch

    else:
        Ghalfa = walkers.Ghalfa.reshape(walkers.nwalkers, walkers.nup * hamiltonian.nbasis)
        Ghalfb = walkers.Ghalfb.reshape(walkers.nwalkers, walkers.ndown * hamiltonian.nbasis)
        vbias_batch_real = trial._rchola.dot(Ghalfa.T.real) + trial._rcholb.dot(Ghalfb.T.real)
        vbias_batch_imag = trial._rchola.dot(Ghalfa.T.imag) + trial._rcholb.dot(Ghalfb.T.imag)
        vbias_batch = xp.empty((walkers.nwalkers, hamiltonian.nchol), dtype=Ghalfa.dtype)
        vbias_batch.real = vbias_batch_real.T.copy()
        vbias_batch.imag = vbias_batch_imag.T.copy()
        synchronize()
        return vbias_batch

def construct_force_bias_kpt_batch_single_det(
    hamiltonian: "KptComplexChol", walkers: "UHFWalkers", trial: "KptSingleDet"
):
    """Compute optimal force bias.

    Uses rotated Green's function.

    Parameters
    ----------
    hamiltonian : class
        hamiltonian object.

    walkers : class
        walkers object.

    trial : class
        Trial wavefunction object.

    Returns
    -------
    vbias_plus : :class:`numpy.ndarray`
        Force bias for Lplus.
    vbias_minus : :class:`numpy.ndarray`
        Force bias for Lminus.
    """
    if walkers.rhf:
        vbias = numpy.zeros((walkers.nwalkers, hamiltonian.nchol, hamiltonian.nk), dtype=numpy.complex128)
        # ghalf shape: nwalkers, nk, nup, nk, nbsf
        Ghalf_reshape = walkers.Ghalfa.reshape(walkers.nwalkers, hamiltonian.nk, trial.nalpha, hamiltonian.nk, hamiltonian.nbasis)
        for iq in range(hamiltonian.nk):
            for ik in range(hamiltonian.nk):
                ikpq = hamiltonian.ikpq_mat[ik, iq]
                vbias[:, :, iq] += 2.0 * numpy.einsum("gip, aip -> ga", trial._rchola[:, ik, :, iq, :], Ghalf_reshape[:, ik, :, ikpq, :], optimize=True)
        synchronize()
        imq = hamiltonian.imq_vec
        vbias_plus = .5 * 1j * (vbias + vbias[:, :, imq])
        vbias_minus = .5 * (vbias - vbias[:, :, imq])
        return vbias_plus, vbias_minus

    else:
        vbias = numpy.zeros((walkers.nwalkers, hamiltonian.nchol, hamiltonian.nk), dtype=numpy.complex128)
        # ghalf shape: nwalkers, nk, nup, nk, nbsf
        Ghalfa_reshape = walkers.Ghalfa.reshape(walkers.nwalkers, hamiltonian.nk, trial.nalpha, hamiltonian.nk, hamiltonian.nbasis)
        Ghalfb_reshape = walkers.Ghalfb.reshape(walkers.nwalkers, hamiltonian.nk, trial.nbeta, hamiltonian.nk, hamiltonian.nbasis)
        for iq in range(hamiltonian.nk):
            for ik in range(hamiltonian.nk):
                ikpq = hamiltonian.ikpq_mat[ik, iq]
                vbias[:, :, iq] += numpy.einsum("gip, aip -> ag", trial._rchola[:, ik, :, iq, :], Ghalfa_reshape[:, ik, :, ikpq, :], optimize=True) + numpy.einsum("gip, bip -> bg", trial._rcholb[:, ik, :, iq, :], Ghalfb_reshape[:, ik, :, ikpq, :], optimize=True)
        synchronize()
        imq = hamiltonian.imq_vec
        vbias_plus = .5 * 1j * (vbias + vbias[:, :, imq])
        vbias_minus = .5 * (vbias - vbias[:, :, imq])
        return vbias_plus, vbias_minus

def construct_force_bias_kptsymm_batch_single_det(
    hamiltonian: "KptComplexCholSymm", walkers: "UHFWalkers", trial: "KptSingleDet"
):
    """Compute optimal force bias.

    Uses rotated Green's function.

    Parameters
    ----------
    hamiltonian : class
        hamiltonian object.

    walkers : class
        walkers object.

    trial : class
        Trial wavefunction object.

    Returns
    -------
    vbias_plus : :class:`numpy.ndarray`
        Force bias for Lplus.
    vbias_minus : :class:`numpy.ndarray`
        Force bias for Lminus.
    """
    if walkers.rhf:
        vbias_plus = numpy.zeros((walkers.nwalkers, hamiltonian.nchol, hamiltonian.unique_nk), dtype=numpy.complex128)
        vbias_minus = numpy.zeros((walkers.nwalkers, hamiltonian.nchol, hamiltonian.unique_nk), dtype=numpy.complex128)
        # ghalf shape: nwalkers, nk, nup, nk, nbsf
        Ghalf_reshape = walkers.Ghalfa.reshape(walkers.nwalkers, hamiltonian.nk, trial.nalpha, hamiltonian.nk, hamiltonian.nbasis)
        for iq in range(len(hamiltonian.Sset)):
            iq_real = hamiltonian.Sset[iq]
            for ik in range(hamiltonian.nk):
                ikpq = hamiltonian.ikpq_mat[iq_real, ik]
                lpluslbar = trial._rchola[:, ik, :, iq, :] + trial._rcholbara[:, ikpq, :, iq, :].transpose(0, 2, 1)
                lminuslbar = trial._rchola[:, ik, :, iq, :] - trial._rcholbara[:, ikpq, :, iq, :].transpose(0, 2, 1)
                vbias_plus[:, :, iq] += 1j * numpy.einsum("gip, aip -> ga", lpluslbar, Ghalf_reshape[:, ik, :, ikpq, :], optimize=True)
                vbias_minus[:, :, iq] += numpy.einsum("gip, aip -> ga", lminuslbar, Ghalf_reshape[:, ik, :, ikpq, :], optimize=True)

        for iq in range(len(hamiltonian.Sset), len(hamiltonian.Sset) + len(hamiltonian.Qplus)):
            iq_real = hamiltonian.Qplus[iq - len(hamiltonian.Sset)]
            for ik in range(hamiltonian.nk):
                ikpq = hamiltonian.ikpq_mat[iq_real, ik]
                lpluslbar = trial._rchola[:, ik, :, iq, :] + trial._rcholbara[:, ikpq, :, iq, :].transpose(0, 2, 1)
                lminuslbar = trial._rchola[:, ik, :, iq, :] - trial._rcholbara[:, ikpq, :, iq, :].transpose(0, 2, 1)
                vbias_plus[:, :, iq] += 1j * math.sqrt(2) * numpy.einsum("gip, aip -> ga", lpluslbar, Ghalf_reshape[:, ik, :, ikpq, :], optimize=True)
                vbias_minus[:, :, iq] += math.sqrt(2) * numpy.einsum("gip, aip -> ga", lminuslbar, Ghalf_reshape[:, ik, :, ikpq, :], optimize=True)
        synchronize()
        return vbias_plus, vbias_minus

    else:
        vbias_plus = numpy.zeros((walkers.nwalkers, hamiltonian.nchol, hamiltonian.unique_nk), dtype=numpy.complex128)
        vbias_minus = numpy.zeros((walkers.nwalkers, hamiltonian.nchol, hamiltonian.unique_nk), dtype=numpy.complex128)
        # ghalf shape: nwalkers, nk, nup, nk, nbsf
        Ghalfa_reshape = walkers.Ghalfa.reshape(walkers.nwalkers, hamiltonian.nk, trial.nalpha, hamiltonian.nk, hamiltonian.nbasis)
        Ghalfb_reshape = walkers.Ghalfb.reshape(walkers.nwalkers, hamiltonian.nk, trial.nbeta, hamiltonian.nk, hamiltonian.nbasis)
        for iq in range(len(hamiltonian.Sset)):
            iq_real = hamiltonian.Sset[iq]
            for ik in range(hamiltonian.nk):
                ikpq = hamiltonian.ikpq_mat[iq_real, ik]
                vbias_plus[:, :, iq] += .5j * (numpy.einsum("gip, aip -> ag", trial._rchola[:, ik, :, iq, :], Ghalfa_reshape[:, ik, :, ikpq, :], optimize=True) + numpy.einsum("gip, bip -> bg", trial._rcholb[:, ik, :, iq, :], Ghalfb_reshape[:, ik, :, ikpq, :], optimize=True))
                vbias_plus[:, :, iq] += .5j * (numpy.einsum("gpi, aip -> ag", trial._rcholbara[:, ik, :, iq, :], Ghalfa_reshape[:, ikpq, :, ik, :], optimize=True) + numpy.einsum("gpi, bip -> bg", trial._rcholbarb[:, ik, :, iq, :], Ghalfb_reshape[:, ikpq, :, ik, :], optimize=True))

                vbias_minus[:, :, iq] += .5 * (numpy.einsum("gip, aip -> ag", trial._rchola[:, ik, :, iq, :], Ghalfa_reshape[:, ik, :, ikpq, :], optimize=True) + numpy.einsum("gip, bip -> bg", trial._rcholb[:, ik, :, iq, :], Ghalfb_reshape[:, ik, :, ikpq, :], optimize=True))
                vbias_plus[:, :, iq] -= .5 * (numpy.einsum("gpi, aip -> ag", trial._rcholbara[:, ik, :, iq, :], Ghalfa_reshape[:, ikpq, :, ik, :], optimize=True) + numpy.einsum("gpi, bip -> bg", trial._rcholbarb[:, ik, :, iq, :], Ghalfb_reshape[:, ikpq, :, ik, :], optimize=True))
        
        for iq in range(len(hamiltonian.Sset), len(hamiltonian.Sset) + len(hamiltonian.Qplus)):
            iq_real = hamiltonian.Qplus[iq - len(hamiltonian.Sset)]
            for ik in range(hamiltonian.nk):
                ikpq = hamiltonian.ikpq_mat[iq_real, ik]
                vbias_plus[:, :, iq] += .5j * math.sqrt(2) * (numpy.einsum("gip, aip -> ag", trial._rchola[:, ik, :, iq, :], Ghalfa_reshape[:, ik, :, ikpq, :], optimize=True) + numpy.einsum("gip, bip -> bg", trial._rcholb[:, ik, :, iq, :], Ghalfb_reshape[:, ik, :, ikpq, :], optimize=True))
                vbias_plus[:, :, iq] += .5j * math.sqrt(2) * (numpy.einsum("gpi, aip -> ag", trial._rcholbara[:, ik, :, iq, :], Ghalfa_reshape[:, ikpq, :, ik, :], optimize=True) + numpy.einsum("gpi, bip -> bg", trial._rcholbarb[:, ik, :, iq, :], Ghalfb_reshape[:, ikpq, :, ik, :], optimize=True))

                vbias_minus[:, :, iq] += .5 * math.sqrt(2) * (numpy.einsum("gip, aip -> ag", trial._rchola[:, ik, :, iq, :], Ghalfa_reshape[:, ik, :, ikpq, :], optimize=True) + numpy.einsum("gip, bip -> bg", trial._rcholb[:, ik, :, iq, :], Ghalfb_reshape[:, ik, :, ikpq, :], optimize=True))
                vbias_plus[:, :, iq] -= .5 * math.sqrt(2) * (numpy.einsum("gpi, aip -> ag", trial._rcholbara[:, ik, :, iq, :], Ghalfa_reshape[:, ikpq, :, ik, :], optimize=True) + numpy.einsum("gpi, bip -> bg", trial._rcholbarb[:, ik, :, iq, :], Ghalfb_reshape[:, ikpq, :, ik, :], optimize=True))
        synchronize()
        return vbias_plus, vbias_minus


def construct_force_bias_kptisdf_batch_single_det(
    hamiltonian: "KptISDF", walkers: "UHFWalkers", trial: "KptSingleDet"
):
    """Compute optimal force bias.

    Uses rotated Green's function.

    Parameters
    ----------
    hamiltonian : class
        hamiltonian object.

    walkers : class
        walkers object.

    trial : class
        Trial wavefunction object.

    Returns
    -------
    vbias_plus : :class:`numpy.ndarray`
        Force bias for Lplus.
    vbias_minus : :class:`numpy.ndarray`
        Force bias for Lminus.
    """
    nisdf = hamiltonian.nisdf
    rotweightsocc, rotweights = hamiltonian.rotweights
    rcholM = hamiltonian.rcholM
    Ghalfa_reshape = walkers.Ghalfa.reshape(walkers.nwalkers, hamiltonian.nk, trial.nalpha, hamiltonian.nk, hamiltonian.nbasis)
    Ghalfb_reshape = walkers.Ghalfb.reshape(walkers.nwalkers, hamiltonian.nk, trial.nbeta, hamiltonian.nk, hamiltonian.nbasis)
    if walkers.rhf:
        AqPG = numpy.zeros((walkers.nwalkers, hamiltonian.nk, nisdf, 8), dtype=numpy.complex128)
        BqPG = numpy.zeros((walkers.nwalkers, hamiltonian.nk, nisdf, 8), dtype=numpy.complex128)
        for iq in range(hamiltonian.nk):
            Glis = hamiltonian.q2G[iq]
            for iG in range(len(Glis)):
                try:
                    ik_lis = hamiltonian.qG2k[(iq, iG)]
                    for ik in ik_lis:
                        ikpq = hamiltonian.ikpq_mat[ik, iq]
                        AqPG[:, iq, :, iG] += 2.0 * numpy.einsum("Pi, Pp, aip", rotweightsocc[:, :, ik].conj(), rotweights[:, :, ikpq], Ghalfa_reshape[:, ik, :, ikpq, :])
                        BqPG[:, iq, :, iG] += 2.0 * numpy.einsum("Pi, Pp, aip", rotweightsocc[:, :, ikpq].conj(), rotweights[:, :, ik], Ghalfa_reshape[:, ikpq, :, ik, :])
                except KeyError:
                    continue
        vbias = numpy.einsum("XqPG, aqPG -> aXq", rcholM, AqPG)
        vbiasconj = numpy.einsum("XqPG, aqPG -> aXq", rcholM, BqPG)
        vbias_plus = .5 * 1j * (vbias + vbiasconj)
        vbias_minus = .5 * (vbias - vbiasconj)
        return vbias_plus, vbias_minus
    else:
        AqPG = numpy.zeros((walkers.nwalkers, hamiltonian.nk, nisdf, 8), dtype=numpy.complex128)
        BqPG = numpy.zeros((walkers.nwalkers, hamiltonian.nk, nisdf, 8), dtype=numpy.complex128)
        for iq in range(hamiltonian.nk):
            Glis = hamiltonian.q2G[iq]
            for iG in range(len(Glis)):
                try:
                    ik_lis = hamiltonian.qG2k[(iq, iG)]
                    for ik in ik_lis:
                        ikpq = hamiltonian.ikpq_mat[ik, iq]
                        AqPG[:, iq, :, iG] += numpy.einsum("Pi, Pp, aip", rotweightsocc[:, :, ik].conj(), rotweights[:, :, ikpq], Ghalfa_reshape[:, ik, :, ikpq, :]) + numpy.einsum("Pi, Pp, aip", rotweightsocc[:, :, ik].conj(), rotweights[:, :, ikpq], Ghalfb_reshape[:, ik, :, ikpq, :])
                        BqPG[:, iq, :, iG] += numpy.einsum("Pi, Pp, aip", rotweightsocc[:, :, ikpq].conj(), rotweights[:, :, ik], Ghalfa_reshape[:, ikpq, :, ik, :]) + numpy.einsum("Pi, Pp, aip", rotweightsocc[:, :, ikpq].conj(), rotweights[:, :, ik], Ghalfb_reshape[:, ikpq, :, ik, :])
                except KeyError:
                    continue
        vbias = numpy.einsum("XqPG, aqPG -> aXq", rcholM, AqPG)
        vbiasconj = numpy.einsum("XqPG, aqPG -> aXq", rcholM, BqPG)
        vbias_plus = .5 * 1j * (vbias + vbiasconj)
        vbias_minus = .5 * (vbias - vbiasconj)
        return vbias_plus, vbias_minus
    return

def construct_force_bias_batch_single_det_chunked(hamiltonian, walkers, trial, handler):
    """Compute optimal force bias.

    Uses rotated Green's function.

    Parameters
    ----------
    hamiltonian : class
        hamiltonian object.

    walkers : class
        walkers object.

    trial : class
        Trial wavefunction object.

    Returns
    -------
    xbar : :class:`numpy.ndarray`
        Force bias.
    """
    assert hamiltonian.chunked
    assert xp.isrealobj(trial._rchola_chunk)

    Ghalfa = walkers.Ghalfa.reshape(walkers.nwalkers, walkers.nup * hamiltonian.nbasis)
    Ghalfb = walkers.Ghalfb.reshape(walkers.nwalkers, walkers.ndown * hamiltonian.nbasis)

    chol_idxs_chunk = hamiltonian.chol_idxs_chunk

    Ghalfa_recv = xp.zeros_like(Ghalfa)
    Ghalfb_recv = xp.zeros_like(Ghalfb)

    Ghalfa_send = Ghalfa.copy()
    Ghalfb_send = Ghalfb.copy()

    srank = handler.scomm.rank

    vbias_batch_real_recv = xp.zeros((hamiltonian.nchol, walkers.nwalkers))
    vbias_batch_imag_recv = xp.zeros((hamiltonian.nchol, walkers.nwalkers))

    vbias_batch_real_send = xp.zeros((hamiltonian.nchol, walkers.nwalkers))
    vbias_batch_imag_send = xp.zeros((hamiltonian.nchol, walkers.nwalkers))

    vbias_batch_real_send[chol_idxs_chunk, :] = trial._rchola_chunk.dot(
        Ghalfa.T.real
    ) + trial._rcholb_chunk.dot(Ghalfb.T.real)
    vbias_batch_imag_send[chol_idxs_chunk, :] = trial._rchola_chunk.dot(
        Ghalfa.T.imag
    ) + trial._rcholb_chunk.dot(Ghalfb.T.imag)

    receivers = handler.receivers
    for _ in range(handler.ssize - 1):
        synchronize()

        handler.scomm.Isend(Ghalfa_send, dest=receivers[srank], tag=1)
        handler.scomm.Isend(Ghalfb_send, dest=receivers[srank], tag=2)
        handler.scomm.Isend(vbias_batch_real_send, dest=receivers[srank], tag=3)
        handler.scomm.Isend(vbias_batch_imag_send, dest=receivers[srank], tag=4)

        sender = numpy.where(receivers == srank)[0]
        req1 = handler.scomm.Irecv(Ghalfa_recv, source=sender, tag=1)
        req2 = handler.scomm.Irecv(Ghalfb_recv, source=sender, tag=2)
        req3 = handler.scomm.Irecv(vbias_batch_real_recv, source=sender, tag=3)
        req4 = handler.scomm.Irecv(vbias_batch_imag_recv, source=sender, tag=4)
        req1.wait()
        req2.wait()
        req3.wait()
        req4.wait()

        handler.scomm.barrier()

        # prepare sending
        vbias_batch_real_send = vbias_batch_real_recv.copy()
        vbias_batch_imag_send = vbias_batch_imag_recv.copy()
        vbias_batch_real_send[chol_idxs_chunk, :] = trial._rchola_chunk.dot(
            Ghalfa_recv.T.real
        ) + trial._rcholb_chunk.dot(Ghalfb_recv.T.real)
        vbias_batch_imag_send[chol_idxs_chunk, :] = trial._rchola_chunk.dot(
            Ghalfa_recv.T.imag
        ) + trial._rcholb_chunk.dot(Ghalfb_recv.T.imag)
        Ghalfa_send = Ghalfa_recv.copy()
        Ghalfb_send = Ghalfb_recv.copy()

    synchronize()
    handler.scomm.Isend(vbias_batch_real_send, dest=receivers[srank], tag=1)
    handler.scomm.Isend(vbias_batch_imag_send, dest=receivers[srank], tag=2)

    sender = numpy.where(receivers == srank)[0]
    req1 = handler.scomm.Irecv(vbias_batch_real_recv, source=sender, tag=1)
    req2 = handler.scomm.Irecv(vbias_batch_imag_recv, source=sender, tag=2)
    req1.wait()
    req2.wait()
    handler.scomm.barrier()

    vbias_batch = xp.empty((walkers.nwalkers, hamiltonian.nchol), dtype=Ghalfa.dtype)
    vbias_batch.real = vbias_batch_real_recv.T.copy()
    vbias_batch.imag = vbias_batch_imag_recv.T.copy()
    synchronize()
    return vbias_batch
