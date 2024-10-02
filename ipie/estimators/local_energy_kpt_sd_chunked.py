# from line_profiler import LineProfiler
from math import ceil, sqrt

import numpy
from numba import jit

from ipie.estimators.local_energy import local_energy_G
from ipie.estimators.kernels import exchange_reduction
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import synchronize

from ipie.systems.generic import Generic
from ipie.hamiltonians.kpt_chunked import KptComplexCholChunked
from ipie.walkers.uhf_walkers import UHFWalkers
from ipie.trial_wavefunction.single_det_kpt import KptSingleDet
from ipie.estimators.local_energy_kpt_sd import kpt_symmchol_ecoul_kernel_uhf, kpt_symmchol_exx_kernel

# from line_profiler import profile

import plum
# Note specialisations occur to because:
# 1. Numba does not allow for mixing types without a warning so need to split
# real and complex components apart when rchol is real. Green's function is
# complex in general.
# Optimize for case when wavefunction is RHF (factor of 2 saving)

@plum.dispatch
def local_energy_kpt_single_det_uhf_chunked(
    system: Generic,
    hamiltonian: KptComplexCholChunked,
    walkers: UHFWalkers,
    trial: KptSingleDet,
):
    """Compute local energy for walker batch (all walkers at once).

    Single determinant UHF case.

    Parameters
    ----------
    system : system object
        System being studied.
    hamiltonian : hamiltonian object
        Hamiltonian being studied.
    walkers : WalkerBatch
        Walkers object.
    trial : trial object
        Trial wavefunction.

    Returns
    -------
    local_energy : np.ndarray
        Total, one-body and two-body energies.
    """
    assert hamiltonian.chunked

    nwalkers = walkers.Ghalfa.shape[0]
    nk = hamiltonian.nk
    nalpha = trial.nalpha
    nbeta = trial.nbeta
    nbasis = hamiltonian.nbasis

    ghalfa = walkers.Ghalfa.reshape(nwalkers, nk, nalpha, nk, nbasis)
    ghalfb = walkers.Ghalfb.reshape(nwalkers, nk, nbeta, nk, nbasis)
    ghalfaT = walkers.Ghalfa.transpose(0, 2, 1).reshape(nwalkers, nk, nbasis, nk, nalpha)
    ghalfbT = walkers.Ghalfb.transpose(0, 2, 1).reshape(nwalkers, nk, nbasis, nk, nbeta)

    diagGhalfa = numpy.zeros((nwalkers, nk, nalpha, nbasis), dtype=numpy.complex128)
    diagGhalfb = numpy.zeros((nwalkers, nk, nbeta, nbasis), dtype=numpy.complex128)
    for ik in range(nk):
        diagGhalfa[:, ik, :, :] = ghalfa[:, ik, :, ik, :]
        diagGhalfb[:, ik, :, :] = ghalfb[:, ik, :, ik, :]
    diagGhalfa = diagGhalfa.reshape(nwalkers, nk * nalpha * nbasis)
    diagGhalfb = diagGhalfb.reshape(nwalkers, nk * nbeta * nbasis)
    e1b = diagGhalfa.dot(trial._rH1a.ravel())
    e1b += diagGhalfb.dot(trial._rH1b.ravel())
    e1b /= nk
    e1b += hamiltonian.ecore

    ghalfa = ghalfa.transpose(1, 3, 0, 2, 4).copy() # nk, nk, nw, nalpha, nbasis
    ghalfb = ghalfb.transpose(1, 3, 0, 2, 4).copy() # nk, nk, nw, nbeta, nbasis
    ghalfaTcoul = ghalfaT.transpose(1, 3, 0, 2, 4).copy() # nk, nk, nw, nbasis, nalpha
    ghalfbTcoul = ghalfbT.transpose(1, 3, 0, 2, 4).copy() # nk, nk, nw, nbasis, nbeta
    ghalfaTx = ghalfaT.transpose(1, 3, 2, 4, 0).copy() # nk, nk, nbasis, nalpha, nw
    ghalfbTx = ghalfbT.transpose(1, 3, 2, 4, 0).copy() # nk, nk, nbasis, nbeta, nw

    ghalfa_send = ghalfa.copy()
    ghalfb_send = ghalfb.copy()
    ghalfaTcoul_send = ghalfaTcoul.copy()
    ghalfbTcoul_send = ghalfbTcoul.copy()
    ghalfaTx_send = ghalfaTx.copy()
    ghalfbTx_send = ghalfbTx.copy()

    ghalfa_recv = xp.zeros_like(ghalfa)
    ghalfb_recv = xp.zeros_like(ghalfb)
    ghalfaTcoul_recv = xp.zeros_like(ghalfaTcoul)
    ghalfbTcoul_recv = xp.zeros_like(ghalfbTcoul)
    ghalfaTx_recv = xp.zeros_like(ghalfaTx)
    ghalfbTx_recv = xp.zeros_like(ghalfbTx)

    handler = walkers.mpi_handler
    senders = handler.senders
    receivers = handler.receivers

    rchola_chunk = trial._rchola_chunk
    rcholb_chunk = trial._rcholb_chunk
    rcholbara_chunk = trial._rcholbara_chunk
    rcholbarb_chunk = trial._rcholbarb_chunk

    ecoul_send = kpt_symmchol_ecoul_kernel_uhf(
        rchola_chunk, rcholb_chunk, rcholbara_chunk, rcholbarb_chunk, ghalfa, ghalfb, ghalfaTcoul, ghalfbTcoul, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus
    )

    exx_send = kpt_symmchol_exx_kernel(rchola_chunk, rcholbara_chunk, ghalfa, ghalfaTx, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus) 
    exx_send += kpt_symmchol_exx_kernel(rcholb_chunk, rcholbarb_chunk, ghalfb, ghalfbTx, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus) 

    exx_recv = exx_send.copy()
    ecoul_recv = ecoul_send.copy()

    for _ in range(handler.ssize - 1):
        for isend, sender in enumerate(senders):
            if handler.srank == isend:
                handler.scomm.Send(ghalfa_send, dest=receivers[isend], tag=1)
                handler.scomm.Send(ghalfb_send, dest=receivers[isend], tag=2)
                handler.scomm.Send(ghalfaTcoul_send, dest=receivers[isend], tag=3)
                handler.scomm.Send(ghalfbTcoul_send, dest=receivers[isend], tag=4)
                handler.scomm.Send(ghalfaTx_send, dest=receivers[isend], tag=5)
                handler.scomm.Send(ghalfbTx_send, dest=receivers[isend], tag=6)
                handler.scomm.Send(ecoul_send, dest=receivers[isend], tag=7)
                handler.scomm.Send(exx_send, dest=receivers[isend], tag=8)
            elif handler.srank == receivers[isend]:
                sender = numpy.where(receivers == handler.srank)[0]
                handler.scomm.Recv(ghalfa_recv, source=sender, tag=1)
                handler.scomm.Recv(ghalfb_recv, source=sender, tag=2)
                handler.scomm.Recv(ghalfaTcoul_recv, source=sender, tag=3)
                handler.scomm.Recv(ghalfbTcoul_recv, source=sender, tag=4)
                handler.scomm.Recv(ghalfaTx_recv, source=sender, tag=5)
                handler.scomm.Recv(ghalfbTx_recv, source=sender, tag=6)
                handler.scomm.Recv(ecoul_recv, source=sender, tag=7)
                handler.scomm.Recv(exx_recv, source=sender, tag=8)
        handler.scomm.barrier()

    # prepare sending
        ecoul_send = ecoul_recv.copy()
        ecoul_send += kpt_symmchol_ecoul_kernel_uhf(
        rchola_chunk, rcholb_chunk, rcholbara_chunk, rcholbarb_chunk, ghalfa_recv, ghalfb_recv, ghalfaTcoul_recv, ghalfbTcoul_recv, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus
    )
        exx_send = exx_recv.copy()
        exx_send += kpt_symmchol_exx_kernel(rchola_chunk, rcholbara_chunk, ghalfa_recv, ghalfaTx_recv, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus)
        exx_send += kpt_symmchol_exx_kernel(rcholb_chunk, rcholbarb_chunk, ghalfb_recv, ghalfbTx_recv, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus)
        ghalfa_send = ghalfa_recv.copy()
        ghalfb_send = ghalfb_recv.copy()
        ghalfaTcoul_send = ghalfaTcoul_recv.copy()
        ghalfbTcoul_send = ghalfbTcoul_recv.copy()
        ghalfaTx_send = ghalfaTx_recv.copy()
        ghalfbTx_send = ghalfbTx_recv.copy()


    if len(senders) > 1:
        for isend, sender in enumerate(senders):
            if handler.srank == sender:  # sending 1 xshifted to 0 xshifted_buf
                handler.scomm.Send(ecoul_send, dest=receivers[isend], tag=1)
                handler.scomm.Send(exx_send, dest=receivers[isend], tag=2)
            elif handler.srank == receivers[isend]:
                sender = numpy.where(receivers == handler.srank)[0]
                handler.scomm.Recv(ecoul_recv, source=sender, tag=1)
                handler.scomm.Recv(exx_recv, source=sender, tag=2)
    e2b = ecoul_recv + exx_recv

    energy = xp.zeros((nwalkers, 3), dtype=numpy.complex128)
    energy[:, 0] = e1b + e2b
    energy[:, 1] = e1b
    energy[:, 2] = e2b

    return energy
