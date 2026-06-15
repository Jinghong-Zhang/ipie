import numpy
from math import ceil
from typing import TYPE_CHECKING

from ipie.config import config
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import synchronize
from ipie.utils.contract_gf_cgto import slice_cgto_kpq, slice_gf_kpq_k_qlis
from ipie.utils.cuquantum_backend import (
    NetworkOptions_optional as NetworkOptions,
    cutensornet_optional as cutensornet,
)

if TYPE_CHECKING:
    from ipie.hamiltonians.kpt_isdf_hamiltonian import KptISDF
    from ipie.trial_wavefunction.single_det_kpt import KptSingleDet
    from ipie.walkers.uhf_walkers import UHFWalkers


def construct_force_bias_batch_single_det_isdf(hamiltonian, walkers, rcgtoa, rcgtob):
    if walkers.rhf:
        Ghalfa = walkers.Ghalfa
        handle = cutensornet.create()
        network_opts = NetworkOptions(handle=handle)
        vbias_batch_real = 2.0 * cutensornet.contract(
            "Pi, Pr, Pg, wir -> wg",
            rcgtoa,
            hamiltonian.cgto,
            hamiltonian.cholM,
            Ghalfa.real,
            options=network_opts,
        )
        vbias_batch_imag = 2.0 * cutensornet.contract(
            "Pi, Pr, Pg, wir -> wg",
            rcgtoa,
            hamiltonian.cgto,
            hamiltonian.cholM,
            Ghalfa.imag,
            options=network_opts,
        )
        vbias_batch = xp.empty((walkers.nwalkers, hamiltonian.nchol), dtype=Ghalfa.dtype)
        vbias_batch.real = vbias_batch_real
        vbias_batch.imag = vbias_batch_imag
        cutensornet.destroy(handle)
        synchronize()

        return vbias_batch

    Ghalfa = walkers.Ghalfa
    Ghalfb = walkers.Ghalfb
    handle = cutensornet.create()
    network_opts = NetworkOptions(handle=handle)
    vbias_batch_real = cutensornet.contract(
        "Pi, Pr, Pg, wir -> wg",
        rcgtoa,
        hamiltonian.cgto,
        hamiltonian.cholM,
        Ghalfa.real,
        options=network_opts,
    ) + cutensornet.contract(
        "Pi, Pr, Pg, wir -> wg",
        rcgtob,
        hamiltonian.cgto,
        hamiltonian.cholM,
        Ghalfb.real,
        options=network_opts,
    )
    vbias_batch_imag = cutensornet.contract(
        "Pi, Pr, Pg, wir -> wg",
        rcgtoa,
        hamiltonian.cgto,
        hamiltonian.cholM,
        Ghalfa.imag,
        options=network_opts,
    ) + cutensornet.contract(
        "Pi, Pr, Pg, wir -> wg",
        rcgtob,
        hamiltonian.cgto,
        hamiltonian.cholM,
        Ghalfb.imag,
        options=network_opts,
    )
    vbias_batch = xp.empty((walkers.nwalkers, hamiltonian.nchol), dtype=Ghalfa.dtype)
    vbias_batch.real = vbias_batch_real
    vbias_batch.imag = vbias_batch_imag
    cutensornet.destroy(handle)
    synchronize()
    return vbias_batch


def contract_qkPp_kPr_qkwpr_to_qwP_cupy(
    rcgto_qkPp, halfrot_cgto, g_qkwpr, max_mem=4.0
):
    """
    Evaluate ``qkPp,kPr,qkwpr->qwP`` with chunked CuPy GEMMs.

    The ISDF grid dimension is chunked explicitly so the intermediate
    ``qkPwr`` tensor stays bounded for large ``nk`` and ``nisdf``.
    """
    nq, nk, nisdf, nocc = rcgto_qkPp.shape
    nwalkers = g_qkwpr.shape[2]
    nbasis = halfrot_cgto.shape[-1]
    dtype = xp.result_type(rcgto_qkPp, halfrot_cgto, g_qkwpr)
    itemsize = xp.dtype(dtype).itemsize
    max_mem_bytes = int(max(float(max_mem), 0.01) * 1024**3)
    out = xp.empty((nq, nwalkers, nisdf), dtype=dtype)

    g_bytes_per_walker = nq * nk * nocc * nbasis * itemsize
    min_t_bytes_per_isdf = nq * nk * nbasis * itemsize
    walker_chunk = max(
        1,
        min(
            nwalkers,
            max_mem_bytes // max(g_bytes_per_walker + min_t_bytes_per_isdf, 1),
        ),
    )

    for wstart in range(0, nwalkers, walker_chunk):
        wstop = min(wstart + walker_chunk, nwalkers)
        wchunk = wstop - wstart
        g_mat = xp.ascontiguousarray(
            g_qkwpr[:, :, wstart:wstop, :, :]
            .transpose(0, 1, 3, 2, 4)
            .reshape(nq, nk, nocc, wchunk * nbasis)
        )

        available = max(max_mem_bytes - g_mat.nbytes, min_t_bytes_per_isdf)
        # Keep one q,k,P,w,r intermediate resident. It is overwritten in-place
        # during the weighted r-reduction.
        isdf_chunk = max(
            1,
            min(nisdf, available // max(nq * nk * wchunk * nbasis * itemsize, 1)),
        )

        for pstart in range(0, nisdf, isdf_chunk):
            pstop = min(pstart + isdf_chunk, nisdf)
            pchunk = pstop - pstart
            rcgto_chunk = xp.ascontiguousarray(rcgto_qkPp[:, :, pstart:pstop, :])
            contracted = xp.empty((nq, nk, pchunk, wchunk * nbasis), dtype=dtype)
            xp.matmul(rcgto_chunk, g_mat, out=contracted)
            contracted = contracted.reshape(nq, nk, pchunk, wchunk, nbasis)
            contracted *= halfrot_cgto[None, :, pstart:pstop, None, :]
            out[:, wstart:wstop, pstart:pstop] = contracted.sum(axis=(1, 4)).transpose(
                0, 2, 1
            )
            del rcgto_chunk, contracted
        del g_mat
    return out


def contract_qkPp_kPr_qkwpr_to_qwP_old_cuquantum(
    rcgto_qkPp, halfrot_cgto, g_qkwpr, network_opts
):
    return cutensornet.contract(
        "qkPp, kPr, qkwpr -> qwP",
        rcgto_qkPp,
        halfrot_cgto,
        g_qkwpr,
        options=network_opts,
    )


def contract_qwP_qPg_to_wgq_old_cuquantum(X_qwP, cholM_qPg, network_opts):
    return cutensornet.contract("qwP, qPg -> wgq", X_qwP, cholM_qPg, options=network_opts)


def _contract_force_bias_x(rcgto_qkPp, halfrot_cgto, g_qkwpr, max_mem):
    return contract_qkPp_kPr_qkwpr_to_qwP_cupy(
        rcgto_qkPp, halfrot_cgto, g_qkwpr, max_mem=max_mem
    )


def _contract_force_bias_chol(X_qwP, cholM_qPg):
    return xp.matmul(X_qwP, cholM_qPg).transpose(1, 2, 0)


def _contract_force_bias_spin_sum(
    rcgtoa_qkPp, ga_qkwpr, rcgtob_qkPp, gb_qkwpr, halfrot_cgto, max_mem
):
    rcgto_qkPp = xp.concatenate((rcgtoa_qkPp, rcgtob_qkPp), axis=-1)
    g_qkwpr = xp.concatenate((ga_qkwpr, gb_qkwpr), axis=3)
    return _contract_force_bias_x(rcgto_qkPp, halfrot_cgto, g_qkwpr, max_mem)


def construct_force_bias_kptisdf_batch_single_det(
    hamiltonian: "KptISDF", walkers: "UHFWalkers", trial: "KptSingleDet", max_mem=4.0
):
    if walkers.rhf:
        if config.get_option("use_gpu"):
            nwalkers = walkers.nwalkers
            vbias_plus = xp.zeros(
                (nwalkers, hamiltonian.nchol, hamiltonian.unique_nk), dtype=numpy.complex128
            )
            vbias_minus = xp.zeros(
                (nwalkers, hamiltonian.nchol, hamiltonian.unique_nk), dtype=numpy.complex128
            )
            Ghalfa_reshape = walkers.Ghalfa.reshape(
                nwalkers,
                hamiltonian.nk,
                trial.nalpha,
                hamiltonian.nk,
                hamiltonian.halfrot_cgto.shape[-1],
            )

            mem_cost_Sset = (
                max(nwalkers * hamiltonian.nbasis, hamiltonian.nisdf)
                * len(hamiltonian.Sset)
                * hamiltonian.nk
                * (trial.nalpha)
                * 3
                * 16
                / (1024**3)
            )
            mem_cost_Qplus = (
                max(nwalkers * hamiltonian.nbasis, hamiltonian.nisdf)
                * len(hamiltonian.Qplus)
                * hamiltonian.nk
                * (trial.nalpha)
                * 4
                * 16
                / (1024**3)
            )

            num_nq_chunks_Sset = max(1, ceil(mem_cost_Sset / max_mem))
            nq_chunk_Sset_size = ceil(len(hamiltonian.Sset) / num_nq_chunks_Sset)
            nq_left = len(hamiltonian.Sset)
            if len(hamiltonian.Sset) > 0:
                for i in range(num_nq_chunks_Sset):
                    nq_chunk = min(nq_left, nq_chunk_Sset_size)
                    nq_left -= nq_chunk
                    q_sls = hamiltonian.Sset[
                        i * nq_chunk_Sset_size : i * nq_chunk_Sset_size + nq_chunk
                    ]
                    ga_kmq = slice_gf_kpq_k_qlis(Ghalfa_reshape, q_sls, hamiltonian.ikmq_mat)
                    rcgtoa_kmq = slice_cgto_kpq(trial._rcgtoa, hamiltonian.ikmq_mat, q_sls)
                    X_wPa = _contract_force_bias_x(
                        rcgtoa_kmq.conj(),
                        hamiltonian.halfrot_cgto,
                        ga_kmq,
                        max_mem,
                    )
                    L_q = hamiltonian.cholM[
                        i * nq_chunk_Sset_size : i * nq_chunk_Sset_size + nq_chunk
                    ]
                    vbias_plus[
                        :, :, i * nq_chunk_Sset_size : i * nq_chunk_Sset_size + nq_chunk
                    ] += 2.0j * _contract_force_bias_chol(X_wPa, L_q)

            num_nq_chunks_Qplus = max(1, ceil(mem_cost_Qplus / max_mem))
            nq_chunk_Qplus_size = ceil(len(hamiltonian.Qplus) / num_nq_chunks_Qplus)
            nq_left = len(hamiltonian.Qplus)
            if len(hamiltonian.Qplus) > 0:
                for i in range(num_nq_chunks_Qplus):
                    nq_chunk = min(nq_left, nq_chunk_Qplus_size)
                    nq_left -= nq_chunk
                    q_sls = hamiltonian.Qplus[
                        i * nq_chunk_Qplus_size : i * nq_chunk_Qplus_size + nq_chunk
                    ]
                    ga_kmq = slice_gf_kpq_k_qlis(Ghalfa_reshape, q_sls, hamiltonian.ikmq_mat)
                    rcgtoa_kmq = slice_cgto_kpq(trial._rcgtoa, hamiltonian.ikmq_mat, q_sls)
                    ga_kpq = slice_gf_kpq_k_qlis(Ghalfa_reshape, q_sls, hamiltonian.ikpq_mat)
                    rcgtoa_kpq = slice_cgto_kpq(trial._rcgtoa, hamiltonian.ikpq_mat, q_sls)
                    X_wPa = _contract_force_bias_x(
                        rcgtoa_kmq.conj(),
                        hamiltonian.halfrot_cgto,
                        ga_kmq,
                        max_mem,
                    )
                    Y_wPa = _contract_force_bias_x(
                        rcgtoa_kpq.conj(),
                        hamiltonian.halfrot_cgto,
                        ga_kpq,
                        max_mem,
                    )
                    L_q = hamiltonian.cholM[
                        i * nq_chunk_Qplus_size
                        + len(hamiltonian.Sset) : i * nq_chunk_Qplus_size
                        + nq_chunk
                        + len(hamiltonian.Sset)
                    ]
                    v1 = _contract_force_bias_chol(X_wPa, L_q)
                    v2 = _contract_force_bias_chol(Y_wPa, L_q.conj())
                    vbias_plus[
                        :,
                        :,
                        i * nq_chunk_Qplus_size
                        + len(hamiltonian.Sset) : i * nq_chunk_Qplus_size
                        + nq_chunk
                        + len(hamiltonian.Sset),
                    ] += (
                        1j * xp.sqrt(2) * (v1 + v2)
                    )
                    vbias_minus[
                        :,
                        :,
                        i * nq_chunk_Qplus_size
                        + len(hamiltonian.Sset) : i * nq_chunk_Qplus_size
                        + nq_chunk
                        + len(hamiltonian.Sset),
                    ] += (
                        1.0 * xp.sqrt(2) * (v1 - v2)
                    )
            synchronize()
            return vbias_plus, vbias_minus
        else:
            pass
    else:
        if config.get_option("use_gpu"):
            nwalkers = walkers.nwalkers
            vbias_plus = xp.zeros(
                (nwalkers, hamiltonian.nchol, hamiltonian.unique_nk), dtype=numpy.complex128
            )
            vbias_minus = xp.zeros(
                (nwalkers, hamiltonian.nchol, hamiltonian.unique_nk), dtype=numpy.complex128
            )
            Ghalfa_reshape = walkers.Ghalfa.reshape(
                nwalkers,
                hamiltonian.nk,
                trial.nalpha,
                hamiltonian.nk,
                hamiltonian.halfrot_cgto.shape[-1],
            )
            Ghalfb_reshape = walkers.Ghalfb.reshape(
                nwalkers,
                hamiltonian.nk,
                trial.nbeta,
                hamiltonian.nk,
                hamiltonian.halfrot_cgto.shape[-1],
            )

            mem_cost_Sset = (
                max(nwalkers * hamiltonian.nbasis, hamiltonian.nisdf)
                * len(hamiltonian.Sset)
                * hamiltonian.nk
                * (trial.nalpha + trial.nbeta)
                * 16
                * 2
                / (1024**3)
            )
            mem_cost_Qplus = (
                max(nwalkers * hamiltonian.nbasis, hamiltonian.nisdf)
                * len(hamiltonian.Qplus)
                * hamiltonian.nk
                * (trial.nalpha + trial.nbeta)
                * 16
                * 2
                / (1024**3)
            )

            num_nq_chunks_Sset = max(1, ceil(mem_cost_Sset / max_mem))
            nq_chunk_Sset_size = ceil(len(hamiltonian.Sset) / num_nq_chunks_Sset)
            nq_left = len(hamiltonian.Sset)
            if len(hamiltonian.Sset) > 0:
                for i in range(num_nq_chunks_Sset):
                    nq_chunk = min(nq_left, nq_chunk_Sset_size)
                    nq_left -= nq_chunk
                    q_sls = hamiltonian.Sset[
                        i * nq_chunk_Sset_size : i * nq_chunk_Sset_size + nq_chunk
                    ]
                    ga_kmq = slice_gf_kpq_k_qlis(Ghalfa_reshape, q_sls, hamiltonian.ikmq_mat)
                    gb_kmq = slice_gf_kpq_k_qlis(Ghalfb_reshape, q_sls, hamiltonian.ikmq_mat)
                    rcgtoa_kmq = slice_cgto_kpq(trial._rcgtoa, hamiltonian.ikmq_mat, q_sls)
                    rcgtob_kmq = slice_cgto_kpq(trial._rcgtob, hamiltonian.ikmq_mat, q_sls)
                    X_wP = _contract_force_bias_spin_sum(
                        rcgtoa_kmq.conj(),
                        ga_kmq,
                        rcgtob_kmq.conj(),
                        gb_kmq,
                        hamiltonian.halfrot_cgto,
                        max_mem,
                    )
                    L_q = hamiltonian.cholM[
                        i * nq_chunk_Sset_size : i * nq_chunk_Sset_size + nq_chunk
                    ]
                    vbias_plus[
                        :, :, i * nq_chunk_Sset_size : i * nq_chunk_Sset_size + nq_chunk
                    ] += 1j * _contract_force_bias_chol(X_wP, L_q)

            num_nq_chunks_Qplus = max(1, ceil(mem_cost_Qplus / max_mem))
            nq_chunk_Qplus_size = ceil(len(hamiltonian.Qplus) / num_nq_chunks_Qplus)
            nq_left = len(hamiltonian.Qplus)
            if len(hamiltonian.Qplus) > 0:
                for i in range(num_nq_chunks_Qplus):
                    nq_chunk = min(nq_left, nq_chunk_Qplus_size)
                    nq_left -= nq_chunk
                    q_sls = hamiltonian.Qplus[
                        i * nq_chunk_Qplus_size : i * nq_chunk_Qplus_size + nq_chunk
                    ]
                    ga_kmq = slice_gf_kpq_k_qlis(Ghalfa_reshape, q_sls, hamiltonian.ikmq_mat)
                    gb_kmq = slice_gf_kpq_k_qlis(Ghalfb_reshape, q_sls, hamiltonian.ikmq_mat)
                    rcgtoa_kmq = slice_cgto_kpq(trial._rcgtoa, hamiltonian.ikmq_mat, q_sls)
                    rcgtob_kmq = slice_cgto_kpq(trial._rcgtob, hamiltonian.ikmq_mat, q_sls)
                    ga_kpq = slice_gf_kpq_k_qlis(Ghalfa_reshape, q_sls, hamiltonian.ikpq_mat)
                    gb_kpq = slice_gf_kpq_k_qlis(Ghalfb_reshape, q_sls, hamiltonian.ikpq_mat)
                    rcgtoa_kpq = slice_cgto_kpq(trial._rcgtoa, hamiltonian.ikpq_mat, q_sls)
                    rcgtob_kpq = slice_cgto_kpq(trial._rcgtob, hamiltonian.ikpq_mat, q_sls)
                    X_wP = _contract_force_bias_spin_sum(
                        rcgtoa_kmq.conj(),
                        ga_kmq,
                        rcgtob_kmq.conj(),
                        gb_kmq,
                        hamiltonian.halfrot_cgto,
                        max_mem,
                    )
                    Y_wP = _contract_force_bias_spin_sum(
                        rcgtoa_kpq.conj(),
                        ga_kpq,
                        rcgtob_kpq.conj(),
                        gb_kpq,
                        hamiltonian.halfrot_cgto,
                        max_mem,
                    )
                    L_q = hamiltonian.cholM[
                        i * nq_chunk_Qplus_size
                        + len(hamiltonian.Sset) : i * nq_chunk_Qplus_size
                        + nq_chunk
                        + len(hamiltonian.Sset)
                    ]
                    v1 = _contract_force_bias_chol(X_wP, L_q)
                    v2 = _contract_force_bias_chol(Y_wP, L_q.conj())
                    vbias_plus[
                        :,
                        :,
                        i * nq_chunk_Qplus_size
                        + len(hamiltonian.Sset) : i * nq_chunk_Qplus_size
                        + nq_chunk
                        + len(hamiltonian.Sset),
                    ] += (
                        0.5j * xp.sqrt(2) * (v1 + v2)
                    )
                    vbias_minus[
                        :,
                        :,
                        i * nq_chunk_Qplus_size
                        + len(hamiltonian.Sset) : i * nq_chunk_Qplus_size
                        + nq_chunk
                        + len(hamiltonian.Sset),
                    ] += (
                        0.5 * xp.sqrt(2) * (v1 - v2)
                    )
            synchronize()
            return vbias_plus, vbias_minus
        else:
            pass


def construct_force_bias_batch_single_det_isdf_chunked(
    hamiltonian, walkers, rcgtoa, rcgtob, handler
):
    assert hamiltonian.chunked
    assert xp.isrealobj(hamiltonian.cholM_chunk)

    Ghalfa = walkers.Ghalfa
    Ghalfb = walkers.Ghalfb

    chol_idxs_chunk = hamiltonian.chol_idxs_chunk

    Ghalfa_recv = xp.ascontiguousarray(xp.zeros_like(Ghalfa))
    Ghalfb_recv = xp.ascontiguousarray(xp.zeros_like(Ghalfb))

    Ghalfa_send = Ghalfa.copy()
    Ghalfb_send = Ghalfb.copy()

    srank = handler.scomm.rank

    vbias_batch_real_recv = xp.zeros((hamiltonian.nchol, walkers.nwalkers))
    vbias_batch_imag_recv = xp.zeros((hamiltonian.nchol, walkers.nwalkers))

    vbias_batch_real_send = xp.zeros((hamiltonian.nchol, walkers.nwalkers))
    vbias_batch_imag_send = xp.zeros((hamiltonian.nchol, walkers.nwalkers))

    handle = cutensornet.create()
    network_opts = NetworkOptions(handle=handle)
    vbias_batch_real_send[chol_idxs_chunk, :] = cutensornet.contract(
        "Pi, Pr, Pg, wir -> gw",
        rcgtoa,
        hamiltonian.cgto,
        hamiltonian.cholM_chunk,
        Ghalfa.real,
        options=network_opts,
    ) + cutensornet.contract(
        "Pi, Pr, Pg, wir -> gw",
        rcgtob,
        hamiltonian.cgto,
        hamiltonian.cholM_chunk,
        Ghalfb.real,
        options=network_opts,
    )
    vbias_batch_imag_send[chol_idxs_chunk, :] = cutensornet.contract(
        "Pi, Pr, Pg, wir -> gw",
        rcgtoa,
        hamiltonian.cgto,
        hamiltonian.cholM_chunk,
        Ghalfa.imag,
        options=network_opts,
    ) + cutensornet.contract(
        "Pi, Pr, Pg, wir -> gw",
        rcgtob,
        hamiltonian.cgto,
        hamiltonian.cholM_chunk,
        Ghalfb.imag,
        options=network_opts,
    )

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

        vbias_batch_real_send = vbias_batch_real_recv.copy()
        vbias_batch_imag_send = vbias_batch_imag_recv.copy()
        vbias_batch_real_send[chol_idxs_chunk, :] = cutensornet.contract(
            "Pi, Pr, Pg, wir -> gw",
            rcgtoa,
            hamiltonian.cgto,
            hamiltonian.cholM_chunk,
            Ghalfa_recv.real,
            options=network_opts,
        ) + cutensornet.contract(
            "Pi, Pr, Pg, wir -> gw",
            rcgtob,
            hamiltonian.cgto,
            hamiltonian.cholM_chunk,
            Ghalfb_recv.real,
            options=network_opts,
        )
        vbias_batch_imag_send[chol_idxs_chunk, :] = cutensornet.contract(
            "Pi, Pr, Pg, wir -> gw",
            rcgtoa,
            hamiltonian.cgto,
            hamiltonian.cholM_chunk,
            Ghalfa_recv.imag,
            options=network_opts,
        ) + cutensornet.contract(
            "Pi, Pr, Pg, wir -> gw",
            rcgtob,
            hamiltonian.cgto,
            hamiltonian.cholM_chunk,
            Ghalfb_recv.imag,
            options=network_opts,
        )
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
