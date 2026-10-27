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

from ipie.addons.thermal.thermofield.utils.linalg import (
    inv_stack,
    principal_log_phase,
    qdt_left_restratify_stack,
    qdt_right_restratify_stack,
    restratify_left_batched,
    restratify_right_batched,
    stabilized_inverse_one_plus_batched,
)
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import to_host


def cyclic_average_hubbard(
    walkers, trial, propagator, nslices, stride, measure_walker_fn, stab_every=20, mem_budget=1.5e8
):
    r"""Cyclic averaging over the recorded Hubbard fields of every walker.

    The slice propagators are rebuilt from `walkers.field_hist` with
    `propagator.construct_slice_propagators`; the histories follow the
    population-control ancestry. Walkers are not changed and no inverse
    propagators are used.

    Parameters
    ----------
    walkers : :class:`ThermofieldWalkers`
        Walkers with the field history of the whole walk.
    trial : :class:`ThermofieldCPTrial`
        Trial (sets nbasis).
    propagator : :class:`ThermofieldHubbardCP`
        Propagator that recorded the fields.
    nslices : int
        Number of slices of the walk.
    stride : int
        Cuts at 0, stride, 2 stride, ...
    measure_walker_fn : callable
        Called as measure_walker_fn(iw, G, cut) with the particle contraction
        :math:`G = I - (I + \Delta_{cut})^{-1}`; ipie's density matrix is G.T.
    stab_every : int
        Restratification interval. Default 20.
    mem_budget : float
        Bytes for the suffix snapshots. Default 1.5e8.

    Returns
    -------
    worst_dev : float
        Largest cyclic-invariance violation of the trace log determinant.
    ld0 : :class:`numpy.ndarray`
        Trace log determinant at cut 0 per walker.
    """
    if not isinstance(nslices, (int, numpy.integer)) or nslices < 1:
        raise ValueError("nslices must be a positive integer.")
    if not isinstance(stride, (int, numpy.integer)) or stride < 1:
        raise ValueError("stride must be a positive integer.")
    if not isinstance(stab_every, (int, numpy.integer)) or stab_every < 1:
        raise ValueError("stab_every must be a positive integer.")
    if not numpy.isfinite(mem_budget) or mem_budget <= 0:
        raise ValueError("mem_budget must be finite and positive.")
    if not hasattr(walkers, "field_hist") or walkers.field_hist.shape != (
        walkers.nwalkers,
        nslices,
        propagator.nfields,
    ):
        raise ValueError("The requested complete Hubbard field history is unavailable.")

    def slice_b(l, ids):
        return propagator.construct_slice_propagators(walkers.field_hist[ids, l])

    callback = measure_walker_fn
    counts = numpy.zeros(walkers.nwalkers, dtype=int)
    if propagator.discrete:

        def callback(iw, green, cut):
            counts[iw] += 1
            measure_walker_fn(iw, green, cut)

    result = cyclic_average_engine(
        walkers.weight,
        nslices,
        trial.nbasis,
        stride,
        slice_b,
        callback,
        stab_every=stab_every,
        mem_budget=mem_budget,
        check_phase=True,
    )
    if propagator.discrete:
        expected = len(range(0, nslices, stride))
        if numpy.any(counts[walkers.weight != 0] != expected):
            raise FloatingPointError("Incomplete discrete-field cyclic reconstruction.")
    return result


def build_slice_propagators(VHS, BH1, exp_nmax, spin_shared=False):
    r"""Slice propagators :math:`B_s = B_{H1,s} e^{V_{HS}} B_{H1,s}` for a stack of VHS.

    The truncated Taylor series and product order of `apply_exponential`
    followed by BH1 @ BV @ BH1, so the propagators are those of the walk.

    Parameters
    ----------
    VHS : :class:`numpy.ndarray`
        Two-body exponents, shape (n, nbasis, nbasis).
    BH1 : :class:`numpy.ndarray`
        One-body propagators, shape (2, nbasis, nbasis).
    exp_nmax : int
        Order of the Taylor series.
    spin_shared : bool
        Copy spin 0 to spin 1; valid when BH1[0] and BH1[1] are bitwise equal.
        Default False.

    Returns
    -------
    B : :class:`numpy.ndarray`
        Shape (n, 2, nbasis, nbasis).
    """
    VHS = numpy.ascontiguousarray(VHS, dtype=numpy.complex128)
    n, M = VHS.shape[0], VHS.shape[-1]
    BV = numpy.empty((n, M, M), dtype=numpy.complex128)
    BV[:] = numpy.eye(M)
    if exp_nmax >= 1:
        # The first-order term VHS @ I is VHS exactly.
        Temp = VHS.copy()
        BV += Temp
        Temp2 = numpy.empty_like(Temp)
        for k in range(2, exp_nmax + 1):
            numpy.matmul(VHS, Temp, out=Temp2)
            Temp2 /= k
            BV += Temp2
            Temp, Temp2 = Temp2, Temp
    out = numpy.empty((n, 2, M, M), dtype=numpy.complex128)
    for s in range(2):
        if s == 1 and spin_shared:
            out[:, 1] = out[:, 0]
            break
        out[:, s] = numpy.matmul(BH1[s], BV) @ BH1[s]  # Symmetric Trotter split.
    return out


def cyclic_average_engine(
    weight,
    nslices,
    nbasis,
    stride,
    slice_b_fn,
    measure_walker_fn,
    stab_every=20,
    mem_budget=1.5e8,
    spin_shared_fn=None,
    check_phase=False,
):
    r"""Measure at every cyclic cut of recorded paths over [0, beta].

    The trace :math:`\det(I + B_L \cdots B_1)` is invariant under cyclic
    shifts, so the path can be cut at any slice w,
    :math:`\Delta_w = P_w S_w` with :math:`P_w = B_w \cdots B_1` and
    :math:`S_w = B_L \cdots B_{w+1}`. A backward sweep stores the suffixes S at
    the cuts and a forward sweep builds the prefixes P, both in QDT form, so
    no inverse propagator is needed.

    Parameters
    ----------
    weight : :class:`numpy.ndarray`
        Walker weights, shape (nwalkers,); walkers with zero weight are skipped.
    nslices : int
        Number of slices L.
    nbasis : int
        Number of basis functions.
    stride : int
        Cuts at w = 0, stride, 2 stride, ... (w = 0 is the endpoint).
    slice_b_fn : callable
        slice_b_fn(l, ids) gives the propagators of slice l applied to walkers
        ids, shape (len(ids), 2, nbasis, nbasis). Must be deterministic: it
        may be evaluated once per sweep.
    measure_walker_fn : callable
        Called as measure_walker_fn(iw, G, w) with
        :math:`G_s = I - (I + P_w S_w)^{-1}`, shape (2, nbasis, nbasis).
    stab_every : int
        Restratification interval of both sweeps. Default 20.
    mem_budget : float
        Bytes for the suffix snapshots; sets the walker chunk size. A chunk
        whose slice propagators also fit is built once for both sweeps.
        Default 1.5e8.
    spin_shared_fn : callable
        spin_shared_fn(ids) is True when the spin-0 and spin-1 propagators of
        walkers ids are bitwise equal over the whole path; spin 1 is then
        copied from spin 0. Default None.
    check_phase : bool
        Include the phase in the cyclic-invariance check. Default False.

    Returns
    -------
    worst_dev : float
        Largest cyclic-invariance violation of the trace log determinant over
        walkers and cuts.
    ld0 : :class:`numpy.ndarray`
        Trace log determinant at cut 0 per walker (NaN for skipped walkers);
        compared with the endpoint of the propagated walker it checks the
        reconstruction.
    """
    nw = len(weight)
    nb = nbasis
    positions = list(range(0, nslices, stride))
    posset = set(positions)
    alive = numpy.flatnonzero(weight != 0.0)
    ld0_out = numpy.full(nw, numpy.nan, dtype=numpy.complex128)
    if alive.size == 0:
        return 0.0, ld0_out
    per_snap = 2.0 * (2 * nb * nb + nb) * 16.0
    chunk = max(1, min(alive.size, int(mem_budget / (len(positions) * per_snap + 1.0))))
    eye = numpy.eye(nb)
    worst_dev = 0.0
    for c0 in range(0, alive.size, chunk):
        ids = alive[c0 : c0 + chunk]
        n = ids.size
        shared = bool(spin_shared_fn(ids)) if spin_shared_fn is not None else False
        nspin = 1 if shared else 2
        # Stacks are spin-major, (nspin, n, M, M), so each spin block is a
        # contiguous batch for the stacked GEMM/LAPACK calls.
        cache = None
        if n * nslices * nspin * nb * nb * 16.0 <= mem_budget:
            cache = numpy.empty((nslices, nspin, n, nb, nb), dtype=numpy.complex128)

        def slice_b(l, fill):
            if cache is None:
                return slice_b_fn(l, ids).transpose(1, 0, 2, 3)[:nspin]
            if fill:
                cache[l] = slice_b_fn(l, ids).transpose(1, 0, 2, 3)[:nspin]
            return cache[l]

        # Backward sweep: suffixes, stored at the cuts.
        Sq = numpy.tile(eye.astype(numpy.complex128), (nspin, n, 1, 1))
        Sd = numpy.zeros((nspin, n, nb))
        St = numpy.tile(eye.astype(numpy.complex128), (nspin, n, 1, 1))
        snaps = {}
        for l in range(nslices - 1, -1, -1):
            if (l + 1) % stab_every == 0:
                for sp in range(nspin):
                    Sq[sp], Sd[sp], St[sp] = qdt_right_restratify_stack(Sq[sp], Sd[sp], St[sp])
            Bl = slice_b(l, fill=True)
            for sp in range(nspin):
                St[sp] = St[sp] @ Bl[sp]
            if l in posset:
                snaps[l] = (Sq.copy(), Sd.copy(), St.copy())
        # Forward sweep: prefixes and the contraction at every cut.
        Pq = numpy.tile(eye.astype(numpy.complex128), (nspin, n, 1, 1))
        Pd = numpy.zeros((nspin, n, nb))
        Pt = numpy.tile(eye.astype(numpy.complex128), (nspin, n, 1, 1))
        ld_ref = numpy.full(n, numpy.nan, dtype=numpy.complex128)
        for w in positions:
            Qs, Ds, Ts = snaps.pop(w)
            lo = numpy.zeros(n, dtype=numpy.complex128)
            G = numpy.empty((n, 2, nb, nb), dtype=numpy.complex128)
            for sp in range(nspin):
                with numpy.errstate(divide="ignore", invalid="ignore"):
                    sgn_t, log_t = numpy.linalg.slogdet(Ts[sp])
                    ldv = log_t + numpy.log(sgn_t)
                # The engine works on the host; the kernels run on the backend.
                ld, inv = stabilized_inverse_one_plus_batched(
                    Pq[sp],
                    Pd[sp],
                    Pt[sp] @ Qs[sp],
                    Ds[sp],
                    Vinv=inv_stack(xp.asarray(Ts[sp])),
                    log_det_V=xp.asarray(ldv),
                )
                ld, inv = to_host(ld), to_host(inv)
                lo += ld
                G[:, sp] = eye - inv
            if shared:
                # Same association as the two-spin sum (0 + ld_0) + ld_1.
                lo += ld
                G[:, 1] = G[:, 0]
            good = numpy.isfinite(lo) & numpy.isfinite(G).all(axis=(1, 2, 3))
            if w == 0:
                ld0_out[ids[good]] = lo[good]
            fresh = numpy.isnan(ld_ref) & good
            ld_ref[fresh] = lo[fresh]
            seen = good & ~fresh
            if seen.any():
                if check_phase:
                    deviation = principal_log_phase(lo[seen] - ld_ref[seen])
                    worst_dev = max(worst_dev, float(numpy.max(numpy.abs(deviation))))
                else:
                    worst_dev = max(
                        worst_dev, float(numpy.max(numpy.abs((lo[seen] - ld_ref[seen]).real)))
                    )
            for j in numpy.flatnonzero(good):
                measure_walker_fn(int(ids[j]), G[j], w)
            for l in range(w, min(w + stride, nslices)):
                Bl = slice_b(l, fill=False)
                for sp in range(nspin):
                    Pq[sp] = Bl[sp] @ Pq[sp]
                if (l + 1) % stab_every == 0:
                    for sp in range(nspin):
                        Pq[sp], Pd[sp], Pt[sp] = qdt_left_restratify_stack(Pq[sp], Pd[sp], Pt[sp])
    return worst_dev, ld0_out


def cyclic_average_full_path(
    weight, B_hist, stride, measure_walker_fn, stab_every=20, mem_budget=1.5e8
):
    """Cyclic averaging from stored slice propagators.

    O(nbasis^2 L) memory per walker, for small basis sets; see
    :func:`cyclic_average_fields` for large ones.

    Parameters
    ----------
    weight : :class:`numpy.ndarray`
        Walker weights, shape (nwalkers,).
    B_hist : :class:`numpy.ndarray`
        Slice propagators, shape (nwalkers, nslices, 2, nbasis, nbasis).
    stride, measure_walker_fn, stab_every, mem_budget :
        As in :func:`cyclic_average_engine`.

    Returns
    -------
    worst_dev, ld0 :
        As in :func:`cyclic_average_engine`.
    """
    return cyclic_average_engine(
        weight,
        B_hist.shape[1],
        B_hist.shape[-1],
        stride,
        lambda l, ids: B_hist[ids, l],
        measure_walker_fn,
        stab_every=stab_every,
        mem_budget=mem_budget,
        spin_shared_fn=lambda ids: numpy.array_equal(B_hist[ids, :, 0], B_hist[ids, :, 1]),
    )


def cyclic_average_fields(
    weight,
    x_hist,
    propagator,
    hamiltonian,
    stride,
    measure_walker_fn,
    stab_every=20,
    mem_budget=1.5e8,
):
    """Cyclic averaging with slice propagators rebuilt from recorded shifted fields.

    O(nfields L) memory per walker instead of O(nbasis^2 L), for large basis
    sets. Every slice propagator is rebuilt with the propagator's own
    `construct_VHS` and Taylor series, as in the walk, so the rebuild is
    exact; ld0 against the propagated endpoint checks it every block. The
    rebuild runs once per slice and sweep unless the propagators of a chunk
    fit in `mem_budget`.

    Parameters
    ----------
    weight : :class:`numpy.ndarray`
        Walker weights, shape (nwalkers,).
    x_hist : :class:`numpy.ndarray`
        Shifted fields, shape (nwalkers, nslices, nfields).
    propagator : :class:`ThermofieldPhaseless`
        Propagator of the walk.
    hamiltonian :
        Hamiltonian.
    stride, measure_walker_fn, stab_every, mem_budget :
        As in :func:`cyclic_average_engine`.

    Returns
    -------
    worst_dev, ld0 :
        As in :func:`cyclic_average_engine`.
    """
    nb = hamiltonian.nbasis
    BH1 = to_host(propagator.BH1)
    bh1_same = numpy.array_equal(BH1[0], BH1[1])

    def slice_b_fn(l, ids):
        x = numpy.ascontiguousarray(x_hist[ids, l].T)  # Shape (nfields, n).
        VHS = propagator.construct_VHS(hamiltonian, x)  # Shape (n, nbasis, nbasis).
        return build_slice_propagators(VHS, BH1, propagator.exp_nmax, spin_shared=bh1_same)

    return cyclic_average_engine(
        weight,
        x_hist.shape[1],
        nb,
        stride,
        slice_b_fn,
        measure_walker_fn,
        stab_every=stab_every,
        mem_budget=mem_budget,
        spin_shared_fn=lambda ids: bh1_same,
    )


def cyclic_average_coarse(
    weight, Bg, measure_batch_fn, stab_every=1, mem_budget=2.0e10, check_phase=False
):
    r"""Cyclic averaging from stored products of consecutive slice propagators.

    Instead of keeping or rebuilding every slice propagator, the driver
    accumulates the product of the s slices between consecutive cuts while it
    propagates (`ThermofieldPhaseless.group_accum`) and stores the
    K = ceil(L / s) group products. The two sweeps of
    :func:`cyclic_average_engine` then run over the K coarse propagators, with
    cuts at the group boundaries (cut 0 is the endpoint),

    .. math::

        \Delta_k = P_k S_k, \qquad P_k = B_{g, k-1} \cdots B_{g, 0}, \qquad
        S_k = B_{g, K-1} \cdots B_{g, k},

    and :math:`\det(I + \Delta_k)` is the same trace weight for every k.

    Parameters
    ----------
    weight : :class:`numpy.ndarray`
        Walker weights on the host, shape (nwalkers,); walkers with zero
        weight are skipped.
    Bg : :class:`numpy.ndarray`
        Group products in chronological order, shape
        (K, nspin, nwalkers, nbasis, nbasis), on the device in GPU runs.
        nspin = 1 declares the spin blocks identical over the whole path.
    measure_batch_fn : callable
        Called once per cut as measure_batch_fn(ids, G, k) with the host walker
        indices ids and their Green's functions
        :math:`G_s = I - (I + P_k S_k)^{-1}`, shape (len(ids), 2, nbasis, nbasis).
    stab_every : int
        Restratification interval in groups; 1 suits groups of about ten or
        more slices. Default 1.
    mem_budget : float
        Bytes for the suffix snapshots; sets the walker chunk size. Default 2e10.
    check_phase : bool
        Include the phase in the cyclic-invariance check. Default False.

    Returns
    -------
    worst_dev : float
        Largest cyclic-invariance violation of the trace log determinant over
        walkers and cuts.
    ld0 : :class:`numpy.ndarray`
        Trace log determinant at cut 0 per walker (NaN for skipped walkers);
        compared with the endpoint of the propagated walker it checks the
        stored products and their bookkeeping through population control.
    """
    K, nspin, nw, nb, _ = Bg.shape
    weight = to_host(weight)
    alive = numpy.flatnonzero(weight != 0.0)
    ld0_out = numpy.full(nw, numpy.nan, dtype=numpy.complex128)
    if alive.size == 0:
        return 0.0, ld0_out
    per_walker = K * nspin * (2 * nb * nb + nb) * 16.0
    chunk = max(1, min(alive.size, int(mem_budget / per_walker)))
    eye = xp.eye(nb, dtype=xp.complex128)
    worst_dev = 0.0
    for c0 in range(0, alive.size, chunk):
        ids = alive[c0 : c0 + chunk]
        n = ids.size
        # A contiguous walker range is indexed by a slice (a view, no copy);
        # otherwise gather.  Alive walkers are almost always contiguous.
        if ids[-1] - ids[0] + 1 == n:
            ids_x = slice(int(ids[0]), int(ids[-1]) + 1)
        else:
            ids_x = xp.asarray(ids)
        # Backward sweep: suffixes, stored at the cuts.
        Sq = xp.tile(eye, (nspin, n, 1, 1))
        Sd = xp.zeros((nspin, n, nb))
        St = xp.tile(eye, (nspin, n, 1, 1))
        snaps = {}
        for k in range(K - 1, -1, -1):
            if (k + 1) % stab_every == 0:
                for sp in range(nspin):
                    Sq[sp], Sd[sp], St[sp] = restratify_right_batched(Sq[sp], Sd[sp], St[sp])
            for sp in range(nspin):
                St[sp] = St[sp] @ Bg[k, sp][ids_x]
            snaps[k] = (Sq.copy(), Sd.copy(), St.copy())
        # Forward sweep: prefixes and the contraction at every cut.
        Pq = xp.tile(eye, (nspin, n, 1, 1))
        Pd = xp.zeros((nspin, n, nb))
        Pt = xp.tile(eye, (nspin, n, 1, 1))
        ld_ref = numpy.full(n, numpy.nan, dtype=numpy.complex128)
        for k in range(K):
            Qs, Ds, Ts = snaps.pop(k)
            lo = xp.zeros(n, dtype=xp.complex128)
            G = xp.empty((n, 2, nb, nb), dtype=xp.complex128)
            for sp in range(nspin):
                with numpy.errstate(divide="ignore", invalid="ignore"):
                    sgn_t, log_t = xp.linalg.slogdet(Ts[sp])
                    ldv = log_t + xp.log(sgn_t)
                ld, inv = stabilized_inverse_one_plus_batched(
                    Pq[sp], Pd[sp], Pt[sp] @ Qs[sp], Ds[sp], Vinv=inv_stack(Ts[sp]), log_det_V=ldv
                )
                lo += ld
                G[:, sp] = eye - inv
            if nspin == 1:
                lo += ld
                G[:, 1] = G[:, 0]
            lo_h = to_host(lo)
            good = numpy.isfinite(lo_h) & to_host(xp.isfinite(G).all(axis=(1, 2, 3)))
            if k == 0:
                ld0_out[ids[good]] = lo_h[good]
            fresh = numpy.isnan(ld_ref) & good
            ld_ref[fresh] = lo_h[fresh]
            seen = good & ~fresh
            if seen.any():
                dev = lo_h[seen] - ld_ref[seen]
                dev = principal_log_phase(dev) if check_phase else dev.real
                worst_dev = max(worst_dev, float(numpy.max(numpy.abs(dev))))
            if good.any():
                sel = numpy.flatnonzero(good)
                measure_batch_fn(ids[sel], G[xp.asarray(sel)] if sel.size < n else G, k)
            for sp in range(nspin):
                Pq[sp] = Bg[k, sp][ids_x] @ Pq[sp]
            if (k + 1) % stab_every == 0:
                for sp in range(nspin):
                    Pq[sp], Pd[sp], Pt[sp] = restratify_left_batched(Pq[sp], Pd[sp], Pt[sp])
    return worst_dev, ld0_out
