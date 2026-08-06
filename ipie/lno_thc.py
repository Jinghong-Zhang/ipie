# Factored THC operations for LNO-AFQMC: propagator pieces (mean-field shift,
# one-body propagator, VHS, force bias) and the LNO fragment two-body energy.
#
# All contractions go through X (nbasis x Nmu) and zeta (Nmu x Ngamma, M = zeta
# zeta^T); the Cholesky-like vectors L_pq^gamma = sum_mu X_p,mu X_q,mu zeta_mu,gamma
# are NEVER materialized.  These mirror, term for term, the GenericRealChol
# propagator/energy so the THC path can replace the CD path.
#
# THC relations used (derived in project_lno_isdf_thc_afqmc_export):
#   rho_mu[w]        = sum_pq X_p,mu G_pq[w] X_q,mu                       (density moment)
#   vbias_gamma[w]   = sum_mu zeta_mu,gamma rho_mu[w]
#   VHS[w]           = isqrt_dt * X diag(zeta . xshifted[w]) X^T
#   v0_ij (h1e_mod)  = 0.5 [ X (M .* (X^T X)) X^T ]_ij
# Half-rotated collocation: Xocc = psi_occ^T X (nocc x Nmu).
#
# PERFORMANCE CONVENTIONS (2026-07 audit):
#   * X, zeta, M are REAL.  Every contraction of a complex walker quantity with
#     them is done as TWO REAL GEMMs on the .real/.imag parts ("split").  This
#     halves the GEMM flops versus the complex-promoted form and, critically,
#     avoids numpy/cupy allocating a full complex COPY of the real operand on
#     every call (10 GB for a 25k x 25k M/zeta at production sizes).
#   * All walker-batched intermediates of size (nw x nbasis x Nmu) or
#     (nw x nocc x Nmu) are built in WALKER CHUNKS (env-tunable) so the peak
#     memory is O(chunk) not O(nw).  Chunking is exact: walkers are independent.
#   * The energy path always contracts against the fp64 X and M, regardless of
#     the (optional, env-gated) reduced-precision propagation factors.
#
# ENV GATES:
#   LNO_VHS_CHUNK / LNO_FB_CHUNK / LNO_EST_CHUNK : walker chunk sizes
#     (default 64) for VHS construction, force bias, and the fragment energy.
#   IPIE_THC_MIXED=1 : PROPAGATION ONLY in reduced precision (float32 /
#     complex64) using the ham.X32/ham.zeta32 factors built by GenericRealTHC;
#     the local-energy/fragment estimator stays fp64.  Set CUPY_TF32=1 in the
#     environment (before cupy is imported) to use TF32 tensor cores on GPU.

import os

import numpy
import scipy.linalg

from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import to_host


def _env_flag(name):
    """True if the environment variable is set to a truthy value."""
    return os.environ.get(name, "") not in ("", "0", "false", "False", "no", "off")


def _env_int(name, default):
    """Positive integer from the environment, else default."""
    try:
        v = int(os.environ.get(name, "") or default)
    except ValueError:
        return default
    return max(1, v)


# Cache for the fragment rotation (audit finding: _as_rot rebuilt eye/asarray on
# every call).  Keyed on the identity of the frag object (kept alive by the
# cache entry so ids cannot be recycled), nocc, and the backend module.
_AS_ROT_CACHE = {}


def _as_rot(frag, nocc, like=None):
    """Fragment occ rotation (nocc x nfrag).  An int n means the OLD slice [:n]
    (= eye[:, :n]); a 2D array is used as the localized-fragment projector basis
    Bf (columns = eigenvectors of M_occ_frag = Re(Uocc_pc^dag UoccF_pc) with
    eigenvalue 1).  The fragment energy then projects the occupied index onto the
    LOCALIZED fragment orbital (paper Eq 16b / qcpbc E_embed), not the first n
    pseudocanonical occupieds (which mixes fragment+bath and breaks monotonicity).

    Returned on the SAME backend as `like` (the operand it will be contracted with):
    the THC energy/setup contracts Bf against the half-rotated densities, which are
    host numpy at trial setup but cupy once cast to the GPU, so we must match `like`
    to avoid a host<->device mismatch.  Falls back to the global backend xp.

    Results are cached (per frag object / int value, nocc and backend)."""
    xpf = _array_module(like)
    if len(_AS_ROT_CACHE) > 64:
        _AS_ROT_CACHE.clear()
    a = frag if (hasattr(frag, "ndim") and frag.ndim >= 2) else numpy.asarray(frag)
    if a.ndim >= 2:
        key = ("mat", id(frag), int(nocc), xpf.__name__)
        hit = _AS_ROT_CACHE.get(key)
        if hit is not None and hit[0] is frag:
            return hit[1]
        Bf = xpf.asarray(a)
        _AS_ROT_CACHE[key] = (frag, Bf)
        return Bf
    key = ("eye", int(frag), int(nocc), xpf.__name__)
    hit = _AS_ROT_CACHE.get(key)
    if hit is not None:
        return hit[1]
    Bf = xpf.eye(nocc)[:, : int(frag)]
    _AS_ROT_CACHE[key] = (None, Bf)
    return Bf


def _array_module(ref):
    """Return the array module (numpy or cupy) that owns `ref` (a host numpy array
    is numpy; a cupy array is cupy).  None -> the globally active backend xp."""
    if ref is None:
        return xp
    try:
        import cupy as _cp
        return _cp.get_array_module(ref)
    except Exception:
        return numpy


def _prop_factors(ham):
    """Propagation collocation/field factors: the reduced-precision (X32, zeta32)
    pair when IPIE_THC_MIXED built them, else the fp64 (X, zeta)."""
    X32 = getattr(ham, "X32", None)
    zeta32 = getattr(ham, "zeta32", None)
    if X32 is not None and zeta32 is not None:
        return X32, zeta32
    return ham.X, ham.zeta


# ----------------------------------------------------------------------
# Propagator pieces
# ----------------------------------------------------------------------

def construct_mean_field_shift_thc(ham, trial):
    r"""mf_gamma = sum_pq L_pq^gamma G^charge_pq = zeta^T rho, with rho from the
    total (alpha+beta) trial density.  Matches the CD convention 1j*real - imag."""
    nb = ham.nbasis
    Gcharge = (trial.G[0] + trial.G[1]).reshape(nb, nb)
    rho = ham.rho_moment(Gcharge)                 # (Nmu,) complex
    mf = ham.zeta.T @ (1.0j * rho.real - rho.imag)
    return numpy.asarray(mf)                       # (Ngamma,)


def construct_one_body_propagator_thc(ham, mf_shift, dt):
    r"""H1 -> H1 - shift, shift_ik = 1j sum_gamma L_ik^gamma mf_gamma
    = 1j sum_mu X_i,mu X_k,mu (zeta mf)_mu = 1j X diag(zeta mf) X^T.

    Built once at setup on the HOST: this uses scipy.linalg.expm (host-only) and
    numpy assembly, so the THC factors and mf_shift are staged off the GPU via
    to_host (a no-op on CPU, cupy .get()/asnumpy on GPU).

    For a real THC Hamiltonian with a real trial density, rho is real, so
    mf_shift is purely imaginary and shift = 1j X diag(zeta mf) X^T is REAL.
    The propagator is therefore stored as a REAL matrix (asserted below), which
    lets propagate_one_body apply it with real GEMMs (half the flops, no
    complex promotion copy).  Under IPIE_THC_MIXED it is downcast to float32."""
    zeta = to_host(ham.zeta)
    X = to_host(ham.X)
    h1e_mod = to_host(ham.h1e_mod)
    mf_shift = to_host(mf_shift)
    y = zeta @ mf_shift                            # (Nmu,) complex
    shift = 1.0j * (X * y[None, :]) @ X.T          # (nbasis x nbasis)
    H1 = numpy.asarray(h1e_mod) - numpy.array([shift, shift])
    imax = float(numpy.max(numpy.abs(H1.imag)))
    assert imax < 1e-10, (
        f"THC one-body propagator: max |imag(H1)| = {imax:.3e} >= 1e-10; "
        "expected a real H1 (real THC Hamiltonian + real trial density)."
    )
    H1r = numpy.ascontiguousarray(H1.real)
    expH1 = numpy.array(
        [scipy.linalg.expm(-0.5 * dt * H1r[0]), scipy.linalg.expm(-0.5 * dt * H1r[1])]
    )                                              # REAL (2, nbasis, nbasis)
    if _env_flag("IPIE_THC_MIXED"):
        expH1 = expH1.astype(numpy.float32)
    return expH1


def construct_VHS_thc(ham, xshifted, isqrt_dt, chunk=None):
    r"""VHS[w] = isqrt_dt sum_gamma xshifted_gamma[w] L_pq^gamma
    = isqrt_dt X diag(zeta . xshifted[w]) X^T.  xshifted: (Ngamma x nwalkers).

    Walker-chunked, real-split evaluation:
      Y = zeta @ xshifted done as two real GEMMs (Yr, Yi) -- no complex
      promotion copy of zeta;
      per chunk of walkers, A[c,p,mu] = X[p,mu]*Y[mu,w] (real) and the
      chunk-batched real GEMM A @ X^T gives the real/imag parts of
      X diag(y_w) X^T, assembled with the (purely imaginary) isqrt_dt.
    Identical math to the old complex per-walker GEMM (same multiply/add set)
    at half the flops, with O(chunk) scratch instead of a per-walker complex
    promotion copy of X^T.  Chunk size: LNO_VHS_CHUNK (default 64).

    Under IPIE_THC_MIXED the ham.X32/zeta32 float32 factors are used and a
    complex64 VHS is returned (propagation only; energies stay fp64)."""
    X, zeta = _prop_factors(ham)
    xpf = _array_module(X)
    rdt = X.dtype
    cdt = numpy.complex64 if rdt == numpy.float32 else numpy.complex128
    nw = xshifted.shape[-1]
    nb = ham.nbasis
    if chunk is None:
        chunk = _env_int("LNO_VHS_CHUNK", 64)
    # CONTIGUOUS copies of the field's real/imag parts: numpy silently skips the
    # BLAS fast path for element-strided views (.real of complex) and falls back
    # to a ~50-100x slower loop.  Likewise the chunk GEMM below is issued as a
    # single 2D (c*nb, Nmu) @ (Nmu, nb) GEMM (always BLAS) rather than a stacked
    # 3D matmul (gufunc fallback on some layouts).  Measured on-node: 16 s ->
    # ~0.15 s per call at nb=202/Nmu=2496/nw=64.
    xr = xpf.ascontiguousarray(xshifted.real, dtype=rdt)   # (Ngamma, nw) real
    xi = xpf.ascontiguousarray(xshifted.imag, dtype=rdt)
    Yr = zeta @ xr                                 # (Nmu, nw) real GEMM
    Yi = zeta @ xi
    z = complex(isqrt_dt)
    ar, ai = z.real, z.imag                        # isqrt_dt = ar + i*ai (ar = 0)
    VHS = xpf.empty((nw, nb, nb), dtype=cdt)
    XT = X.T
    for s in range(0, nw, chunk):
        e = min(s + chunk, nw)
        c = e - s
        A = X[None, :, :] * Yr[:, s:e].T[:, None, :]           # (c, nb, Nmu) real, C-contig
        R = (A.reshape(c * nb, -1) @ XT).reshape(c, nb, nb)     # 2D real GEMM
        A = X[None, :, :] * Yi[:, s:e].T[:, None, :]
        I = (A.reshape(c * nb, -1) @ XT).reshape(c, nb, nb)
        # isqrt_dt * (R + i I) = (ar R - ai I) + i (ai R + ar I)
        VHS[s:e] = R * complex(ar, ai) + I * complex(-ai, ar)
    return VHS


def construct_force_bias_thc(ham, Xocca, Xoccb, Ghalfa, Ghalfb, chunk=None):
    r"""vbias_gamma[w] = zeta^T (rho_a[w] + rho_b[w]), rho from the half-rotated
    collocation.  Ghalfa/Ghalfb: (nwalkers, nocc, nbasis).  Returns (Ngamma, nwalkers).

    Closed shell / rhf walkers: pass Ghalfb=None (Xoccb ignored) and the total
    density is 2*rho_a (bitwise identical to rho_a + rho_a).

    Walker-chunked, real-split: B = Ghalf @ X as two real GEMMs per chunk (no
    (nw x nocc x Nmu) complex intermediate, no promotion copy of X or zeta),
    rho accumulated per chunk, then vbias = rho @ zeta as two real GEMMs.
    Chunk size: LNO_FB_CHUNK (default 64).  Under IPIE_THC_MIXED the float32
    factors are used and a complex64 vbias is returned."""
    X, zeta = _prop_factors(ham)
    xpf = _array_module(X)
    rdt = X.dtype
    nw = Ghalfa.shape[0]
    nmu = X.shape[1]
    if chunk is None:
        chunk = _env_int("LNO_FB_CHUNK", 64)
    rho_r = xpf.empty((nw, nmu), dtype=rdt)
    rho_i = xpf.empty((nw, nmu), dtype=rdt)

    def _accum(Xocc, Ghalf, first):
        Xo = xpf.asarray(Xocc, dtype=rdt)
        nocc = Ghalf.shape[1]
        nbas = Ghalf.shape[2]
        for s in range(0, nw, chunk):
            e = min(s + chunk, nw)
            c = e - s
            # Contiguous copies + single 2D GEMM: strided .real views and
            # stacked 3D matmuls dodge BLAS in numpy (~100x slower).
            Gr = xpf.ascontiguousarray(Ghalf[s:e].real, dtype=rdt)
            Gi = xpf.ascontiguousarray(Ghalf[s:e].imag, dtype=rdt)
            Br = (Gr.reshape(c * nocc, nbas) @ X).reshape(c, nocc, nmu)  # real GEMM
            Bi = (Gi.reshape(c * nocc, nbas) @ X).reshape(c, nocc, nmu)
            rr = (Br * Xo[None]).sum(axis=1)       # (c, Nmu)
            ri = (Bi * Xo[None]).sum(axis=1)
            if first:
                rho_r[s:e] = rr
                rho_i[s:e] = ri
            else:
                rho_r[s:e] += rr
                rho_i[s:e] += ri

    _accum(Xocca, Ghalfa, first=True)
    if Ghalfb is None:
        rho_r *= 2.0                               # closed shell: rho_a + rho_a
        rho_i *= 2.0
    else:
        _accum(Xoccb, Ghalfb, first=False)
    vb_r = rho_r @ zeta                            # (nw, Ngamma) real GEMM
    vb_i = rho_i @ zeta
    return (vb_r + 1j * vb_i).T                    # (Ngamma, nw) complex


# ----------------------------------------------------------------------
# LNO fragment two-body energy (THC analog of ecorrcoul_lno / ecorrxx_lno)
# ----------------------------------------------------------------------

def _half_rotate_B(ham, Ghalf):
    r"""B = Ghalf @ X against the fp64 energy collocation, as two real GEMMs
    (identical values to the complex-promoted matmul, no complex copy of X).

    Contiguous copies + 2D reshape GEMMs: numpy skips BLAS for element-strided
    .real views and for some stacked 3D matmuls (~100x slower fallback)."""
    X = ham.X
    xpf = _array_module(X)
    nw, nocc, nbas = Ghalf.shape
    nmu = X.shape[1]
    Gr = xpf.ascontiguousarray(Ghalf.real)
    Gi = xpf.ascontiguousarray(Ghalf.imag)
    Br = (Gr.reshape(nw * nocc, nbas) @ X).reshape(nw, nocc, nmu)
    Bi = (Gi.reshape(nw * nocc, nbas) @ X).reshape(nw, nocc, nmu)
    return Br + 1j * Bi


def lno_coul_thc(ham, Xocc, Ghalfa, Ghalfb, frag, Btot=None):
    r"""Fragment Coulomb energy ecoul[w, k] projected onto the localized fragment.

    ecoul[w,k] = 0.5 sum_gamma X2[w,gamma] X1[w,gamma,k], with the TOTAL density:
      X2[w,gamma]   = zeta^T rho_tot[w],  rho_tot_mu = sum_j Xocc_j,mu Btot[w]_j,mu  (FULL occ)
      X1[w,gamma,k] = (zeta^T P[w])_gamma,k,  P[w]_mu,k = Xocc^f_k,mu Btot^f[w]_k,mu
    where Xocc^f = Bf^T Xocc and Btot^f = Bf^T Btot rotate the occ index onto the
    fragment basis Bf (= _as_rot(frag)).  Summing over k gives sum_{ii'} M_occ_frag[i,i']
    Xocc_i Btot_i' = the fragment-projected trace.  Btot = (Ghalfa+Ghalfb) @ X.
    Returns (nwalkers, nfrag) complex.

    Btot may be passed in (shared with the exchange -- see frag_2body_thc); if
    Ghalfb is None (rhf walkers) Btot = 2 * Ghalfa @ X.  The rho @ M contraction
    is real-split (M is real; no 2x-memory complex promotion of M)."""
    Bf = _as_rot(frag, Xocc.shape[0], like=Xocc)          # (nocc x nfrag), matches Xocc backend
    xpf = _array_module(Xocc)
    if Btot is None:
        Ba = _half_rotate_B(ham, Ghalfa)
        if Ghalfb is None:
            Btot = 2.0 * Ba                               # closed shell
        else:
            Btot = Ba + _half_rotate_B(ham, Ghalfb)
    nw = Btot.shape[0]
    rho_r = xpf.ascontiguousarray(Btot.real)
    rho_i = xpf.ascontiguousarray(Btot.imag)
    rho_r = (rho_r * Xocc[None]).sum(axis=1)              # (nw, Nmu) real
    rho_i = (rho_i * Xocc[None]).sum(axis=1)
    # ENERGY metric M (G=0-included), not the propagation zeta: canonical
    # periodic-AFQMC convention (local energy from the physical two-body).
    Mrho = (rho_r @ ham.M) + 1j * (rho_i @ ham.M)          # (nw, Nmu), split real GEMMs
    Xocc_f = Bf.T @ Xocc                                   # (nfrag, Nmu)
    # Per-walker 2D GEMMs: a stacked matmul((nfrag,nocc), (nw,nocc,Nmu)) hits
    # numpy's non-BLAS gufunc fallback.
    BfT = Bf.T
    Btot_f = xpf.empty((nw, BfT.shape[0], Btot.shape[2]), dtype=Btot.dtype)
    for w in range(nw):
        Btot_f[w] = BfT @ Btot[w]                          # (nfrag, Nmu)
    Ppin = Xocc_f[None] * Btot_f                           # (nw, nfrag, Nmu)
    ecoul = 0.5 * (Mrho[:, None, :] * Ppin).sum(axis=-1)   # (nw, nfrag)
    return ecoul


def lno_exx_thc(ham, Xocc, Ghalf, frag, B=None):
    r"""Fragment exchange energy exx[w, k] for one spin, projected onto the fragment.

    exx[w,k] = 0.5 sum_{mu,nu} Xocc^f_k,mu M_mu,nu D[w]_mu,nu B^f[w]_k,nu,
      D[w] = B[w]^T Xocc (FULL occ sum),  B[w] = Ghalf[w] @ X.

    Two equivalent contractions (identical math, chosen by fragment size):
      * FUSED (nfrag == 1, the production case): never form the Nmu x Nmu D.
        exx = 0.5 sum_j (a.B_j) (M (b.Xocc_j)): Q = (b*Xocc) @ M as ONE batched
        real-split GEMM reusing M; contract with (a*B).  Cost per fragment
        orbital equals one D build, so for nfrag == 1 FUSED is the same flops
        with no Nmu^2 scratch.
      * D-BASED (nfrag >= 2): build D once per walker into preallocated REAL
        buffers Dbuf_r/Dbuf_i (real-split: no complex promotion of M or Xocc,
        half the GEMM flops of the complex form), Hadamard with M in place,
        then nfrag cheap quadratic forms.  Cost is INDEPENDENT of nfrag,
        whereas FUSED scales linearly with nfrag -- hence the branch point
        (the old nfrag*4 <= nocc threshold ran FUSED at nfrag-times the
        D-BASED flops and three (nw x nocc x Nmu) temporaries).
    B may be passed in (shared with the Coulomb term -- see frag_2body_thc).
    Returns (nwalkers, nfrag) complex."""
    Bf = _as_rot(frag, Xocc.shape[0], like=Xocc)          # (nocc x nfrag), matches Xocc backend
    xpf = _array_module(Xocc)                             # backend that owns the THC data
    if B is None:
        B = _half_rotate_B(ham, Ghalf)                    # (nw, nocc, Nmu) FULL occ
    nw = B.shape[0]
    Nmu = ham.nmu
    nocc = Xocc.shape[0]
    nfrag = Bf.shape[1]
    M = ham.M
    Xocc_f = Bf.T @ Xocc                                  # (nfrag, Nmu)
    exx = xpf.zeros((nw, nfrag), dtype=numpy.complex128)  # match Xocc backend (cupy on GPU)

    if nfrag == 1:                                        # FUSED: no Nmu^2 intermediate
        # Per-walker GEMV for the fragment contraction: tensordot would
        # materialize a transposed copy of B (7.7 GB per chunk at production)
        # and stacked matmul hits numpy's non-BLAS fallback.
        Bf0 = Bf[:, 0]
        Bfrag = xpf.empty((nw, Nmu), dtype=B.dtype)       # (nw, Nmu) complex
        for w in range(nw):
            Bfrag[w] = Bf0 @ B[w]
        Pr = xpf.ascontiguousarray(Bfrag.real)[:, None, :] * Xocc[None]  # (nw, nocc, Nmu)
        Qr = (Pr.reshape(nw * nocc, Nmu) @ M).reshape(nw, nocc, Nmu)     # real GEMM, M reused
        Pi = xpf.ascontiguousarray(Bfrag.imag)[:, None, :] * Xocc[None]
        Qi = (Pi.reshape(nw * nocc, Nmu) @ M).reshape(nw, nocc, Nmu)
        C = Xocc_f[0][None, None, :] * B                              # (nw, nocc, Nmu) complex
        Cr = xpf.ascontiguousarray(C.real)
        Ci = xpf.ascontiguousarray(C.imag)
        re = (Cr * Qr - Ci * Qi).sum(axis=(1, 2))
        im = (Cr * Qi + Ci * Qr).sum(axis=(1, 2))
        exx[:, 0] = 0.5 * (re + 1j * im)
        return exx

    # D-BASED: preallocated real D buffers, reused across walkers.  The D build
    # uses contiguous real/imag copies of B[w] and dot(..., out=): matmul with
    # strided views and/or out= skips BLAS in numpy (measured 40x slower).
    Dbuf_r = xpf.empty((Nmu, Nmu), dtype=M.dtype)
    Dbuf_i = xpf.empty((Nmu, Nmu), dtype=M.dtype)
    BfT = Bf.T
    for w in range(nw):
        Bw = B[w]                                         # (nocc, Nmu) complex
        Bw_r = xpf.ascontiguousarray(Bw.real)
        Bw_i = xpf.ascontiguousarray(Bw.imag)
        xpf.dot(Bw_r.T, Xocc, out=Dbuf_r)                 # Re D = (B^T Xocc).real
        xpf.dot(Bw_i.T, Xocc, out=Dbuf_i)                 # Im D
        Dbuf_r *= M                                       # M . D in place (real Hadamard)
        Dbuf_i *= M
        Tr = Xocc_f @ Dbuf_r                              # (nfrag, Nmu) real GEMM
        Ti = Xocc_f @ Dbuf_i
        Bfrag_w = BfT @ Bw                                # (nfrag, Nmu) complex
        Bfrag_r = xpf.ascontiguousarray(Bfrag_w.real)
        Bfrag_i = xpf.ascontiguousarray(Bfrag_w.imag)
        re = (Tr * Bfrag_r - Ti * Bfrag_i).sum(axis=1)
        im = (Tr * Bfrag_i + Ti * Bfrag_r).sum(axis=1)
        exx[w] = 0.5 * (re + 1j * im)
    return exx


def frag_2body_thc(ham, Xocca, Xoccb, Ghalfa, Ghalfb, frag, chunk=None):
    r"""Per-walker fragment two-body energy sum_k (ecoul - exx_a - exx_b), summed
    over the fragment-projected index k.  `frag` is either an int n_frag (OLD slice
    [:n], wrong basis) or the localized-fragment rotation Bf (correct, monotonic).

    Ghalfb=None means closed-shell / rhf walkers (spin-beta density == spin-alpha):
    Btot = 2*Ba and exx_b = exx_a, bitwise identical to passing Ghalfb=Ghalfa.

    Walker-chunked (LNO_EST_CHUNK, default 64): the half-rotated B factors are
    built ONCE per spin per chunk and shared between the Coulomb and exchange
    terms (previously Ghalf @ X was recomputed 3x per call).
    Returns (nwalkers,) complex."""
    xpf = _array_module(Xocca)
    nw = Ghalfa.shape[0]
    if chunk is None:
        chunk = _env_int("LNO_EST_CHUNK", 64)
    out = xpf.zeros(nw, dtype=numpy.complex128)
    for s in range(0, nw, chunk):
        e = min(s + chunk, nw)
        Ba = _half_rotate_B(ham, Ghalfa[s:e])
        if Ghalfb is None:
            Btot = 2.0 * Ba
            ecoul = lno_coul_thc(ham, Xocca, None, None, frag, Btot=Btot)
            exx = 2.0 * lno_exx_thc(ham, Xocca, None, frag, B=Ba)
        else:
            Bb = _half_rotate_B(ham, Ghalfb[s:e])
            Btot = Ba + Bb
            ecoul = lno_coul_thc(ham, Xocca, None, None, frag, Btot=Btot)
            exx = lno_exx_thc(ham, Xocca, None, frag, B=Ba)
            exx = exx + lno_exx_thc(ham, Xoccb, None, frag, B=Bb)
        out[s:e] = (ecoul - exx).sum(axis=1)
    return out
