# COMPLEX THC kernels for the no-TRS (general k-mesh) metal LNO-AFQMC path.
#
# Companion to lno_thc.py (real) and lno_thc_uhf.py (per-spin), for
# GenericComplexTHC.  See that class' docstring for the factorization; the one
# fact that drives every kernel here is:
#
#     VHS[w]   = isqrt_dt * X diag(Zc @ x[:,w]) X^H
#     vbias[w] = Zc^T @ rho[:,w]
#     rho_mu   = sum_pq X[p,mu] conj(X[q,mu]) G_pq
#
# with ONE shared REAL field matrix Zc = [Re(zeta), Im(zeta)] (Nmu x 2*Ngamma).
# Because Zc is real, every FIELD-space contraction keeps the real-split GEMM
# trick of the real kernels (two real GEMMs instead of a complex one, and no
# complex promotion copy of the field factors).  Only the X-contractions are
# genuinely complex, so the intrinsic cost is ~2-4x the real path -- that is
# complex arithmetic, not a defect.
#
# Environment (mirrors lno_thc.py):
#   LNO_VHS_CHUNK / LNO_FB_CHUNK / LNO_EST_CHUNK : walker chunk sizes.  The
#     complex VHS chunk scratch is (chunk x nbasis x Nmu) COMPLEX, i.e. 2x the
#     real path's, so the default here is 32 (vs 64) to hold the same footprint.
#   IPIE_THC_MIXED=1 : PROPAGATION ONLY in reduced precision (complex64 X /
#     float32 Zc) via ham.X32/ham.Zc32.  Energies stay fp64.

import numpy
import scipy.linalg

from ipie.utils.backend import to_host
from ipie.lno_thc import _env_flag, _env_int, _array_module, _as_rot


def _prop_factors_cx(ham):
    """Propagation (X, Zc) factors: the reduced-precision pair when
    IPIE_THC_MIXED built them, else fp64."""
    X32 = getattr(ham, "X32", None)
    Zc32 = getattr(ham, "Zc32", None)
    if X32 is not None and Zc32 is not None:
        return X32, Zc32
    return ham.X, ham.Zc


# ----------------------------------------------------------------------
# Propagator pieces
# ----------------------------------------------------------------------

def construct_mean_field_shift_thc_cx(ham, trial):
    r"""mf_f = sum_pq L_pq^f G^charge_pq = Zc^T rho, from the total (alpha+beta)
    trial density.  Same CD convention (1j*real - imag) as the real kernel.
    Returns (2*Ngamma,) complex."""
    nb = ham.nbasis
    Gcharge = (trial.G[0] + trial.G[1]).reshape(nb, nb)
    rho = ham.rho_moment(Gcharge)                       # (Nmu,) complex
    mf = ham.Zc.T @ (1.0j * rho.real - rho.imag)        # real matrix, complex rhs
    return numpy.asarray(mf)                            # (2*Ngamma,)


def construct_one_body_propagator_thc_cx(ham, mf_shift, dt):
    r"""H1 -> H1 - shift, shift = 1j X diag(Zc @ mf) X^H.

    Built once at setup on the HOST (scipy.linalg.expm is host-only), so the THC
    factors and mf_shift are staged off the GPU via to_host.

    Unlike the REAL kernel -- where a real Hamiltonian + real trial density make
    the shift real and the propagator is stored as a REAL matrix -- here X is
    complex, so `shift` is Hermitian but genuinely COMPLEX.  The propagator is
    therefore complex; we assert HERMITICITY instead of realness (the physical
    invariant: a non-Hermitian shift would break the one-body propagation)."""
    Zc = to_host(ham.Zc)
    X = to_host(ham.X)
    h1e_mod = to_host(ham.h1e_mod)
    mf_shift = to_host(mf_shift)
    y = Zc @ mf_shift                                   # (Nmu,) complex
    # physical operator matrix: X* diag(y) X^T (see construct_VHS_thc_cx note)
    shift = 1.0j * (X.conj() * y[None, :]) @ X.T        # (nbasis x nbasis)
    H1 = numpy.asarray(h1e_mod) - numpy.array([shift, shift])
    herm = float(numpy.max(numpy.abs(H1 - H1.conj().transpose(0, 2, 1))))
    scale = max(1.0, float(numpy.max(numpy.abs(H1))))
    assert herm < 1e-8 * scale, (
        f"complex THC one-body propagator: max |H1 - H1^H| = {herm:.3e} "
        f"(scale {scale:.3e}); expected a Hermitian H1."
    )
    expH1 = numpy.array(
        [scipy.linalg.expm(-0.5 * dt * H1[0]), scipy.linalg.expm(-0.5 * dt * H1[1])]
    )                                                   # COMPLEX (2, nbasis, nbasis)
    if _env_flag("IPIE_THC_MIXED"):
        expH1 = expH1.astype(numpy.complex64)
    return expH1


def construct_VHS_thc_cx(ham, xshifted, isqrt_dt, chunk=None):
    r"""VHS[w] = isqrt_dt * X diag(Zc @ xshifted[:,w]) X^H.
    xshifted : (2*Ngamma x nwalkers) complex.  Returns (nw, nbasis, nbasis) complex.

    Field space (Zc REAL) is contracted with two real GEMMs -- no complex
    promotion copy of Zc -- exactly as the real kernel does with zeta.  The
    X-contraction is a single 2D COMPLEX GEMM per walker chunk:
        A[c,p,mu] = X[p,mu] * Y[mu,w]      then      A @ X^H.
    Issued as one (c*nb, Nmu) @ (Nmu, nb) GEMM (always BLAS zgemm) rather than a
    stacked 3D matmul, which falls off the BLAS path in numpy.

    Chunk: LNO_VHS_CHUNK (default 32 here; the scratch is complex, so 32 holds
    the same bytes as the real kernel's 64)."""
    X, Zc = _prop_factors_cx(ham)
    xpf = _array_module(X)
    cdt = X.dtype                                       # complex64 or complex128
    rdt = numpy.float32 if cdt == numpy.complex64 else numpy.float64
    nw = xshifted.shape[-1]
    nb = ham.nbasis
    if chunk is None:
        chunk = _env_int("LNO_VHS_CHUNK", 32)
    # Contiguous real/imag copies: numpy skips BLAS for element-strided .real
    # views of a complex array (~50-100x slower fallback).
    xr = xpf.ascontiguousarray(xshifted.real, dtype=rdt)     # (2*Ngamma, nw)
    xi = xpf.ascontiguousarray(xshifted.imag, dtype=rdt)
    Yr = Zc @ xr                                        # (Nmu, nw) real GEMM
    Yi = Zc @ xi
    Y = (Yr + 1j * Yi).astype(cdt)
    VHS = xpf.empty((nw, nb, nb), dtype=cdt)
    # PHYSICAL operator matrix (c' = h c convention): VHS = X* diag(Y) X^T.
    # The X diag(Y) X^H form is its TRANSPOSE -- identical for real X, wrong
    # (breaks h1 -> U^H h U covariance) for complex X.  See frag_gates.py.
    Xc_ = xpf.ascontiguousarray(X.conj())
    XT = xpf.ascontiguousarray(X.T)                     # (Nmu, nb)
    z = cdt.type(isqrt_dt) if hasattr(cdt, "type") else isqrt_dt
    for s in range(0, nw, chunk):
        e = min(s + chunk, nw)
        c = e - s
        # Y[:, s:e].T is a TRANSPOSED VIEW (non-contiguous); broadcasting against
        # it yields a non-contiguous (c, nb, Nmu) temporary, so the reshape below
        # must copy and the GEMM can fall off the BLAS fast path.  Materialize the
        # chunk contiguously first -- same discipline as the real kernel, where
        # the equivalent fix was measured at 16 s -> 0.15 s per call.
        Yc = xpf.ascontiguousarray(Y[:, s:e].T)         # (c, Nmu) complex
        A = Xc_[None, :, :] * Yc[:, None, :]            # (c, nb, Nmu) C-contiguous
        VHS[s:e] = (A.reshape(c * nb, -1) @ XT).reshape(c, nb, nb) * z
    return VHS


def construct_force_bias_thc_cx(ham, Xoccac, Xoccbc, Ghalfa, Ghalfb, chunk=None):
    r"""vbias_f[w] = (Zc^T rho_t[w])_f with the PHYSICAL density bubble
    rho_mu = sum_i Xoccc[i,mu] (Ghalf @ X)[i,mu]   (Xoccc = psi^H conj(X)).
    NOTE the arguments are the CONJUGATE half-rotations trial._thc_Xoccac/bc --
    the Xocc/Bc pairing was the wrong-for-complex bubble (frag_gates.py).
    Ghalfa/Ghalfb : (nwalkers, nocc, nbasis).  Returns (nfields, nwalkers)."""
    X, Zc = _prop_factors_cx(ham)
    xpf = _array_module(X)
    cdt = X.dtype
    rdt = numpy.float32 if cdt == numpy.complex64 else numpy.float64
    nw = Ghalfa.shape[0]
    nmu = X.shape[1]
    if chunk is None:
        chunk = _env_int("LNO_FB_CHUNK", 64)
    rho = xpf.zeros((nw, nmu), dtype=cdt)

    def _accum(Xoccc, Ghalf, scale):
        Xo = xpf.asarray(Xoccc, dtype=cdt)
        nocc = Ghalf.shape[1]
        nbas = Ghalf.shape[2]
        for s in range(0, nw, chunk):
            e = min(s + chunk, nw)
            c = e - s
            G = xpf.ascontiguousarray(Ghalf[s:e], dtype=cdt)
            B = (G.reshape(c * nocc, nbas) @ X).reshape(c, nocc, nmu)    # complex GEMM
            rho[s:e] += scale * (B * Xo[None]).sum(axis=1)               # (c, Nmu)

    _accum(Xoccac, Ghalfa, 2.0 if Ghalfb is None else 1.0)
    if Ghalfb is not None:
        _accum(Xoccbc, Ghalfb, 1.0)
    # Zc is REAL -> two real GEMMs, no complex promotion of the field factors.
    vb_r = xpf.ascontiguousarray(rho.real, dtype=rdt) @ Zc               # (nw, 2*Ngamma)
    vb_i = xpf.ascontiguousarray(rho.imag, dtype=rdt) @ Zc
    return (vb_r + 1j * vb_i).T                                          # (2*Ngamma, nw)


# ----------------------------------------------------------------------
# Two-body energy (full cluster) -- PHYSICAL (gauge-covariant) bubbles
# ----------------------------------------------------------------------
#
# Derived from the textbook mixed estimator (chemist ERI (ab|cd)_chem =
# (ba|cd)_export; standard mixed 1RDM gamma == ipie's G = conj(psi) Ghalf), the
# UNIQUE pairing that is (i) gauge-invariant under complex orbital rotations and
# (ii) equal to the validated real production kernel on real bases
# (frag_gates.py gates A/A'/B/B' all ~1e-14) is:
#
#   rho_mu  = sum_i Xoc[i,mu] B[i,mu]          (ONE bubble, both Coulomb sides)
#   T_mn    = sum_i B[i,m] Xoc[i,n]            (exchange, contracted with T^T)
#   E2      = 0.5 rho_t M rho_t  -  0.5 sum_s sum_mn M_mn T^s_mn T^s_nm
#
# with B = Ghalf @ X and Xoc = psi^H conj(X)  ==  ONLY these two objects.
# The previous kernels' Xo = psi^H X and Bc = Ghalf conj(X) bubbles were a
# WRONG index pairing: on a real basis the export ERI is p<->q symmetric, so
# the wrong pairing is INVISIBLE there (which is how it survived every
# real-basis validation); on a complex basis it breaks gauge covariance and
# under-samples fragment correlation.  See frag_gates.py.

def two_body_energy_thc_cx(ham, Xocc, Xoccc, Ghalfa, Ghalfb, chunk=None):
    r"""Full-cluster two-body energy (ecoul - exx) per walker, physical bubbles.

    Xocc (= psi^H X) is UNUSED (kept for signature compatibility); the physical
    bubbles need only Xoccc = psi^H conj(X) and B = Ghalf X.
    Ghalfb=None -> rhf/closed-shell.  Returns (nwalkers,) complex."""
    X = ham.X
    xpf = _array_module(X)
    M = ham.M
    nw = Ghalfa.shape[0]
    nmu = ham.nmu
    if chunk is None:
        chunk = _env_int("LNO_EST_CHUNK", 64)
    Xoc = xpf.asarray(Xoccc)

    def _rho(Ghalf):
        nocc, nbas = Ghalf.shape[1], Ghalf.shape[2]
        rho = xpf.empty((nw, nmu), dtype=numpy.complex128)
        for s in range(0, nw, chunk):
            e = min(s + chunk, nw)
            c = e - s
            G = xpf.ascontiguousarray(Ghalf[s:e])
            B = (G.reshape(c * nocc, nbas) @ X).reshape(c, nocc, nmu)
            rho[s:e] = (B * Xoc[None]).sum(axis=1)
        return rho

    rho_a = _rho(Ghalfa)
    rho_t = 2.0 * rho_a if Ghalfb is None else rho_a + _rho(Ghalfb)
    ecoul = 0.5 * ((rho_t @ M) * rho_t).sum(axis=1)

    def _exx(Ghalf):
        nocc = Ghalf.shape[1]
        out = xpf.zeros(nw, dtype=numpy.complex128)
        for w in range(nw):
            B = xpf.ascontiguousarray(Ghalf[w]) @ X          # (nocc, Nmu)
            T = B.T @ Xoc                                    # (Nmu, Nmu)
            out[w] = 0.5 * (M * T * T.T).sum()
        return out

    exx = _exx(Ghalfa)
    exx = 2.0 * exx if Ghalfb is None else exx + _exx(Ghalfb)
    return ecoul - exx


# ----------------------------------------------------------------------
# LNO FRAGMENT two-body energy -- PHYSICAL projection
# ----------------------------------------------------------------------
#
# Fragment = project the trial-occ line of the SECOND density with the
# Hermitian PSD Pocc = Bf Bf^H (the real-path convention; frag_gates gate B'
# proves the physical form == the real production frag kernel to 7e-15, and
# gate A' proves gauge invariance).  In bubbles: Bf^T pairs with the DENSITY
# factor (B), Bf^H with the TRIAL factor (Xoc) -- same rule as the 1-body:
#
#   rhoF_mu = sum_k (Bf^T B)[k,mu] (Bf^H Xoc)[k,mu]
#   TF_mn   = sum_k (Bf^T B)[k,m]  (Bf^H Xoc)[k,n]
#   E2F     = 0.5 rho_t M rhoF_t * 2      (project second density only)
#           - 0.5 sum_s sum_mn M_mn TF^s_mn T^s_nm

def frag_2body_thc_cx(ham, Xocc, Xoccc, Ghalfa, Ghalfb, frag, chunk=None):
    r"""Fragment-projected two-body energy per walker (physical projection).
    Xocc unused (signature compatibility).  `frag` = Bf (nocc x nfrag) or int.
    Ghalfb=None -> rhf.  Returns (nw,) complex."""
    X = ham.X
    xpf = _array_module(X)
    M = ham.M
    nw = Ghalfa.shape[0]
    nmu = ham.nmu
    nocc = Xoccc.shape[0]
    if chunk is None:
        chunk = _env_int("LNO_EST_CHUNK", 64)
    Bf = xpf.asarray(_as_rot(frag, nocc, like=Xoccc), dtype=numpy.complex128)
    Xoc = xpf.asarray(Xoccc)
    XocP = Bf.conj().T @ Xoc                       # (nfrag, Nmu), trial side

    def _bubbles(Ghalf):
        no, nbas = Ghalf.shape[1], Ghalf.shape[2]
        rho = xpf.empty((nw, nmu), dtype=numpy.complex128)
        rhoF = xpf.empty((nw, nmu), dtype=numpy.complex128)
        for s in range(0, nw, chunk):
            e = min(s + chunk, nw)
            c = e - s
            G = xpf.ascontiguousarray(Ghalf[s:e])
            B = (G.reshape(c * no, nbas) @ X).reshape(c, no, nmu)
            rho[s:e] = (B * Xoc[None]).sum(axis=1)
            BF = xpf.einsum("ik,wim->wkm", Bf, B, optimize=True)   # Bf^T B
            rhoF[s:e] = (BF * XocP[None]).sum(axis=1)
        return rho, rhoF

    ra, raF = _bubbles(Ghalfa)
    if Ghalfb is None:
        rt, rtF = 2.0 * ra, 2.0 * raF
    else:
        rb, rbF = _bubbles(Ghalfb)
        rt, rtF = ra + rb, raF + rbF
    ecoul = 0.5 * ((rt @ M) * rtF).sum(axis=1)

    def _exx(Ghalf):
        no = Ghalf.shape[1]
        out = xpf.zeros(nw, dtype=numpy.complex128)
        for w in range(nw):
            B = xpf.ascontiguousarray(Ghalf[w]) @ X
            T = B.T @ Xoc                                    # (Nmu, Nmu) full
            TF = (Bf.T @ B).T @ XocP                         # (Nmu, Nmu) frag
            out[w] = 0.5 * (M * TF * T.T).sum()
        return out

    exx = _exx(Ghalfa)
    exx = 2.0 * exx if Ghalfb is None else exx + _exx(Ghalfb)
    return ecoul - exx
