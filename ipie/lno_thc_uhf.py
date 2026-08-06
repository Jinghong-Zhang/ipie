# Factored THC operations for the UHF-LNO path: per-spin cluster bases with a
# SHARED auxiliary (ISDF) index.  Companion to ipie/lno_thc.py, which owns the
# closed-shell / shared-basis kernels; every function here mirrors its RHF
# counterpart term for term, with the single substitution
#
#     X  ->  X_a for alpha-basis contractions, X_b for beta-basis contractions
#
# while zeta and M (and therefore the HS field x_gamma) remain SHARED between
# the spins.  Key relations (theory/lno_uhf.md section 6):
#
#   rho_mu[w]      = sum_i Xocc^a_{i mu} (Ghalfa X_a)_{i mu}
#                  + sum_j Xocc^b_{j mu} (Ghalfb X_b)_{j mu}     (TOTAL charge density)
#   vbias_gamma[w] = sum_mu zeta_{mu gamma} rho_mu[w]            (one field set)
#   VHS^s[w]       = isqrt_dt * X_s diag(zeta . xshifted[w]) X_s^T   (same field, per-spin block)
#   v0^s           = 0.5 [ X_s (M .* (X_s^T X_s)) X_s^T ]        (RHF formula per spin)
#
# CPU-only (numpy); the GPU/mixed-precision fast paths of the RHF module are
# deliberately not wired up here.

import numpy
import scipy.linalg

from ipie.hamiltonians.thc import GenericRealTHCUhf
from ipie.lno_thc import _array_module, _as_rot, _env_int, lno_exx_thc
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import to_host


# ----------------------------------------------------------------------
# Propagator pieces
# ----------------------------------------------------------------------

def construct_mean_field_shift_thc_uhf(ham, trial):
    r"""mf_gamma = zeta^T rho_tot with rho_tot from the trial's alpha + beta
    densities, each contracted through its OWN collocation over the SHARED mu
    index.  Matches the CD/RHF convention 1j*real - imag."""
    Ga = numpy.asarray(trial.G[0])
    Gb = numpy.asarray(trial.G[1])
    rho = ham.rho_moment_spin(Ga, 0) + ham.rho_moment_spin(Gb, 1)   # (Nmu,)
    mf = ham.zeta.T @ (1.0j * rho.real - rho.imag)
    return numpy.asarray(mf)                                        # (Ngamma,)


def construct_one_body_propagator_thc_uhf(ham, mf_shift, dt):
    r"""Per-spin one-body propagators [expH1a, expH1b] (a LIST: the two spin
    blocks live in different-sized bases).  The mean-field shift term uses the
    SHARED field moment y = zeta mf but the spin's own collocation:
        shift^s = 1j X_s diag(y) X_s^T,   H1^s = h1e_mod^s - shift^s.
    Mirrors construct_one_body_propagator_thc (real H1 asserted, host expm)."""
    zeta = to_host(ham.zeta)
    mf_shift = to_host(mf_shift)
    y = zeta @ mf_shift                                             # (Nmu,) complex
    out = []
    for spin in (0, 1):
        X = to_host(ham.X_spin(spin))
        h1e_mod = numpy.asarray(to_host(ham.h1e_mod[spin]))
        shift = 1.0j * (X * y[None, :]) @ X.T                       # (n_s x n_s)
        H1 = h1e_mod - shift
        imax = float(numpy.max(numpy.abs(H1.imag)))
        assert imax < 1e-10, (
            f"THC-UHF one-body propagator (spin {spin}): max |imag(H1)| = {imax:.3e} >= 1e-10; "
            "expected a real H1 (real THC Hamiltonian + real trial density)."
        )
        out.append(scipy.linalg.expm(-0.5 * dt * numpy.ascontiguousarray(H1.real)))
    return out                                                       # [expH1a, expH1b]


def construct_VHS_thc_uhf(ham, xshifted, isqrt_dt, chunk=None):
    r"""Per-spin VHS blocks (VHSa, VHSb) from ONE sampled field:
        VHS^s[w] = isqrt_dt X_s diag(zeta . xshifted[w]) X_s^T.
    xshifted: (Ngamma x nwalkers) complex.  The field moment Y = zeta @ xshifted
    is computed ONCE (shared zeta) and applied to each spin's collocation with
    the same chunked real-split scheme as construct_VHS_thc.  Returns a tuple
    (VHSa (nw, n_a, n_a), VHSb (nw, n_b, n_b))."""
    zeta = ham.zeta
    xpf = _array_module(zeta)
    rdt = zeta.dtype
    nw = xshifted.shape[-1]
    if chunk is None:
        chunk = _env_int("LNO_VHS_CHUNK", 64)
    xr = xpf.ascontiguousarray(xshifted.real, dtype=rdt)            # (Ngamma, nw)
    xi = xpf.ascontiguousarray(xshifted.imag, dtype=rdt)
    Yr = zeta @ xr                                                  # (Nmu, nw) real GEMM
    Yi = zeta @ xi
    z = complex(isqrt_dt)
    ar, ai = z.real, z.imag
    out = []
    for spin in (0, 1):
        X = ham.X_spin(spin)
        nb = X.shape[0]
        VHS = xpf.empty((nw, nb, nb), dtype=numpy.complex128)
        XT = X.T
        for s in range(0, nw, chunk):
            e = min(s + chunk, nw)
            c = e - s
            A = X[None, :, :] * Yr[:, s:e].T[:, None, :]            # (c, nb, Nmu) real
            R = (A.reshape(c * nb, -1) @ XT).reshape(c, nb, nb)
            A = X[None, :, :] * Yi[:, s:e].T[:, None, :]
            I = (A.reshape(c * nb, -1) @ XT).reshape(c, nb, nb)
            VHS[s:e] = R * complex(ar, ai) + I * complex(-ai, ar)
        out.append(VHS)
    return tuple(out)


def construct_force_bias_thc_uhf(ham, Xocca, Xoccb, Ghalfa, Ghalfb, chunk=None):
    r"""vbias_gamma[w] = zeta^T (rho_a[w] + rho_b[w]); the alpha term contracts
    Xocca/Ghalfa through X_a, the beta term Xoccb/Ghalfb through X_b, over the
    SHARED mu index.  NO factor 2 (genuine two-spin sum).  Returns
    (Ngamma, nwalkers) complex.  Mirrors construct_force_bias_thc's chunked
    real-split structure with a per-spin collocation."""
    zeta = ham.zeta
    xpf = _array_module(zeta)
    rdt = zeta.dtype
    nw = Ghalfa.shape[0]
    nmu = ham.nmu
    if chunk is None:
        chunk = _env_int("LNO_FB_CHUNK", 64)
    rho_r = xpf.zeros((nw, nmu), dtype=rdt)
    rho_i = xpf.zeros((nw, nmu), dtype=rdt)

    def _accum(X, Xocc, Ghalf):
        Xo = xpf.asarray(Xocc, dtype=rdt)
        nocc = Ghalf.shape[1]
        nbas = Ghalf.shape[2]
        for s in range(0, nw, chunk):
            e = min(s + chunk, nw)
            c = e - s
            Gr = xpf.ascontiguousarray(Ghalf[s:e].real, dtype=rdt)
            Gi = xpf.ascontiguousarray(Ghalf[s:e].imag, dtype=rdt)
            Br = (Gr.reshape(c * nocc, nbas) @ X).reshape(c, nocc, nmu)
            Bi = (Gi.reshape(c * nocc, nbas) @ X).reshape(c, nocc, nmu)
            rho_r[s:e] += (Br * Xo[None]).sum(axis=1)
            rho_i[s:e] += (Bi * Xo[None]).sum(axis=1)

    _accum(ham.Xa, Xocca, Ghalfa)
    _accum(ham.Xb, Xoccb, Ghalfb)
    vb_r = rho_r @ zeta
    vb_i = rho_i @ zeta
    return (vb_r + 1j * vb_i).T                                     # (Ngamma, nw)


# ----------------------------------------------------------------------
# LNO fragment two-body energy (per-spin bases, shared metric)
# ----------------------------------------------------------------------

def _half_rotate_B_spin(X, Ghalf):
    r"""B = Ghalf @ X_s as two real GEMMs (spin-resolved analog of
    lno_thc._half_rotate_B, with the collocation passed explicitly)."""
    xpf = _array_module(X)
    nw, nocc, nbas = Ghalf.shape
    nmu = X.shape[1]
    Gr = xpf.ascontiguousarray(Ghalf.real)
    Gi = xpf.ascontiguousarray(Ghalf.imag)
    Br = (Gr.reshape(nw * nocc, nbas) @ X).reshape(nw, nocc, nmu)
    Bi = (Gi.reshape(nw * nocc, nbas) @ X).reshape(nw, nocc, nmu)
    return Br + 1j * Bi


class _SpinView:
    """Minimal hamiltonian view for reusing lno_thc kernels that only touch
    .M / .nmu (and would touch .X only when B is not supplied)."""

    def __init__(self, ham, spin):
        self.X = ham.X_spin(spin)
        self.M = ham.M
        self.nmu = ham.nmu


def frag_2body_thc_uhf(ham, Xocca, Xoccb, Ghalfa, Ghalfb, frag_a, frag_b, chunk=None):
    r"""Per-walker fragment two-body energy for per-spin cluster bases.

    Coulomb: the TOTAL density rho_tot couples through the shared M to the
    per-spin fragment-pinned densities:
        ecoul[w] = 0.5 sum_s sum_k sum_mu (M rho_tot[w])_mu P^s[w]_{k mu},
        P^s[w]_{k mu} = (Bf_s^T Xocc^s)_{k mu} (Bf_s^T B^s[w])_{k mu},
    which reduces EXACTLY to lno_thc.lno_coul_thc's Btot pinning in the
    closed-shell limit (Xocca == Xoccb, Ghalfa == Ghalfb, frag_a == frag_b):
    there P = Xf_a * (Ba_f + Bb_f) = P^a + P^b.

    Exchange: per spin with the spin's own collocation (reuses lno_exx_thc,
    which only touches M/nmu once B is supplied).

    frag_a / frag_b select the fragment pinning of each spin's occupied index
    (int n -> first n occupieds; 2D array -> localized-fragment basis Bf; None
    -> skip that spin's pinned terms, per the per-(sigma, fragment) estimator
    of theory/lno_uhf.md section 6).  Returns (nwalkers,) complex."""
    xpf = _array_module(Xocca)
    nw = Ghalfa.shape[0]
    if chunk is None:
        chunk = _env_int("LNO_EST_CHUNK", 64)
    M = ham.M
    out = xpf.zeros(nw, dtype=numpy.complex128)
    for s0 in range(0, nw, chunk):
        e0 = min(s0 + chunk, nw)
        c = e0 - s0
        Ba = _half_rotate_B_spin(ham.Xa, Ghalfa[s0:e0])             # (c, na, Nmu)
        Bb = _half_rotate_B_spin(ham.Xb, Ghalfb[s0:e0])             # (c, nb, Nmu)
        # Total charge density on the shared mu grid.
        rho = (Ba * xpf.asarray(Xocca)[None]).sum(axis=1)           # (c, Nmu) complex
        rho = rho + (Bb * xpf.asarray(Xoccb)[None]).sum(axis=1)
        rho_r = xpf.ascontiguousarray(rho.real)
        rho_i = xpf.ascontiguousarray(rho.imag)
        Mrho = (rho_r @ M) + 1j * (rho_i @ M)                       # (c, Nmu)
        etot = xpf.zeros(c, dtype=numpy.complex128)
        for Xocc, B, frag in ((Xocca, Ba, frag_a), (Xoccb, Bb, frag_b)):
            if frag is None:
                continue
            Bf = _as_rot(frag, Xocc.shape[0], like=Xocc)            # (nocc_s, nfrag_s)
            Xocc_f = Bf.T @ Xocc                                    # (nfrag_s, Nmu)
            BfT = Bf.T
            Bfrag = xpf.empty((c, BfT.shape[0], B.shape[2]), dtype=B.dtype)
            for w in range(c):
                Bfrag[w] = BfT @ B[w]
            P = Xocc_f[None] * Bfrag                                # (c, nfrag_s, Nmu)
            ecoul = 0.5 * (Mrho[:, None, :] * P).sum(axis=-1)       # (c, nfrag_s)
            exx = lno_exx_thc(_SpinView(ham, 0 if B is Ba else 1), Xocc, None, frag, B=B)
            etot = etot + (ecoul - exx).sum(axis=1)
        out[s0:e0] = etot
    return out


# ----------------------------------------------------------------------
# Local energy (per-spin bases)
# ----------------------------------------------------------------------

def local_energy_thc_uhf(system, hamiltonian, walkers, trial):
    """Local energy for the per-spin-basis THC Hamiltonian.  Mirrors
    ipie.estimators.energy.local_energy_thc: e1 from the per-spin half-rotated
    one-body integrals, e2 from the fragment two-body with the pinning defaulting
    to ALL occupieds of each spin (or hamiltonian.n_frag first occupieds when
    set).  The walkers.rhf / LNO_FAST_ESTIMATOR fast paths are deliberately not
    honored on the UHF path."""
    Ga = xp.asarray(walkers.Ghalfa)                 # (nw, nalpha, n_a_orb)
    Gb = xp.asarray(walkers.Ghalfb)                 # (nw, nbeta,  n_b_orb)
    na = Ga.shape[1]
    nb = Gb.shape[1]
    nfrag = getattr(hamiltonian, "n_frag", None)
    frag_a = min(int(nfrag), na) if nfrag else na
    frag_b = min(int(nfrag), nb) if nfrag else nb
    rH1a = xp.asarray(trial._rH1a)
    rH1b = xp.asarray(trial._rH1b)
    e1 = xp.einsum("ij,wij->w", rH1a, Ga) + xp.einsum("ij,wij->w", rH1b, Gb)
    e2 = frag_2body_thc_uhf(
        hamiltonian, trial._thc_Xocca, trial._thc_Xoccb, Ga, Gb, frag_a, frag_b
    )
    energy = xp.zeros((walkers.nwalkers, 3), dtype=numpy.complex128)
    energy[:, 1] = e1
    energy[:, 2] = e2
    energy[:, 0] = hamiltonian.ecore + e1 + e2
    return xp.array(energy)
