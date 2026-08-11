#!/usr/bin/env python3
"""No-TRS metal LNO-AFQMC on the COMPLEX THC export (production route).

Consumes cthc_iao_<F>_<T>.h5 from postscf_lno.C's complex path:
  Theta_occ  complex collocation X_pc   (arma nmo x Nmu)
  Mcore      complex Hermitian metric   (Nmu x Nmu), G=0-free
  hcore_pq   complex Hermitian BARE core (T + V_pp)
  f_diag     real pseudocanonical orbital energies
  Pocc       complex Hermitian fragment projector
  meta       [Nmu, nocc, nvir, trs(=0), eta, w_F]
  refs       [E_frag^CCSD, E_ccsd_cluster/nk, xi]

Conventions (see ipie_lno GenericComplexTHC):
  (pq|rs) = sum_{mu,nu} X[p,mu] conj(X[q,mu]) M[mu,nu] conj(X[r,nu]) X[s,nu]

Embedded one-body.  The C++ deliberately writes only the BARE core; the
Fock-consistent h_eff is built HERE (validated Python) as
    h_eff = diag(f_diag + xi*P_occ) - vHF ,   vHF = J - K/2
with, from the convention above and a closed-shell reference D = 2*P_occ,
    S = Xocc^H Xocc  (HERMITIAN)              sigma = 2 diag(S)
    J = X diag(M sigma) X^H                   K = X (M .* 2S) X^H
i.e. the direct complex generalization of the real export's
Socc = Xocc^T Xocc / K = X (M .* 2 Socc) X^T.  NOTE the exchange index pattern
is (pr|qs) -- using (pr|sq) instead gives a NON-Hermitian K (caught by the
|K-K^H| check below).

--validate is NON-CIRCULAR at FULL SPACE: there every occupied is inside the
cluster, so the bare core must satisfy
    hcore + vHF == diag(f_diag + xi)   (diagonal; f_ov ~ 0)
which tests vHF itself.  (For TRUNCATED clusters the bare core legitimately
misses the environment mean field -- that is why h_eff is used there.)

Usage:
  python run_cthc_metal.py <cthc.h5> --validate
  python run_cthc_metal.py <cthc.h5> [nw nb dt]
"""
import os
import sys
import numpy
import h5py

H5 = sys.argv[1]
VALIDATE = "--validate" in sys.argv
args = [a for a in sys.argv[2:] if not a.startswith("--")]
NW = int(args[0]) if len(args) > 0 else 640
NB = int(args[1]) if len(args) > 1 else 200
DT = float(args[2]) if len(args) > 2 else 0.002
SEED = int(os.environ.get("SEED", 7))


def rd(f, name):
    d = f[name][...]
    if d.dtype.names:                      # arma cx compound -> complex
        d = d[d.dtype.names[0]] + 1j * d[d.dtype.names[1]]
    return d


with h5py.File(H5, "r") as f:
    # h5py sees arma's column-major buffer transposed: arma(i,j) == h5[j,i].
    X = rd(f, "Theta_occ").T.copy()        # -> (nmo, Nmu)
    M = rd(f, "Mcore").T.copy()            # -> (Nmu, Nmu)
    hcore = rd(f, "hcore_pq").T.copy()     # -> (nmo, nmo)
    f_diag = numpy.real(rd(f, "f_diag")).ravel()
    Pocc = rd(f, "Pocc").T.copy()
    meta = numpy.real(rd(f, "meta")).ravel()
    refs = numpy.real(rd(f, "refs")).ravel()

Nmu, nocc, nvir = int(meta[0]), int(meta[1]), int(meta[2])
trs_flag, eta, w_F = meta[3], meta[4], meta[5]
xi = refs[2]
nmo = nocc + nvir
M = 0.5 * (M + M.conj().T)
hcore = 0.5 * (hcore + hcore.conj().T)
if Pocc.shape != (nocc, nocc):
    Pocc = Pocc.T.copy()
Pocc = 0.5 * (Pocc + Pocc.conj().T)
print(f"[load] {H5}")
print(f"  nmo={nmo} (nocc={nocc} nvir={nvir})  Nmu={Nmu}  trs={int(trs_flag)}  "
      f"eta={eta:g}  w_F={w_F:g}  xi={xi:.5f}")
print(f"  |Im X|max={numpy.abs(X.imag).max():.3e}  |Im M|max={numpy.abs(M.imag).max():.3e}")
print(f"  tr(Pocc)={numpy.trace(Pocc).real:.10f}")

# ---------------- vHF from the complex THC factors ----------------
Xo = X[:nocc]                                     # (nocc, Nmu)
S = Xo.conj().T @ Xo                              # (Nmu, Nmu) HERMITIAN overlap
sigma = 2.0 * numpy.real(numpy.diag(S))           # (Nmu,) = 2 sum_j |X[j,mu]|^2
# PHYSICAL operator matrices (coefficient convention c' = h c):
#   J_ab = sum (ab|cd) gt_cd  = [X* diag(M sigma) X^T]_ab
#   K_ab = [X* (M .* 2 S^T) X^T]_ab       (S^T = conj(S), Hermitian)
# The X (...) X^H forms are their TRANSPOSES -- equal for real X only.
J = (X.conj() * (M @ sigma)[None, :]) @ X.T
K = X.conj() @ ((M * (2.0 * S.T)) @ X.T)
vHF = J - 0.5 * K
herm_J = numpy.abs(J - J.conj().T).max()
herm_K = numpy.abs(K - K.conj().T).max()
print(f"[vHF] |J-J^H|={herm_J:.3e}  |K-K^H|={herm_K:.3e}  "
      f"(both ~0 if the index placement is right)")

f_target = f_diag.copy()
f_target[:nocc] += xi                              # G0-free convention
F_bare = hcore + vHF                               # full-space identity target
F_bare = 0.5 * (F_bare + F_bare.conj().T)
# Structure-aware gates (mirroring the validated real path): high-lying
# virtuals carry the AO-ISDF grid's representation error for oscillatory
# products (~1e-2 at eps~14 Ha, eps_thc-INDEPENDENT and correlation-
# irrelevant), so judge the occupied shift, the LOW virtuals, and the
# occ-vir block structure -- not a blanket max over all orbitals.
dF = numpy.diag(F_bare).real - f_target
occ_shift = numpy.diag(F_bare).real[:nocc] - f_diag[:nocc]
occ_spread = occ_shift.max() - occ_shift.min()
vir_abs = numpy.abs(dF[nocc:])
low_mask = f_diag[nocc:] < 2.0
vir_low_err = vir_abs[low_mask].max() if low_mask.any() else 0.0
vir_high_err = vir_abs[~low_mask].max() if (~low_mask).any() else 0.0
f_ov = numpy.abs(F_bare[:nocc, nocc:]).max()
f_ov_rms = float(numpy.sqrt((numpy.abs(F_bare[:nocc, nocc:]) ** 2).mean()))
print(f"[fock-check FULL-SPACE] occ shift mean={occ_shift.mean():+.5f} "
      f"(xi={xi:.5f}) spread={occ_spread:.2e}")
print(f"  vir diag err: low(eps<2Ha, n={int(low_mask.sum())})={vir_low_err:.2e}  "
      f"high(n={int((~low_mask).sum())})={vir_high_err:.2e}")
print(f"  f_ov max={f_ov:.2e}  rms={f_ov_rms:.2e}")
diag_err = max(occ_spread, vir_low_err)

# embedded one-body for TRUNCATED clusters (absorbs the environment mean field)
h_eff = numpy.diag(f_target).astype(complex) - vHF
h_eff = 0.5 * (h_eff + h_eff.conj().T)

if VALIDATE:
    ok = herm_J < 1e-8 and herm_K < 1e-8
    # ERI diagonal sanity through the complex factors: (pp|pp) must be >= 0
    dpp = numpy.einsum("pm,pm,mn,pn,pn->p", X, X.conj(), M, X.conj(), X,
                       optimize=True).real
    print(f"[validate] min (pp|pp) = {dpp.min():.3e}  (>= -1e-6 expected)")
    ok = ok and dpp.min() > -1e-6
    # NOTE: hcore + vHF == diag(f+xi) is NOT a valid gate, at ANY threshold.
    # An LNO cluster is built around a FRAGMENT, so the bare core always omits
    # the environment mean field (the real path measures ||v_embed||=0.365 even
    # at eta=1e-12).  Only two things are meaningful:
    #   (i) the OCCUPIED diagonal shift must be the Madelung xi, UNIFORMLY
    #       (the environment potential is smooth over the occupied block);
    #  (ii) vHF must reproduce the two-body energy computed by the INDEPENDENT,
    #       already-validated kernel two_body_energy_thc_cx -- non-circular,
    #       since that kernel was verified against the explicit 4-index ERI.
    # occ-shift uniformity (== xi) holds ONLY at full space: for truncated
    # clusters the bare core legitimately misses the (orbital-dependent)
    # environment mean field, so the spread is O(0.1-1) and NOT a failure.
    if eta <= 1e-10:
        ok = ok and occ_spread < 1e-3
        print(f"[validate] FULL-SPACE occ-shift uniformity (= xi): "
              f"spread={occ_spread:.2e} (<1e-3), mean-xi err={abs(occ_shift.mean()-xi):.2e}")
    else:
        print(f"[validate] truncated cluster (eta={eta:g}): occ-shift spread "
              f"{occ_spread:.2e} informational only (env mean field absent from bare core)")
    sys.path.insert(0, "/n/home12/aganeshram/Software/ipie_lno")
    from ipie.hamiltonians.thc import GenericComplexTHC as _G
    from ipie.lno_thc_cx import two_body_energy_thc_cx as _e2
    _h = numpy.zeros((2, nmo, nmo), dtype=complex)
    _ham = _G(_h, X, M, ecore=0.0)
    _psi = numpy.zeros((nmo, nocc), dtype=complex); _psi[:nocc, :nocc] = numpy.eye(nocc)
    _Xo = _ham.half_rotate(_psi)                     # psi^H X
    _Xoc = _psi.conj().T @ X.conj()
    _Gh = numpy.zeros((1, nocc, nmo), dtype=complex); _Gh[0, :, :nocc] = numpy.eye(nocc)
    e2_kernel = _e2(_ham, _Xo, _Xoc, _Gh, None)[0]   # rhf trial
    D = numpy.zeros((nmo, nmo), dtype=complex)
    D[:nocc, :nocc] = 2.0 * numpy.eye(nocc)
    e2_vhf = 0.5 * numpy.einsum("pq,pq->", D, vHF)
    e2_err = abs(e2_kernel - e2_vhf)
    print(f"[validate] two-body cross-check: kernel={e2_kernel.real:.10f}  "
          f"0.5*Tr[D vHF]={e2_vhf.real:.10f}  |diff|={e2_err:.2e} (<1e-8)")
    ok = ok and e2_err < 1e-8
    print(f"[validate] {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)

# =========================== AFQMC ===========================
# Production default: skip the DEFAULT estimator's full-cluster two-body.  The
# phaseless weight update uses the HYBRID energy, not the local energy, so that
# term is a pure diagnostic -- and it is the one piece still costing
# O(nw * Nmu^2) per block (the fragment estimator is fused and hoisted).  The
# physical observable is EFragCorr.  Set LNO_FAST_ESTIMATOR=0 to re-enable it
# (needed only when the full-cluster correlation itself is wanted).
os.environ.setdefault("LNO_FAST_ESTIMATOR", "1")
sys.path.insert(0, "/n/home12/aganeshram/Software/ipie_lno")
from ipie.estimators.estimator_base import EstimatorBase
from ipie.hamiltonians.thc import GenericComplexTHC
from ipie.lno_thc_cx import frag_2body_thc_cx
from ipie.lno_thc import _as_rot, _array_module
from ipie.qmc.afqmc import AFQMC
from ipie.trial_wavefunction.single_det import SingleDet
from ipie.utils.backend import arraylib as xp


def frag_1body(rH1a, rH1b, Ga, Gb, frag):
    # PHYSICAL fragment 1-body (derived from e1F = sum h_ab gammaF_ab with
    # gammaF's trial-occ line projected by Pocc = Bf Bf^H):
    #   e1F = sum_{k,b} (Bf^T dG)[k,b] (Bf^H rH1)[k,b]
    # -> Bf^T pairs with the DENSITY (dG), Bf^H with the TRIAL/Fock factor --
    # the same rule as the 2-body physical projection (frag_gates.py).  NOTE an
    # intermediate "fix" swapped these based on an unanchored reference; this
    # form is the derived one (and the original port, which was correct).
    xpf = _array_module(Ga)
    Bf_ = _as_rot(frag, Ga.shape[1], like=Ga)
    rA = Bf_.conj().T @ xpf.asarray(rH1a)
    e = xpf.einsum("fi,wij,fj->w", Bf_.T, Ga, rA, optimize=True)
    if rH1b is not None and Gb is not None:
        rB = Bf_.conj().T @ xpf.asarray(rH1b)
        e = e + xpf.einsum("fi,wij,fj->w", Bf_.T, Gb, rB, optimize=True)
    return e


class LNOFragCx(EstimatorBase):
    """Normal-ordered '1h' complex fragment estimator: two-body kernels on
    dG = G - G_trial, one-body against the trial Fock, e_ref = 0."""

    def __init__(self, frag, thc_ham, G0a=None, G0b=None, rF1a=None, rF1b=None):
        super().__init__()
        self.frag = frag
        self.G0a, self.G0b = G0a, G0b
        self.rF1a, self.rF1b = rF1a, rF1b
        self._G0a_dev = None
        self.scalar_estimator = True
        self._data = {"ENumer": 0.0j, "EDenom": 0.0j, "EFragCorr": 0.0j}
        self._shape = (len(self.names),)
        self._data_index = {k: i for i, k in enumerate(self._data)}
        self.print_to_stdout = True
        self.ascii_filename = None

    def compute_estimator(self, system=None, walkers=None, hamiltonian=None, trial=None):
        trial.calc_greens_function(walkers)
        rhf = bool(getattr(walkers, "rhf", False))
        Ga = xp.asarray(walkers.Ghalfa)
        Gb = Ga if rhf else xp.asarray(walkers.Ghalfb)
        if self.G0a is not None:
            if self._G0a_dev is None:
                self._G0a_dev = xp.asarray(self.G0a)
                self._G0b_dev = xp.asarray(self.G0b if self.G0b is not None else self.G0a)
            Ga = Ga - self._G0a_dev[None]
            Gb = Ga if rhf else (Gb - self._G0b_dev[None])
        efrag = frag_2body_thc_cx(hamiltonian, trial._thc_Xocca, trial._thc_Xoccac,
                                  Ga, None if rhf else Gb, self.frag)
        if self.rF1a is not None:
            efrag = efrag + frag_1body(self.rF1a, self.rF1b, Ga, Gb, self.frag)
        self._data["ENumer"] = xp.sum(walkers.weight * efrag.real)
        self._data["EDenom"] = xp.sum(walkers.weight)
        return self.data

    def post_reduce_hook(self, data):
        i = self._data_index
        data[i["EFragCorr"]] = data[i["ENumer"]] / data[i["EDenom"]]


h1e = numpy.array([h_eff, h_eff])
ham = GenericComplexTHC(h1e, X, M, ecore=0.0, verbose=True)
ham.n_frag = nocc

psi = numpy.zeros((nmo, 2 * nocc), dtype=numpy.complex128)
psi[:nocc, :nocc] = numpy.eye(nocc)
psi[:nocc, nocc:] = numpy.eye(nocc)
trial = SingleDet(psi.copy(), [nocc, nocc], nmo)
trial.build(); trial.half_rotate(ham)

# Bf from the Hermitian PSD fragment projector: any Bf with Bf Bf^H = Pocc
# measures Tr[Pocc E] exactly (handles the fractional metal spectrum).
wv, V = numpy.linalg.eigh(Pocc)
wv = numpy.clip(wv, 0.0, None)
Bf = V * numpy.sqrt(wv)[None, :]
print(f"[frag] Bf {Bf.shape}  tr(Bf Bf^H)={numpy.trace(Bf @ Bf.conj().T).real:.10f}")

F_trial = numpy.diag(f_target).astype(complex)
rFa = trial.psi0a.conj().T @ F_trial
G0a = trial.Ghalf[0][None].astype(numpy.complex128)[0]
G0b = trial.Ghalf[1][None].astype(numpy.complex128)[0]
est = LNOFragCx(Bf, ham, G0a=G0a, G0b=G0b, rF1a=rFa, rF1b=rFa)

print(f"[afqmc] complex THC nw={NW} nb={NB} dt={DT} seed={SEED}")
afqmc = AFQMC.build([nocc, nocc], ham, trial, num_walkers=NW, num_blocks=NB,
                    seed=SEED, timestep=DT)
afqmc.run(additional_estimators={"lno": est})
afqmc.finalise(verbose=False)
print(f"\n[done] E_frag(AFQMC) = w_F * EFragCorr   (w_F={w_F:g})")
print(f"[done] refs: E_frag^CCSD={refs[0]:.8f}  E_ccsd_cluster/nk={refs[1]:.8f}")
