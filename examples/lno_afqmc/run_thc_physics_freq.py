"""Physics LNO-AFQMC with REAL integrals from the THC cache, THC vs CD.

The full-orbital THC export is in the pseudocanonical LNO basis (the *_pc
rotations), so the cluster Fock is diagonal = orbital energies (ps_occ | ps_vir).
The bare one-body core (= postscf h_pq) is then reconstructed factored from THC:
    h = diag(eps) - v_HF[G_HF],   v_HF = 2 J - K   (RHF, occ projector G_HF)
    J_pq = sum_k^occ (pq|kk) = [ X diag(M d) X^T ]_pq,  d_nu = sum_k Xocc_k,nu^2
    K_pq = sum_k^occ (pk|kq) = [ X (M . S_occ) X^T ]_pq, S_occ = Xocc^T Xocc
Then AFQMC is run through BOTH the THC and the CD (chol = W zeta) paths on the
same real Hamiltonian; the fragment correlation energy EFragCorr must agree.

Usage: python run_thc_physics.py <thc_full.h5> <case_dir> [n_frag] [nw] [nb]
"""
import os
import sys
import numpy
import h5py

from ipie.config import MPI
from ipie.estimators.estimator_base import EstimatorBase
from ipie.estimators.local_energy_sd import (
    ecorrcoul_lno_real_rchol_uhf,
    ecorrxx_lno_real_rchol,
)
from ipie.hamiltonians.generic import GenericRealChol
from ipie.hamiltonians.thc import GenericRealTHC
from ipie.lno_thc import frag_2body_thc, _as_rot, _array_module
from ipie.qmc.afqmc import AFQMC
from ipie.trial_wavefunction.single_det import SingleDet
from ipie.utils.backend import arraylib as xp


def load_arma_vec(path):
    """Read an armadillo binary vector saved by arma::vec::save (ARMA_MAT_BIN)."""
    import numpy as np
    with open(path, "rb") as f:
        header = f.readline().decode().strip()      # e.g. ARMA_MAT_BIN_FN008
        dims = f.readline().decode().split()
        nrow, ncol = int(dims[0]), int(dims[1])
        data = np.fromfile(f, dtype=np.float64, count=nrow * ncol)
    return data.reshape(ncol, nrow).T.ravel()        # column-major -> flat


def load_arma_mat(path):
    """Read an armadillo binary matrix (real FN* or complex FC*), column-major."""
    import numpy as np
    with open(path, "rb") as f:
        header = f.readline().decode().strip()
        dims = f.readline().decode().split()
        nrow, ncol = int(dims[0]), int(dims[1])
        dt = np.complex128 if "FC" in header else np.float64
        data = np.fromfile(f, dtype=dt, count=nrow * ncol)
    return data.reshape(ncol, nrow).T                # (nrow x ncol)


def fragment_rotation(case_dir):
    """Localized-fragment occ rotation Bf (nocc x n_frag): the eigenvectors of the
    fragment projector M_occ_frag = Re(Uocc_pc^dag UoccF_pc) with eigenvalue ~1.
    Projecting the occ index onto Bf reproduces qcpbc's (monotonic) E_embed; the
    old slice(0,n_frag) of pseudocanonical occupieds mixes fragment+bath -> wrong."""
    import numpy as np
    Uocc = load_arma_mat(f"{case_dir}/Uocc_pc.arma")
    UoccF = load_arma_mat(f"{case_dir}/UoccF_pc.arma")
    Mfrag = np.real(np.conj(Uocc).T @ UoccF)         # (nocc x nocc) projector
    w, V = np.linalg.eigh(Mfrag)
    Bf = V[:, w > 0.5]                               # columns with eigenvalue ~1
    return np.ascontiguousarray(Bf), Mfrag


path = sys.argv[1] if len(sys.argv) > 1 else \
    "/n/home12/aganeshram/scratch/d222_cache_geo/thc_full_thr0.h5"
case = sys.argv[2] if len(sys.argv) > 2 else \
    "/n/home12/aganeshram/scratch/d222_cache_geo/case_0_thr0"
n_frag = int(sys.argv[3]) if len(sys.argv) > 3 else 2
nw = int(sys.argv[4]) if len(sys.argv) > 4 else 40
nb = int(sys.argv[5]) if len(sys.argv) > 5 else 30
DT = float(sys.argv[7]) if len(sys.argv) > 7 else 0.005   # timestep (LNO-AFQMC wants <=0.002)
SEED = int(os.environ.get("SEED", 7))   # env-overridable for independent statistics streams

# ---- UHF path (per-spin cluster bases, shared auxiliary index) --------------
# Dispatched EARLY: the closed-shell loader below assumes top-level X/M/meta
# datasets, which a per-spin export (a/... and b/... groups) does not carry.
# All UHF logic lives in run_thc_physics_uhf.py; this hook is the only change
# to the RHF driver.  Usage: python run_thc_physics.py <h5> <case> [n_frag]
# [nw] [nb] uhf [dt]   (synthesizes a UHF problem from an RHF h5 when the
# per-spin groups are absent -- see run_thc_physics_uhf.py).
if (sys.argv[6] if len(sys.argv) > 6 else "both") == "uhf":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import run_thc_physics_uhf as _uhf

    _uhf.main(path, n_frag=n_frag, nw=nw, nb=nb, dt=DT, seed=SEED)
    sys.exit(0)
# -----------------------------------------------------------------------------

with h5py.File(path, "r") as f:
    X = numpy.array(f["X"]).astype(numpy.float64)
    M = numpy.array(f["M"]).astype(numpy.float64)
    meta = numpy.array(f["meta"]).ravel()
    h_pq = numpy.array(f["h_pq"]).astype(numpy.float64) if "h_pq" in f else None
nbasis, Nmu, nocc, nvir = int(meta[0]), int(meta[1]), int(meta[2]), int(meta[3])
if X.shape != (nbasis, Nmu):
    X = X.T.copy()
nelec = [nocc, nocc]
print(f"X {X.shape}, M {M.shape}, nbasis={nbasis} Nmu={Nmu} nocc={nocc} nvir={nvir}")

if h_pq is not None:
    # ---- THIN CONSUMER: qcpbc exported X in the LNO basis, the real bare-core one-body
    #      h_pq, and n_frag.  NO integral work in Python (no v_HF reconstruction, no basis
    #      rotation, no fragment projector) -- the fragment is just the first n_frag occ.
    if h_pq.shape != (nbasis, nbasis):
        h_pq = h_pq.T.copy()
    h1e = numpy.array([h_pq, h_pq])
    n_frag_eff = int(round(meta[5])) if len(meta) > 5 else n_frag
    off = numpy.linalg.norm(h_pq - numpy.diag(numpy.diag(h_pq)))
    print(f"[consumer] qcpbc LNO-basis export: H1=h_pq (||offdiag||={off:.2e}, non-diagonal), "
          f"fragment = first n_frag={n_frag_eff} occ (slice; no Python integral work)")
else:
    # ---- LEGACY fallback: pseudocanonical export -> reconstruct h + rotate to LNO here.
    #      (New exports carry h_pq + n_frag and take the thin-consumer path above.) ----
    eps_occ = load_arma_vec(f"{case}/ps_occ.arma"); eps_vir = load_arma_vec(f"{case}/ps_vir.arma")
    eps = numpy.concatenate([eps_occ, eps_vir])
    frag_rot, Mfrag = fragment_rotation(case); n_frag_eff = int(round(numpy.trace(Mfrag)))
    Xocc = X[:nocc]; Socc = Xocc.T @ Xocc; d = numpy.diag(Socc).copy()
    J = (X * (M @ d)[None, :]) @ X.T; K = X @ ((M * Socc) @ X.T); vHF = 2.0 * J - K
    h_pc = numpy.diag(eps) - vHF
    psE_occ = load_arma_mat(f"{case}/ps_eigvec_occ.arma").real
    psE_vir = load_arma_mat(f"{case}/ps_eigvec_vir.arma").real
    U = numpy.zeros((nbasis, nbasis)); U[:nocc, :nocc] = psE_occ.conj(); U[nocc:, nocc:] = psE_vir.conj()
    X = U @ X; h1 = U @ h_pc @ U.T; h1e = numpy.array([0.5 * (h1 + h1.T), 0.5 * (h1 + h1.T)])
    print(f"[legacy] reconstructed h + rotated pc->LNO in Python; fragment n_frag={n_frag_eff}")

psi = numpy.zeros((nbasis, 2 * nocc))
psi[:nocc, :nocc] = numpy.eye(nocc)
psi[:nocc, nocc:] = numpy.eye(nocc)


class _TrialWalkers:
    def __init__(self, trial):
        self.Ghalfa = trial.Ghalf[0][None].astype(numpy.complex128)
        self.Ghalfb = trial.Ghalf[1][None].astype(numpy.complex128)
        self.weight = numpy.ones(1); self.nwalkers = 1


def _frag_2body_cd(trial, walkers, frag):
    """CD-path fragment two-body energy.  In the LNO basis the fragment is the
    first n_frag occupieds, so `frag` is the int n_frag (-> slice); a (nocc x nfrag)
    matrix is also accepted (projector basis) and rotates the pinned occ index."""
    naux = trial._rchola.shape[0]
    no = walkers.Ghalfa.shape[1]; nbf = walkers.Ghalfa.shape[2]
    frag_rot = _as_rot(frag, no)                         # int -> eye[:, :n] slice
    rchola = trial._rchola.reshape(naux, no, nbf); rcholb = trial._rcholb.reshape(naux, no, nbf)
    Ga = walkers.Ghalfa
    # rhf walkers never recompute Ghalfb (beta == alpha)
    Gb = Ga if bool(getattr(walkers, "rhf", False)) else walkers.Ghalfb
    rca_f = numpy.einsum("xia,ik->xka", rchola, frag_rot, optimize=True)
    rcb_f = numpy.einsum("xia,ik->xka", rcholb, frag_rot, optimize=True)
    Ga_f = numpy.einsum("wia,ik->wka", Ga, frag_rot, optimize=True)
    Gb_f = numpy.einsum("wia,ik->wka", Gb, frag_rot, optimize=True)
    ecoul = ecorrcoul_lno_real_rchol_uhf(rca_f, rchola, Ga_f, Ga, Gb_f, Gb)
    exx = ecorrxx_lno_real_rchol(rca_f, rchola, Ga_f, Ga)
    exx += ecorrxx_lno_real_rchol(rcb_f, rcholb, Gb_f, Gb)
    return (ecoul - exx).sum(axis=1)


def frag_1body(rH1a, rH1b, Ga, Gb, frag):
    """Fragment-projected 1-body correlation (Fock occ-vir coupling).

    The LNO basis is NON-canonical (f_ia != 0), so the 2-body-only fragment
    energy omits the one-body term sum_{i in frag, j} rH1[i,j] G[i,j] that the
    canonical-basis formula (Eq 16b, arXiv:2308.12430) absorbs into the Fock.
    Occ index is projected onto the fragment the same way as the 2-body.  This
    is the piece that makes  sum_frag EFragCorr = (ETotal - E_HF)  in any basis.
    """
    xpf = _array_module(Ga)
    Bf = _as_rot(frag, Ga.shape[1], like=Ga)                 # (nocc, n_frag)
    rA = Bf.T @ xpf.asarray(rH1a)                             # (n_frag, nbasis)
    e = xpf.einsum("fi,wij,fj->w", Bf.T, Ga, rA, optimize=True)
    if rH1b is not None and Gb is not None:
        rB = Bf.T @ xpf.asarray(rH1b)
        e = e + xpf.einsum("fi,wij,fj->w", Bf.T, Gb, rB, optimize=True)
    return e


class LNOFrag(EstimatorBase):
    def __init__(self, frag, e_ref, thc, G0a=None, G0b=None, rF1a=None, rF1b=None):
        super().__init__()
        self.frag = frag; self.e_ref = e_ref; self.thc = thc
        # Normal-ordered mode (G0a is set): measure <N[V_F]> by evaluating the exactly
        # quadratic 2-body kernels on the FLUCTUATION dG = Ghalf - Ghalf_trial, and the
        # 1-body against the FOCK (rF1) instead of h.  This equals the '1h' LNO-MP2/CCSD
        # fragment-energy convention at the state level:  (t2 + t1 x t1)(2v - v) + 2 f.t1
        # restricted to i in F.  The RAW operator's mixed value additionally carries
        # singles mean-field dressing (fragment HF density x bath t1), which cancels in
        # sum_F (closure blind) but redistributes fragment<->bath at truncation --
        # the d333 shallow-fragment anomaly (predicted -0.0469 vs measured -0.0472, thr1).
        self.G0a = G0a; self.G0b = G0b
        self.rF1a = rF1a; self.rF1b = rF1b
        self._G0a_dev = None; self._G0b_dev = None   # backend-resident cache (one H2D total)
        self.scalar_estimator = True
        self._data = {"ENumer": 0.0j, "EDenom": 0.0j, "EFragCorr": 0.0j}
        self._shape = (len(self.names),)
        self._data_index = {k: i for i, k in enumerate(self._data)}
        self.print_to_stdout = True; self.ascii_filename = None

    def compute_estimator(self, system=None, walkers=None, hamiltonian=None, trial=None):
        trial.calc_greens_function(walkers)
        # xp.asarray keeps the walker densities on the active backend (cupy on GPU);
        # numpy.asarray on a cupy array raises (implicit host conversion forbidden).
        # rhf walkers (LNO_FAST_ESTIMATOR, closed shell): Ghalfb is never
        # recomputed -- beta == alpha; alias it and let frag_2body_thc take the
        # exact closed-shell shortcut (Ghalfb=None -> Btot=2Ba, exx_b=exx_a).
        rhf = bool(getattr(walkers, "rhf", False))
        Ga = xp.asarray(walkers.Ghalfa)
        Gb = Ga if rhf else xp.asarray(walkers.Ghalfb)
        normal_ordered = self.G0a is not None
        if normal_ordered:
            if self._G0a_dev is None:
                self._G0a_dev = xp.asarray(self.G0a)
                self._G0b_dev = xp.asarray(self.G0b if self.G0b is not None else self.G0a)
            Ga = Ga - self._G0a_dev[None]
            Gb = Ga if rhf else (Gb - self._G0b_dev[None])
        if self.thc:
            efrag = frag_2body_thc(hamiltonian, trial._thc_Xocca, trial._thc_Xoccb,
                                   Ga, None if rhf else Gb, self.frag)
        else:
            efrag = _frag_2body_cd(trial, walkers, self.frag)
        # + fragment 1-body coupling.  Normal-ordered mode: Fock against dG (its ov
        # block is the '1h' singles term 2 f.t1; f_ov == 0 for LNO exports, asserted
        # at setup).  Raw mode: h_frag against G with the HF part removed via e_ref.
        if normal_ordered:
            _ra, _rb = self.rF1a, self.rF1b
        else:
            _ra = getattr(trial, "_rH1a_frag", trial._rH1a)
            _rb = getattr(trial, "_rH1b_frag", trial._rH1b)
        if _ra is not None:
            efrag = efrag + frag_1body(_ra, _rb, Ga, Gb, self.frag)
        self._data["ENumer"] = xp.sum(walkers.weight * efrag.real)
        self._data["EDenom"] = xp.sum(walkers.weight)
        return self.data

    def post_reduce_hook(self, data):
        i = self._data_index
        data[i["EFragCorr"]] = data[i["ENumer"]] / data[i["EDenom"]] - self.e_ref


def run(which):
    thc = (which == "thc")
    M_prop = None
    prop_path = sys.argv[1].replace("_lno.h5", "_prop.h5")
    if prop_path != sys.argv[1] and os.path.exists(prop_path):
        with h5py.File(prop_path, "r") as fp:
            M_prop = numpy.array(fp["M"]).astype(numpy.float64)
        print(f"[metric] propagation M from {prop_path} (energy M from main h5)")
    base = GenericRealTHC(h1e, X, M, ecore=0.0, M_prop=M_prop)
    # energy estimator: cheap fragment 2-body by default; full cluster e2 when
    # LNO_ENERGY_FULL is set (=> ETotal-E_HF = canonical correlation, for the
    # LNO-vs-canonical validation).
    base.n_frag = nocc if os.environ.get("LNO_ENERGY_FULL") else n_frag_eff
    if thc:
        ham = base
    else:
        W = (X[:, None, :] * X[None, :, :]).reshape(nbasis * nbasis, Nmu)
        chol = (W @ base.zeta).astype(numpy.float64)
        ham = GenericRealChol(h1e, chol.copy(), ecore=0.0)
    trial = SingleDet(psi.copy(), nelec, nbasis)
    trial.build(); trial.half_rotate(ham)
    # Fragment-estimator one-body: Fock-referenced (h_frag = F - vHF) per the
    # 1h partition; falls back to H1 for legacy h5s (where H1 == F - vHF).
    with h5py.File(sys.argv[1], "r") as _f:
        _has_hfrag = "h_frag" in _f
        _hfrag = numpy.array(_f["h_frag"]).astype(numpy.float64) if _has_hfrag else None
    if _has_hfrag:
        if _hfrag.shape[0] != h1e.shape[0]: _hfrag = _hfrag.T.copy()
        trial._rH1a_frag = trial.psi0a.T @ _hfrag
        trial._rH1b_frag = trial.psi0b.T @ _hfrag
        print("[frag-1body] Fock-referenced h_frag loaded for EFragCorr")
    tw = _TrialWalkers(trial)
    if os.environ.get("LNO_EST_RAW"):
        # legacy raw-'1h'-operator estimator (carries singles mean-field dressing;
        # do NOT combine its EFragCorr with the '1h' MP2/CCSD composites)
        if thc:
            e_ref = float(frag_2body_thc(ham, trial._thc_Xocca, trial._thc_Xoccb,
                                         tw.Ghalfa, tw.Ghalfb, n_frag_eff)[0].real)
        else:
            e_ref = float(_frag_2body_cd(trial, tw, n_frag_eff)[0].real)
        _ra = getattr(trial, "_rH1a_frag", trial._rH1a)
        _rb = getattr(trial, "_rH1b_frag", trial._rH1b)
        e_ref += float(frag_1body(_ra, _rb, tw.Ghalfa, tw.Ghalfb, n_frag_eff)[0].real)
        print(f"[frag-est] RAW '1h' operator mode, e_ref = {e_ref:.8f}")
        lno_est = LNOFrag(n_frag_eff, e_ref, thc)
    else:
        # normal-ordered '1h' estimator (default): kernels on dG = G - G_trial,
        # 1-body against the trial Fock; matches the '1h' MP2/CCSD fragment
        # energy convention that the Delta composites assume.  e_ref = 0 (the
        # estimator reads exactly 0 at the trial).
        Xocc_t = trial.psi0a.T @ X                              # (nocc, Nmu)
        rho = 2.0 * numpy.einsum("im,im->m", Xocc_t, Xocc_t)    # closed-shell density
        Jm = (X * (M @ rho)[None, :]) @ X.T                     # (nbasis, nbasis)
        Ddm = 2.0 * (Xocc_t.T @ Xocc_t)                         # (Nmu, Nmu)
        Km = X @ (M * Ddm) @ X.T
        Fm = h1e[0] + Jm - 0.5 * Km    # h1e is spin-stacked (2, n, n)
        Pocc = trial.psi0a @ trial.psi0a.T
        fov_max = float(numpy.abs(trial.psi0a.T @ Fm @ (numpy.eye(nbasis) - Pocc)).max())
        print(f"[frag-est] NORMAL-ORDERED '1h' mode, |f_ov|max = {fov_max:.2e}"
              + ("  (WARNING: f_ov != 0 -- 1-body singles term active)" if fov_max > 1e-8 else ""))
        rFa = trial.psi0a.T @ Fm
        rFb = trial.psi0b.T @ Fm
        e_ref = 0.0
        # keep tw.Ghalf on whatever backend it lives on (cupy under IPIE_USE_GPU);
        # compute_estimator moves it with xp.asarray -- numpy.array(cupy) would raise
        lno_est = LNOFrag(n_frag_eff, 0.0, thc,
                          G0a=tw.Ghalfa[0], G0b=tw.Ghalfb[0],
                          rF1a=rFa, rF1b=rFb)
    afqmc = AFQMC.build(nelec, ham, trial, num_walkers=nw, num_blocks=nb, seed=SEED, timestep=DT, num_steps_per_block=int(os.environ.get("NSTEPS_BLK","25")))
    afqmc.run(additional_estimators={"lno": lno_est})
    afqmc.finalise(verbose=True)
    afqmc.finalise(verbose=False)
    return e_ref


mode = sys.argv[6] if len(sys.argv) > 6 else "both"   # both | thc | cd
eref_cd = eref_thc = None
if mode in ("both", "cd"):
    print("\n########## CD path (real h) ##########"); eref_cd = run("chol")
if mode in ("both", "thc"):
    print("\n########## THC path (real h) ##########"); eref_thc = run("thc")
print(f"\n[ref] HF fragment 2-body: CD={eref_cd}  THC={eref_thc}")
print("EFragCorr (last block) of the two 'lno' blocks = the THC vs CD fragment correlation energy.")
