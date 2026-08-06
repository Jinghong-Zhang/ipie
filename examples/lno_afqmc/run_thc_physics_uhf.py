"""UHF LNO-AFQMC driver: PER-SPIN cluster bases with a SHARED auxiliary index.

Companion to run_thc_physics.py (which dispatches here for mode == "uhf"; the
RHF path is untouched).  Design (theory/lno_uhf.md section 6): the alpha and
beta walkers live in different orbital bases (n_a_orb vs n_b_orb) but the
two-body factorization shares one set of ISDF points mu and one central metric
M; only the collocations differ per spin (X_a, X_b).  One HS field x_mu couples
the TOTAL charge density.

H5 layouts accepted
-------------------
1. Per-spin export (preferred; groups "a" and "b" present):
     a/X (n_a_orb x Nmu)  or transposed     a/h_pq (n_a_orb x n_a_orb)
     b/X (n_b_orb x Nmu)                    b/h_pq (n_b_orb x n_b_orb)
     Mcore (or M) (Nmu x Nmu)               shared metric
     meta = [n_a_orb, n_b_orb, Nmu, nocc_a, nocc_b, (n_frag)]
     a/h_frag, b/h_frag (optional)          Fock-referenced fragment one-body
2. DEBUG SYNTH mode (per-spin groups absent, or LNO_UHF_SYNTH=1): a closed-shell
   export (top-level X, M, h_pq, meta) is duplicated into both spins.  This is
   the closed-shell-UHF consistency gate: results must agree with the RHF driver
   at the statistics level (and bit-for-bit per trajectory at fixed seed).

The rhf fast path is never used: LNO_FAST_ESTIMATOR is explicitly stripped and
the walkers/trial classes are genuinely two-spin.
"""
import os
import sys

import numpy
import h5py

from ipie.config import MPI
from ipie.estimators.estimator_base import EstimatorBase
from ipie.hamiltonians.thc import GenericRealTHCUhf
from ipie.lno_thc import _array_module, _as_rot
from ipie.lno_thc_uhf import frag_2body_thc_uhf
from ipie.qmc.afqmc import AFQMC
from ipie.trial_wavefunction.single_det_uhf_basis import SingleDetUhfBasis
from ipie.utils.backend import arraylib as xp
from ipie.utils.mpi import MPIHandler
from ipie.walkers.uhf_walkers_spinbasis import UHFWalkersSpinBasis


def _frag_1body_spin(rF1, Ghalf, frag):
    """One-spin fragment-projected 1-body coupling (Fock ov block against dG).
    Same math as run_thc_physics.frag_1body but for a single spin sector with
    its own fragment rotation."""
    xpf = _array_module(Ghalf)
    Bf = _as_rot(frag, Ghalf.shape[1], like=Ghalf)               # (nocc_s, nfrag_s)
    rA = Bf.T @ xpf.asarray(rF1)                                 # (nfrag_s, n_s_orb)
    return xpf.einsum("fi,wij,fj->w", Bf.T, Ghalf, rA, optimize=True)


class LNOMullikenUhf(EstimatorBase):
    """Fragment-projected correlation correction to the Mulliken SPIN population.

    The paper's magnetization (arXiv:2602.16679 Eq. 31/32) is the expectation of
    a ONE-BODY operator O_A = 0.5(Pi_A S + S Pi_A), with a + sign for alpha and a
    - sign for beta.  qcpbc exports it already rotated into each spin's cluster
    basis as a/mulliken_A, b/mulliken_A.

    Because <O> = Tr[O gamma] is linear in the 1RDM, the LNO decomposition of the
    energy carries over verbatim:

        m_A(T) = m_A^UHF + sum_{s,iF} w^s_iF <Q_F O_A>^{s,iF}_corr + Delta_M(T)

    IMPORTANT -- there is NO 0.5 here.  The energy's both-spin-lists factor comes
    from double counting opposite-spin PAIRS (two occupied indices, pinned once
    on each).  A one-body operator has a single index to pin, so summing the
    spin-s fragment list covers the spin-s occupied space exactly once and the
    two spin lists contribute to m with opposite signs rather than redundantly.

    This is a MIXED estimator on the normal-ordered dG (ipie has no back
    propagation), so it is biased at first order in the trial error; the driver
    also reports enough to form the extrapolated estimator 2<O>_mix - <O>_HF.
    The paper instead used a relaxed finite-field response estimator with
    correlated sampling -- a caveat to carry, not a blocker, since the whole
    correlation correction is only ~5% of m (HF 1.78 -> AFQMC 1.69)."""

    def __init__(self, frag_a, frag_b, G0a, G0b, rOa, rOb):
        super().__init__()
        self.frag_a = frag_a
        self.frag_b = frag_b
        self.G0a = G0a
        self.G0b = G0b
        self.rOa = rOa
        self.rOb = rOb
        self.scalar_estimator = True
        self._data = {"MNumer": 0.0j, "MDenom": 0.0j, "MFragSpin": 0.0j}
        self._shape = (len(self.names),)
        self._data_index = {k: i for i, k in enumerate(self._data)}
        self.print_to_stdout = True
        self.ascii_filename = None

    def compute_estimator(self, system=None, walkers=None, hamiltonian=None, trial=None):
        trial.calc_greens_function(walkers)
        m = None
        if self.frag_a is not None:
            dGa = xp.asarray(walkers.Ghalfa) - xp.asarray(self.G0a)[None]
            m = _frag_1body_spin(self.rOa, dGa, self.frag_a)          # + alpha
        if self.frag_b is not None:
            dGb = xp.asarray(walkers.Ghalfb) - xp.asarray(self.G0b)[None]
            t = _frag_1body_spin(self.rOb, dGb, self.frag_b)          # - beta
            m = -t if m is None else m - t
        self._data["MNumer"] = xp.sum(walkers.weight * m.real)
        self._data["MDenom"] = xp.sum(walkers.weight)
        return self.data

    def post_reduce_hook(self, data):
        i = self._data_index
        data[i["MFragSpin"]] = data[i["MNumer"]] / data[i["MDenom"]]


class LNOFragUhf(EstimatorBase):
    """Per-(spin, fragment) normal-ordered '1h' fragment estimator.

    Kernels are evaluated on the fluctuation dG^s = Ghalf^s - Ghalf^s_trial and
    the 1-body against the per-spin trial Fock (rF1a/rF1b); e_ref = 0 (the
    estimator reads exactly 0 at the trial).  frag_a / frag_b pin the fragment
    occupieds of each spin; pass None to skip a spin's pinned terms (the
    per-sigma estimator of theory/lno_uhf.md section 6).  The Coulomb kernel
    always couples to the TOTAL density of both spins."""

    def __init__(self, frag_a, frag_b, G0a, G0b, rF1a, rF1b):
        super().__init__()
        self.frag_a = frag_a
        self.frag_b = frag_b
        self.G0a = G0a
        self.G0b = G0b
        self.rF1a = rF1a
        self.rF1b = rF1b
        self.scalar_estimator = True
        self._data = {"ENumer": 0.0j, "EDenom": 0.0j, "EFragCorr": 0.0j}
        self._shape = (len(self.names),)
        self._data_index = {k: i for i, k in enumerate(self._data)}
        self.print_to_stdout = True
        self.ascii_filename = None

    def compute_estimator(self, system=None, walkers=None, hamiltonian=None, trial=None):
        trial.calc_greens_function(walkers)
        dGa = xp.asarray(walkers.Ghalfa) - xp.asarray(self.G0a)[None]
        dGb = xp.asarray(walkers.Ghalfb) - xp.asarray(self.G0b)[None]
        efrag = frag_2body_thc_uhf(
            hamiltonian, trial._thc_Xocca, trial._thc_Xoccb, dGa, dGb,
            self.frag_a, self.frag_b,
        )
        if self.frag_a is not None and self.rF1a is not None:
            efrag = efrag + _frag_1body_spin(self.rF1a, dGa, self.frag_a)
        if self.frag_b is not None and self.rF1b is not None:
            efrag = efrag + _frag_1body_spin(self.rF1b, dGb, self.frag_b)
        self._data["ENumer"] = xp.sum(walkers.weight * efrag.real)
        self._data["EDenom"] = xp.sum(walkers.weight)
        return self.data

    def post_reduce_hook(self, data):
        i = self._data_index
        data[i["EFragCorr"]] = data[i["ENumer"]] / data[i["EDenom"]]


def _maybe_T(A, nrow, ncol):
    """Return A oriented (nrow x ncol), transposing if the export is flipped."""
    if A.shape == (nrow, ncol):
        return A
    assert A.shape == (ncol, nrow), f"unexpected shape {A.shape} for ({nrow},{ncol})"
    return numpy.ascontiguousarray(A.T)


def load_uhf_problem(path, n_frag_default):
    """Read a per-spin export, or synthesize one from a closed-shell h5."""
    with h5py.File(path, "r") as f:
        per_spin = ("a" in f and "b" in f) and not os.environ.get("LNO_UHF_SYNTH")
        if per_spin:
            meta = numpy.array(f["meta"]).ravel()
            n_a, n_b, nmu = int(meta[0]), int(meta[1]), int(meta[2])
            nocc_a, nocc_b = int(meta[3]), int(meta[4])
            n_frag = int(round(meta[5])) if len(meta) > 5 else n_frag_default
            # meta[6] = ispin: WHICH spin owns this fragment.  The exporter builds
            # the cluster for one (sigma, iF) pair -- sector `ispin` holds the
            # fragment IBOs first (postscf_lnothc.C:1962-1968), the opposite
            # sector holds a Dij bath.  Pinning the bath sector would be
            # meaningless, so the estimator pins `ispin` only.
            ispin = int(round(meta[6])) if len(meta) > 6 else None
            Xa = _maybe_T(numpy.array(f["a/X"]).astype(numpy.float64), n_a, nmu)
            Xb = _maybe_T(numpy.array(f["b/X"]).astype(numpy.float64), n_b, nmu)
            key_M = "Mcore" if "Mcore" in f else "M"
            M = numpy.array(f[key_M]).astype(numpy.float64)
            h_a = _maybe_T(numpy.array(f["a/h_pq"]).astype(numpy.float64), n_a, n_a)
            h_b = _maybe_T(numpy.array(f["b/h_pq"]).astype(numpy.float64), n_b, n_b)
            hf_a = hf_b = None
            if "a/h_frag" in f:
                hf_a = _maybe_T(numpy.array(f["a/h_frag"]).astype(numpy.float64), n_a, n_a)
                hf_b = _maybe_T(numpy.array(f["b/h_frag"]).astype(numpy.float64), n_b, n_b)
            # Mulliken SPIN operator on one atom, already in each spin's cluster
            # basis (qcpbc $rem lno_mulliken_atom).  Optional: absent -> no
            # magnetization estimator is registered.
            mull_a = mull_b = None
            if "a/mulliken_A" in f and "b/mulliken_A" in f:
                mull_a = _maybe_T(numpy.array(f["a/mulliken_A"]).astype(numpy.float64), n_a, n_a)
                mull_b = _maybe_T(numpy.array(f["b/mulliken_A"]).astype(numpy.float64), n_b, n_b)
            print(f"[uhf] per-spin export: n_a={n_a} n_b={n_b} Nmu={nmu} "
                  f"nocc_a={nocc_a} nocc_b={nocc_b} n_frag={n_frag} "
                  f"ispin={'?' if ispin is None else ('a', 'b')[ispin]}")
        else:
            # DEBUG SYNTH: duplicate the closed-shell export into both spins.
            meta = numpy.array(f["meta"]).ravel()
            nbasis, nmu, nocc = int(meta[0]), int(meta[1]), int(meta[2])
            n_frag = int(round(meta[5])) if len(meta) > 5 else n_frag_default
            X = _maybe_T(numpy.array(f["X"]).astype(numpy.float64), nbasis, nmu)
            M = numpy.array(f["M"]).astype(numpy.float64)
            assert "h_pq" in f, (
                "UHF synth mode needs the thin-consumer export (h_pq present); "
                "the legacy pseudocanonical layout is not supported here."
            )
            h = _maybe_T(numpy.array(f["h_pq"]).astype(numpy.float64), nbasis, nbasis)
            Xa = Xb = X
            h_a = h_b = h
            n_a = n_b = nbasis
            nocc_a = nocc_b = nocc
            ispin = None                     # closed-shell: fragment lives in both
            hf_a = hf_b = None
            mull_a = mull_b = None
            if "h_frag" in f:
                hf = _maybe_T(numpy.array(f["h_frag"]).astype(numpy.float64), nbasis, nbasis)
                hf_a = hf_b = hf
            print(f"[uhf] SYNTH mode (closed-shell h5 duplicated per spin): "
                  f"n_a=n_b={n_a} Nmu={nmu} nocc_a=nocc_b={nocc_a} n_frag={n_frag}")
    return dict(Xa=Xa, Xb=Xb, M=M, h_a=h_a, h_b=h_b, hf_a=hf_a, hf_b=hf_b,
                n_a=n_a, n_b=n_b, nocc_a=nocc_a, nocc_b=nocc_b, n_frag=n_frag,
                ispin=ispin, mull_a=mull_a, mull_b=mull_b)


def main(path, n_frag=2, nw=40, nb=30, dt=0.005, seed=7):
    # The closed-shell fast path must never engage on the UHF path (it would
    # alias the beta sector onto alpha).
    if os.environ.pop("LNO_FAST_ESTIMATOR", None) is not None:
        print("[uhf] WARNING: LNO_FAST_ESTIMATOR stripped (rhf fast path is "
              "invalid with per-spin bases)")
    os.environ.pop("IPIE_THC_MIXED", None)   # mixed precision unsupported on UHF path

    P = load_uhf_problem(path, n_frag)
    Xa, Xb, M = P["Xa"], P["Xb"], P["M"]
    n_a, n_b = P["n_a"], P["n_b"]
    nocc_a, nocc_b = P["nocc_a"], P["nocc_b"]
    n_frag = P["n_frag"]
    nelec = (nocc_a, nocc_b)

    # Propagation metric (G=0-free) if the companion export exists, as in RHF.
    M_prop = None
    prop_path = path.replace("_lno.h5", "_prop.h5")
    if prop_path != path and os.path.exists(prop_path):
        with h5py.File(prop_path, "r") as fp:
            M_prop = numpy.array(fp["M"]).astype(numpy.float64)
        print(f"[uhf] propagation M from {prop_path} (energy M from main h5)")

    ham = GenericRealTHCUhf(P["h_a"], P["h_b"], Xa, Xb, M, ecore=0.0,
                            M_prop=M_prop, verbose=True)
    ham.n_frag = n_frag

    # Identity trial determinants in each spin's own (pseudocanonical cluster) basis.
    psia = numpy.zeros((n_a, nocc_a)); psia[:nocc_a, :nocc_a] = numpy.eye(nocc_a)
    psib = numpy.zeros((n_b, nocc_b)); psib[:nocc_b, :nocc_b] = numpy.eye(nocc_b)
    trial = SingleDetUhfBasis(psia, psib, nelec)
    trial.build()
    trial.half_rotate(ham)

    # ---- per-spin trial Fock / density (replaces the closed-shell 2.0* lines):
    #      rho = rho_a + rho_b enters J (shared mu grid); K is per spin. ----
    Xocc_a = trial._thc_Xocca                                    # (nocc_a, Nmu)
    Xocc_b = trial._thc_Xoccb
    rho = numpy.einsum("im,im->m", Xocc_a, Xocc_a) + \
        numpy.einsum("im,im->m", Xocc_b, Xocc_b)                 # TOTAL density (no 2.0)
    Mrho = M @ rho
    rF = {}
    fov = {}
    for spin, (X_s, h_s, Xo_s, psi_s) in enumerate(
        ((Xa, P["h_a"], Xocc_a, psia), (Xb, P["h_b"], Xocc_b, psib))
    ):
        J_s = (X_s * Mrho[None, :]) @ X_s.T                      # J[rho_a + rho_b]
        D_s = Xo_s.T @ Xo_s                                      # (Nmu, Nmu), per spin
        K_s = X_s @ (M * D_s) @ X_s.T                            # K^sigma (no 0.5)
        F_s = h_s + J_s - K_s
        Pocc = psi_s @ psi_s.T
        fov[spin] = float(numpy.abs(psi_s.T @ F_s @ (numpy.eye(F_s.shape[0]) - Pocc)).max())
        rF[spin] = psi_s.T @ F_s                                 # (nocc_s, n_s_orb)
    print(f"[uhf frag-est] NORMAL-ORDERED '1h' mode, |f_ov|max = "
          f"a: {fov[0]:.2e}  b: {fov[1]:.2e}"
          + ("  (WARNING: f_ov != 0 -- 1-body singles term active)"
             if max(fov.values()) > 1e-8 else ""))

    G0a, G0b = trial.Ghalf[0], trial.Ghalf[1]

    # ---- fragment pinning ----------------------------------------------------
    # An export is built for ONE (sigma, iF) pair: sector `ispin` carries the
    # fragment IBOs in its leading nocc_F occupied columns, the opposite sector
    # carries a Dij bath (postscf_lnothc.C:1962-1968, :2015-2018).  Pinning the
    # bath sector would measure a bath "fragment" that does not exist, so we pin
    # `ispin` only; the opposite-spin occupied index is summed freely inside
    # frag_2body_thc_uhf via the TOTAL density rho_tot.  ONE estimator per file.
    #
    # Summing the resulting EFragCorr over BOTH spin fragment lists covers every
    # opposite-spin pair (i in alpha, j in beta) twice -- once pinned on i, once
    # pinned on j -- which is exactly the double counting qcpbc removes with the
    # 0.5 on E_os(embed,raw) (postscf_lnothc.C:2238-2243).  The same 0.5 must
    # therefore be applied to the AFQMC fragment sum; extrap_uhf.py owns it as an
    # explicit knob and gate G1 (closed-shell limit) arbitrates its value.
    ispin = P["ispin"]
    if ispin is None:
        # Closed-shell / SYNTH: the spatial fragment lives in both spin sectors,
        # matching the RHF driver's frag_1body/frag_2body (which pin the same Bf
        # in alpha and beta within a single estimator).  One estimator, both pins.
        frag_a = min(n_frag, nocc_a)
        frag_b = min(n_frag, nocc_b)
        print(f"[uhf frag-est] closed-shell pinning: frag_a={frag_a} frag_b={frag_b} "
              "(RHF-comparable; EFragCorr should match the RHF driver)")
    else:
        frag_a = min(n_frag, nocc_a) if ispin == 0 else None
        frag_b = min(n_frag, nocc_b) if ispin == 1 else None
        print(f"[uhf frag-est] open-shell pinning: spin={('a', 'b')[ispin]} "
              f"nocc_F={n_frag} (opposite sector unpinned, summed via rho_tot)")
    est = {"lno": LNOFragUhf(frag_a, frag_b, G0a, G0b, rF[0], rF[1])}

    # ---- magnetization (optional; only if qcpbc exported the spin operator) ----
    if P["mull_a"] is not None:
        rOa = psia.T @ P["mull_a"]                               # (nocc_a, n_a_orb)
        rOb = psib.T @ P["mull_b"]
        # HF/trial reference: Tr[G0^s O^s] per spin, alpha - beta.  Printed so the
        # analysis can form m = m_HF + <dG term> and the extrapolated estimator.
        m_trial = float(numpy.einsum("ij,ij->", G0a, rOa)
                        - numpy.einsum("ij,ij->", G0b, rOb))
        print(f"[uhf mulliken] trial (HF) cluster spin population = {m_trial:+.6f}  "
              f"(the estimator reports the CORRELATION correction on dG; "
              f"no 0.5 -- one-body operators are not pair-double-counted)")
        est["mull"] = LNOMullikenUhf(frag_a, frag_b, G0a, G0b, rOa, rOb)

    # Walkers with per-spin bases, passed explicitly (the factory in
    # AFQMC.build assumes one shared basis).
    mpih = MPIHandler()
    walkers = UHFWalkersSpinBasis(psia.astype(numpy.float64), psib.astype(numpy.float64),
                                  nocc_a, nocc_b, n_a, n_b, nw, mpih)
    walkers.build(trial)
    assert walkers.rhf is False

    afqmc = AFQMC.build(nelec, ham, trial, walkers=walkers, num_walkers=nw,
                        num_blocks=nb, seed=seed, timestep=dt, mpi_handler=mpih)
    afqmc.run(additional_estimators=est)
    afqmc.finalise(verbose=False)
    print("[uhf] EFragCorr of block 'lno' = this (spin, fragment) correlation "
          "energy (normal-ordered '1h'; e_ref = 0).  The composite applies the "
          "0.5 both-spin-lists factor in extrap_uhf.py, not here.")


if __name__ == "__main__":
    _path = sys.argv[1]
    _n_frag = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    _nw = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    _nb = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    _dt = float(sys.argv[5]) if len(sys.argv) > 5 else 0.005
    main(_path, n_frag=_n_frag, nw=_nw, nb=_nb, dt=_dt,
         seed=int(os.environ.get("SEED", 7)))
