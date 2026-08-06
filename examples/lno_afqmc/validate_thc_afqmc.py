"""End-to-end LNO-AFQMC: THC vs Cholesky on the SAME cluster ERI.

Loads the full-orbital THC factors X, M (thc_full_thr*.h5), builds:
  - GenericRealTHC(h, X, M)                          (factored THC path)
  - GenericRealChol(h, chol) with chol = W @ zeta    (same ERI, CD path)
Runs both AFQMC with identical seed/walkers/trial and compares the fragment
correlation energy EFragCorr.  Same ERI + validated ops => must agree to
statistical/floating-point precision.

Usage: python validate_thc_afqmc.py <thc_full.h5> [n_frag] [nwalkers] [nblocks]
"""
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
from ipie.lno_thc import frag_2body_thc
from ipie.qmc.afqmc import AFQMC
from ipie.trial_wavefunction.single_det import SingleDet
from ipie.utils.backend import arraylib as xp

path = sys.argv[1] if len(sys.argv) > 1 else \
    "/n/home12/aganeshram/scratch/d222_cache_geo/thc_full_thr0.h5"
n_frag = int(sys.argv[2]) if len(sys.argv) > 2 else 2
nw = int(sys.argv[3]) if len(sys.argv) > 3 else 40
nb = int(sys.argv[4]) if len(sys.argv) > 4 else 30
SEED = 7

with h5py.File(path, "r") as f:
    X = numpy.array(f["X"]).astype(numpy.float64)
    M = numpy.array(f["M"]).astype(numpy.float64)
    meta = numpy.array(f["meta"]).ravel()
nbasis, Nmu, nocc, nvir = int(meta[0]), int(meta[1]), int(meta[2]), int(meta[3])
if X.shape != (nbasis, Nmu):
    X = X.T.copy()
print(f"X {X.shape}, M {M.shape}, nbasis={nbasis} Nmu={Nmu} nocc={nocc} nvir={nvir}")

# A fixed random symmetric one-body (same for both paths; EFragCorr is 2-body).
rng = numpy.random.RandomState(11)
h1 = rng.randn(nbasis, nbasis); h1 = 0.5 * (h1 + h1.T)
h1e = numpy.array([h1, h1])
nelec = [nocc, nocc]

# reference determinant = first nocc orbitals (X ordered [occ | vir])
psi = numpy.zeros((nbasis, 2 * nocc))
psi[:nocc, :nocc] = numpy.eye(nocc)
psi[:nocc, nocc:] = numpy.eye(nocc)


class _TrialWalkers:
    def __init__(self, trial):
        self.Ghalfa = trial.Ghalf[0][None].astype(numpy.complex128)
        self.Ghalfb = trial.Ghalf[1][None].astype(numpy.complex128)
        self.weight = numpy.ones(1)
        self.nwalkers = 1


def _frag_2body_cd(trial, walkers, n_frag):
    naux = trial._rchola.shape[0]
    no = walkers.Ghalfa.shape[1]; nbf = walkers.Ghalfa.shape[2]
    rchola = trial._rchola.reshape(naux, no, nbf)
    rcholb = trial._rcholb.reshape(naux, no, nbf)
    Ga, Gb = walkers.Ghalfa, walkers.Ghalfb
    pin = slice(0, n_frag)
    ecoul = ecorrcoul_lno_real_rchol_uhf(rchola[:, pin], rchola, Ga[:, pin], Ga, Gb[:, pin], Gb)
    exx = ecorrxx_lno_real_rchol(rchola[:, pin], rchola, Ga[:, pin], Ga)
    exx += ecorrxx_lno_real_rchol(rcholb[:, pin], rcholb, Gb[:, pin], Gb)
    return (ecoul - exx).sum(axis=1)


class LNOFrag(EstimatorBase):
    def __init__(self, n_frag, e_ref, thc):
        super().__init__()
        self.n_frag = n_frag; self.e_ref = e_ref; self.thc = thc
        self.scalar_estimator = True
        self._data = {"ENumer": 0.0j, "EDenom": 0.0j, "EFragCorr": 0.0j}
        self._shape = (len(self.names),)
        self._data_index = {k: i for i, k in enumerate(self._data)}
        self.print_to_stdout = True; self.ascii_filename = None

    def compute_estimator(self, system=None, walkers=None, hamiltonian=None, trial=None):
        trial.calc_greens_function(walkers)
        if self.thc:
            Ga = numpy.asarray(walkers.Ghalfa); Gb = numpy.asarray(walkers.Ghalfb)
            efrag = frag_2body_thc(hamiltonian, trial._thc_Xocca, trial._thc_Xoccb, Ga, Gb, self.n_frag)
        else:
            efrag = _frag_2body_cd(trial, walkers, self.n_frag)
        self._data["ENumer"] = xp.sum(walkers.weight * efrag.real)
        self._data["EDenom"] = xp.sum(walkers.weight)
        return self.data

    def post_reduce_hook(self, data):
        i = self._data_index
        data[i["EFragCorr"]] = data[i["ENumer"]] / data[i["EDenom"]] - self.e_ref


def run(which):
    thc = (which == "thc")
    if thc:
        ham = GenericRealTHC(h1e, X, M, ecore=0.0)
    else:
        zeta = GenericRealTHC(h1e, X, M, ecore=0.0).zeta
        W = (X[:, None, :] * X[None, :, :]).reshape(nbasis * nbasis, Nmu)
        chol = (W @ zeta).astype(numpy.float64)
        ham = GenericRealChol(h1e, chol.copy(), ecore=0.0)
    trial = SingleDet(psi.copy(), nelec, nbasis)
    trial.build(); trial.half_rotate(ham)
    if thc:
        e_ref = float(frag_2body_thc(ham, trial._thc_Xocca, trial._thc_Xoccb,
                                     _TrialWalkers(trial).Ghalfa, _TrialWalkers(trial).Ghalfb,
                                     n_frag)[0].real)
    else:
        e_ref = float(_frag_2body_cd(trial, _TrialWalkers(trial), n_frag)[0].real)
    afqmc = AFQMC.build(nelec, ham, trial, num_walkers=nw, num_blocks=nb, seed=SEED)
    est = LNOFrag(n_frag, e_ref, thc)
    afqmc.run(additional_estimators={"lno": est})
    afqmc.finalise(verbose=False)
    return e_ref


print("\n########## CD path ##########")
eref_cd = run("chol")
print("\n########## THC path ##########")
eref_thc = run("thc")
print(f"\n[ref] HF fragment 2-body: CD={eref_cd:.10f}  THC={eref_thc:.10f}  diff={eref_cd-eref_thc:.2e}")
print("Compare the EFragCorr columns from the two 'lno' estimator blocks above "
      "(same seed/ERI -> should match to ~1e-8).")
