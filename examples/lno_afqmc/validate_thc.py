"""Validate the factored THC LNO-AFQMC ops against the Cholesky (CD) path.

Loads the full-orbital THC factors X, M exported by qcpbc lno_isdf
(thc_full_thr*.h5) and builds the EQUIVALENT Cholesky factor
    chol_pq^gamma = sum_mu X[p,mu] X[q,mu] zeta[mu,gamma],   M = zeta zeta^T,
i.e. the same factorization the THC keeps factored.  CD and THC therefore
represent the SAME two-body operator, so every op must agree to ~1e-12 --
isolating any THC-algebra error from THC truncation.

Checks:
  (C) field decomposition exact:  sum_gamma L^gamma L^gamma == (pq|rs)
  (B) h1e_mod v0:                 THC v0 vs 0.5 sum_k (ik|jk)
  (D) mean-field one-body shift:  THC vs dense  sum_rs (pq|rs) G^charge_rs
  (E) VHS:                        THC vs CD-from-same-zeta for the same fields
  (F) LNO fragment energy:        THC ecoul/exx vs the existing CD kernels

Usage:  python validate_thc.py <thc_full.h5> [n_frag] [nwalkers]
"""
import sys
import numpy
import scipy.linalg
import h5py

from ipie.hamiltonians.generic import GenericRealChol
from ipie.hamiltonians.thc import GenericRealTHC
from ipie.lno_thc import (
    construct_mean_field_shift_thc,
    construct_one_body_propagator_thc,
    construct_VHS_thc,
    construct_force_bias_thc,
    lno_coul_thc,
    lno_exx_thc,
)
from ipie.estimators.local_energy_sd import (
    ecorrcoul_lno_real_rchol_uhf,
    ecorrxx_lno_real_rchol,
)

numpy.random.seed(7)
path = sys.argv[1] if len(sys.argv) > 1 else \
    "/n/home12/aganeshram/scratch/d222_cache_geo/thc_full_thr0.h5"

with h5py.File(path, "r") as f:
    X = numpy.array(f["X"]).astype(numpy.float64)      # (nbasis x Nmu)
    M = numpy.array(f["M"]).astype(numpy.float64)      # (Nmu x Nmu)
    meta = numpy.array(f["meta"]).ravel()
# arma writes column-major; h5py may transpose. Enforce X: (nbasis, Nmu).
nbasis, Nmu, nocc, nvir = int(meta[0]), int(meta[1]), int(meta[2]), int(meta[3])
if X.shape != (nbasis, Nmu):
    X = X.T.copy()
assert X.shape == (nbasis, Nmu), f"X shape {X.shape} != ({nbasis},{Nmu})"
assert M.shape == (Nmu, Nmu)
print(f"loaded X {X.shape}, M {M.shape}  (nbasis={nbasis} Nmu={Nmu} nocc={nocc} nvir={nvir})")

n_frag = int(sys.argv[2]) if len(sys.argv) > 2 else min(nocc, 2)
nw = int(sys.argv[3]) if len(sys.argv) > 3 else 3

# ---- THC Hamiltonian (random symmetric h1e just for h1e_mod/one-body checks) ----
h1 = numpy.random.randn(nbasis, nbasis); h1 = 0.5 * (h1 + h1.T)
h1e = numpy.array([h1, h1])
ham_thc = GenericRealTHC(h1e, X, M, ecore=0.0, verbose=True)
zeta = ham_thc.zeta                                    # (Nmu x Ngamma)
ng = ham_thc.nfields

# ---- equivalent CD factor chol = W @ zeta, W[(p,q),mu] = X[p,mu] X[q,mu] ----
W = (X[:, None, :] * X[None, :, :]).reshape(nbasis * nbasis, Nmu)   # (M^2 x Nmu)
chol = (W @ zeta).astype(numpy.float64)                            # (M^2 x Ngamma)
ham_cd = GenericRealChol(h1e, chol.copy(), ecore=0.0, verbose=False)

# dense ERI V[p,q,r,s] = sum_gamma chol[(pq),g] chol[(rs),g]
V = (chol @ chol.T).reshape(nbasis, nbasis, nbasis, nbasis)

def rel(a, b):
    d = numpy.linalg.norm(numpy.asarray(a) - numpy.asarray(b))
    n = numpy.linalg.norm(numpy.asarray(b))
    return d / max(n, 1e-300)

print("\n=== checks (THC factored vs CD/dense, same factorization) ===")

# (C) field decomposition exact (by construction M = zeta zeta^T)
Vthc = numpy.einsum("pm,qm,mn,rn,sn->pqrs", X, X, M, X, X, optimize=True)
print(f"(C) sum_g L^g L^g == (pq|rs):           rel = {rel(Vthc, V):.3e}")

# (B) h1e_mod: v0 = 0.5 sum_k (ik|jk)
v0_thc = h1 - numpy.asarray(ham_thc.h1e_mod)[0]
v0_dense = 0.5 * numpy.einsum("ikjk->ij", V, optimize=True)
print(f"(B) h1e_mod v0 (THC vs dense):          rel = {rel(v0_thc, v0_dense):.3e}")

# build a random RHF-like trial (orthonormal occ) and a fake trial Green's function
C = numpy.linalg.qr(numpy.random.randn(nbasis, nbasis))[0]
psi_occ = C[:, :nocc]                                  # (nbasis x nocc)
Gt = psi_occ @ psi_occ.T                               # (nbasis x nbasis) idempotent
class _T:                                              # minimal trial stub
    pass
trial = _T(); trial.G = numpy.array([Gt, Gt]); trial.psi0a = psi_occ; trial.psi0b = psi_occ

# (D) mean-field shift mf and the one-body propagator, THC vs CD (same factorization)
mf_thc = construct_mean_field_shift_thc(ham_thc, trial)
Gch = (trial.G[0] + trial.G[1]).ravel()
mf_cd = 1j * (chol.T @ Gch.real) - (chol.T @ Gch.imag)     # CD construct_mean_field_shift
print(f"(D) mean-field shift mf (THC vs CD):    rel = {rel(mf_thc, mf_cd):.3e}")
dt = 0.01
expH1_thc = construct_one_body_propagator_thc(ham_thc, mf_thc, dt)
shift_cd = 1j * (chol @ mf_cd).reshape(nbasis, nbasis)
H1_cd = numpy.asarray(ham_cd.h1e_mod) - numpy.array([shift_cd, shift_cd])
expH1_cd = numpy.array(
    [scipy.linalg.expm(-0.5 * dt * H1_cd[0]), scipy.linalg.expm(-0.5 * dt * H1_cd[1])])
print(f"(D') one-body expH1 (THC vs CD):        rel = {rel(expH1_thc, expH1_cd):.3e}")

# (E) VHS: THC vs CD (same zeta) for identical random fields
xshift = (numpy.random.randn(ng, nw) + 1j * numpy.random.randn(ng, nw))
isqrt_dt = 1j * numpy.sqrt(0.01)
VHS_thc = construct_VHS_thc(ham_thc, xshift, isqrt_dt)
VHS_cd = numpy.zeros_like(VHS_thc)
for w in range(nw):
    VHS_cd[w] = isqrt_dt * (chol @ xshift[:, w]).reshape(nbasis, nbasis)
print(f"(E) VHS (THC vs CD, same fields):       rel = {rel(VHS_thc, VHS_cd):.3e}")

# (F) LNO fragment energy: THC kernels vs the existing CD kernels on the same ERI.
# random walker Ghalf (nw x nocc x nbasis), complex
Ga = numpy.random.randn(nw, nocc, nbasis) + 1j * numpy.random.randn(nw, nocc, nbasis)
Gb = numpy.random.randn(nw, nocc, nbasis) + 1j * numpy.random.randn(nw, nocc, nbasis)
# CD half-rotated cholesky rchola[g, i, b] = sum_p psi_pi chol_3[p, b, g]
chol3 = chol.reshape(nbasis, nbasis, ng)
# rchola is REAL (real orbitals + real chol); the CD ..._real_rchol kernels require it.
rchola = numpy.einsum("pi,pbg->gib", psi_occ, chol3, optimize=True).astype(numpy.float64)
Xocc = ham_thc.half_rotate(psi_occ)                    # (nocc x Nmu)

ecoul_cd = ecorrcoul_lno_real_rchol_uhf(
    rchola[:, :n_frag], rchola, Ga[:, :n_frag], Ga, Gb[:, :n_frag], Gb)
ecoul_thc = lno_coul_thc(ham_thc, Xocc, Ga, Gb, n_frag)
print(f"(F) LNO Coulomb  (THC vs CD kernel):    rel = {rel(ecoul_thc, ecoul_cd):.3e}")

exx_cd = ecorrxx_lno_real_rchol(rchola[:, :n_frag], rchola, Ga[:, :n_frag], Ga)
exx_thc = lno_exx_thc(ham_thc, Xocc, Ga, n_frag)
print(f"(F) LNO exchange (THC vs CD kernel):    rel = {rel(exx_thc, exx_cd):.3e}")

print("\n=== done (all rel errors should be ~1e-12) ===")
