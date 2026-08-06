"""Gate test for the per-spin-basis THC-UHF path (GenericRealTHCUhf).

(a) SYMMETRIC: X_a == X_b, h_a == h_b, na == nb, duplicated trial.  The UHF
    class must reproduce GenericRealTHC's v0 (h1e_mod), mean-field shift,
    force bias, and -- after identical seeded propagation steps -- the local
    energy and the fragment (EFragCorr-style) estimator, all to 1e-12.  The
    RHF side runs with the rhf fast path DISABLED (walkers.rhf = False, no
    LNO_FAST_ESTIMATOR) so beta is genuinely propagated: apples-to-apples.

(b) ASYMMETRIC: n_a_orb != n_b_orb, na != nb.  Init + several propagation
    steps + local energy + fragment estimator must stay finite (no NaN/Inf).

Pure CPU numpy, no MPI required:  python3 test_thc_uhf_gate.py
"""
import os
import sys

import numpy

# The RHF comparison must not take any closed-shell shortcut.
os.environ.pop("LNO_FAST_ESTIMATOR", None)
os.environ.pop("IPIE_THC_MIXED", None)

from ipie.hamiltonians.thc import GenericRealTHC, GenericRealTHCUhf
from ipie.lno_thc import construct_force_bias_thc, construct_mean_field_shift_thc, frag_2body_thc
from ipie.lno_thc_uhf import (
    construct_force_bias_thc_uhf,
    construct_mean_field_shift_thc_uhf,
    frag_2body_thc_uhf,
    local_energy_thc_uhf,
)
from ipie.estimators.energy import local_energy
from ipie.propagation.phaseless_generic import PhaselessGeneric
from ipie.systems.generic import Generic
from ipie.trial_wavefunction.single_det import SingleDet
from ipie.trial_wavefunction.single_det_uhf_basis import SingleDetUhfBasis
from ipie.utils.mpi import MPIHandler
from ipie.walkers.uhf_walkers import UHFWalkers
from ipie.walkers.uhf_walkers_spinbasis import UHFWalkersSpinBasis

TOL = 1e-12
NW = 6          # walkers
DT = 0.005
NFRAG = 2
failures = []


def check(name, err, tol=TOL):
    ok = err < tol
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<52s} max|diff| = {err:.3e}")
    if not ok:
        failures.append(name)


def finite(name, arr):
    ok = bool(numpy.all(numpy.isfinite(numpy.asarray(arr))))
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<52s} finite = {ok}")
    if not ok:
        failures.append(name)


def make_problem(rng, n_orb, nmu, nocc):
    X = rng.standard_normal((n_orb, nmu)) * 0.3
    A = rng.standard_normal((nmu, nmu)) * 0.2
    M = A @ A.T + 0.1 * numpy.eye(nmu)          # PSD
    h = rng.standard_normal((n_orb, n_orb)) * 0.1
    h = 0.5 * (h + h.T) + numpy.diag(numpy.linspace(-1.0, 1.0, n_orb))
    psi = numpy.zeros((n_orb, nocc))
    psi[:nocc, :nocc] = numpy.eye(nocc)
    return X, M, h, psi


# ======================================================================
print("== (a) SYMMETRIC gate: THC-UHF with X_a == X_b vs GenericRealTHC ==")
rng = numpy.random.default_rng(12345)
n_orb, nmu, nocc = 8, 20, 3
X, M, h, psi_occ = make_problem(rng, n_orb, nmu, nocc)
nelec = (nocc, nocc)
mpih = MPIHandler()

# ---- RHF reference (rhf fast path disabled) ----
ham_r = GenericRealTHC(numpy.array([h, h]), X, M, ecore=0.0)
psi_stack = numpy.hstack([psi_occ, psi_occ])
trial_r = SingleDet(psi_stack.copy(), nelec, n_orb)
trial_r.build()
trial_r.half_rotate(ham_r)
walkers_r = UHFWalkers(psi_stack.copy(), nocc, nocc, n_orb, NW, mpih)
walkers_r.build(trial_r)
assert walkers_r.rhf is False, "gate requires the rhf fast path disabled"

# ---- UHF side (duplicated everything) ----
ham_u = GenericRealTHCUhf(h, h, X, X, M, ecore=0.0)
trial_u = SingleDetUhfBasis(psi_occ.copy(), psi_occ.copy(), nelec)
trial_u.build()
trial_u.half_rotate(ham_u)
walkers_u = UHFWalkersSpinBasis(psi_occ.copy(), psi_occ.copy(), nocc, nocc,
                                n_orb, n_orb, NW, mpih)
walkers_u.build(trial_u)

# v0 / h1e_mod
check("v0 (h1e_mod alpha)", numpy.abs(numpy.asarray(ham_u.h1e_mod[0]) - numpy.asarray(ham_r.h1e_mod[0])).max())
check("v0 (h1e_mod beta)", numpy.abs(numpy.asarray(ham_u.h1e_mod[1]) - numpy.asarray(ham_r.h1e_mod[1])).max())
check("zeta (shared fields)", numpy.abs(ham_u.zeta - ham_r.zeta).max())

# mean-field shift
mf_r = construct_mean_field_shift_thc(ham_r, trial_r)
mf_u = construct_mean_field_shift_thc_uhf(ham_u, trial_u)
check("mean-field shift", numpy.abs(mf_u - mf_r).max())

# force bias on the initial (trial-identical) walkers
vb_r = construct_force_bias_thc(ham_r, trial_r._thc_Xocca, trial_r._thc_Xoccb,
                                walkers_r.Ghalfa, walkers_r.Ghalfb)
vb_u = construct_force_bias_thc_uhf(ham_u, trial_u._thc_Xocca, trial_u._thc_Xoccb,
                                    walkers_u.Ghalfa, walkers_u.Ghalfb)
check("force bias (initial walkers)", numpy.abs(vb_u - vb_r).max())

# ---- identical seeded propagation: 3 steps through the full phaseless kernel ----
prop_r = PhaselessGeneric(DT)
prop_r.build(ham_r, trial_r, walkers_r)
prop_u = PhaselessGeneric(DT)
prop_u.build(ham_u, trial_u, walkers_u)
check("one-body propagator (alpha)", numpy.abs(numpy.asarray(prop_u.expH1[0]) - numpy.asarray(prop_r.expH1[0])).max())
check("one-body propagator (beta)", numpy.abs(numpy.asarray(prop_u.expH1[1]) - numpy.asarray(prop_r.expH1[1])).max())

for step in range(3):
    numpy.random.seed(777 + step)
    prop_r.propagate_walkers(walkers_r, ham_r, trial_r, 0.0)
    numpy.random.seed(777 + step)
    prop_u.propagate_walkers(walkers_u, ham_u, trial_u, 0.0)
check("walker phia after 3 steps", numpy.abs(walkers_u.phia - walkers_r.phia).max())
check("walker phib after 3 steps", numpy.abs(walkers_u.phib - walkers_r.phib).max())
check("walker weights after 3 steps", numpy.abs(walkers_u.weight - walkers_r.weight).max())

# local energy through the plum dispatch (full-cluster e2)
system = Generic(nelec)
trial_r.calc_greens_function(walkers_r)
trial_u.calc_greens_function(walkers_u)
e_r = numpy.asarray(local_energy(system, ham_r, walkers_r, trial_r))
e_u = numpy.asarray(local_energy(system, ham_u, walkers_u, trial_u))
check("local energy ETotal", numpy.abs(e_u[:, 0] - e_r[:, 0]).max())
check("local energy E1Body", numpy.abs(e_u[:, 1] - e_r[:, 1]).max())
check("local energy E2Body", numpy.abs(e_u[:, 2] - e_r[:, 2]).max())

# fragment (EFragCorr-style) estimator: 2-body kernels on normal-ordered dG
G0a_r, G0b_r = trial_r.Ghalf[0], trial_r.Ghalf[1]
dGa_r = walkers_r.Ghalfa - G0a_r[None]
dGb_r = walkers_r.Ghalfb - G0b_r[None]
ef_r = frag_2body_thc(ham_r, trial_r._thc_Xocca, trial_r._thc_Xoccb, dGa_r, dGb_r, NFRAG)
G0a_u, G0b_u = trial_u.Ghalf[0], trial_u.Ghalf[1]
dGa_u = walkers_u.Ghalfa - G0a_u[None]
dGb_u = walkers_u.Ghalfb - G0b_u[None]
ef_u = frag_2body_thc_uhf(ham_u, trial_u._thc_Xocca, trial_u._thc_Xoccb,
                          dGa_u, dGb_u, NFRAG, NFRAG)
check("fragment 2-body estimator (dG)", numpy.abs(ef_u - ef_r).max())

# ======================================================================
print("\n== (b) ASYMMETRIC gate: n_a_orb != n_b_orb, na != nb runs finite ==")
rng = numpy.random.default_rng(999)
na_orb, nb_orb, na, nb_e = 8, 6, 3, 2
nmu2 = 20
Xa, _, ha, psia = make_problem(rng, na_orb, nmu2, na)
Xb = rng.standard_normal((nb_orb, nmu2)) * 0.3
hb = rng.standard_normal((nb_orb, nb_orb)) * 0.1
hb = 0.5 * (hb + hb.T) + numpy.diag(numpy.linspace(-1.0, 1.0, nb_orb))
A2 = rng.standard_normal((nmu2, nmu2)) * 0.2
M2 = A2 @ A2.T + 0.1 * numpy.eye(nmu2)
psib = numpy.zeros((nb_orb, nb_e)); psib[:nb_e, :nb_e] = numpy.eye(nb_e)

ham_x = GenericRealTHCUhf(ha, hb, Xa, Xb, M2, ecore=0.0, verbose=True)
trial_x = SingleDetUhfBasis(psia.copy(), psib.copy(), (na, nb_e))
trial_x.build()
trial_x.half_rotate(ham_x)
walkers_x = UHFWalkersSpinBasis(psia.copy(), psib.copy(), na, nb_e,
                                na_orb, nb_orb, NW, mpih)
walkers_x.build(trial_x)

mf_x = construct_mean_field_shift_thc_uhf(ham_x, trial_x)
finite("mean-field shift", mf_x)
prop_x = PhaselessGeneric(DT)
prop_x.build(ham_x, trial_x, walkers_x)
finite("one-body propagator alpha", prop_x.expH1[0])
finite("one-body propagator beta", prop_x.expH1[1])
numpy.random.seed(4242)
for step in range(5):
    prop_x.propagate_walkers(walkers_x, ham_x, trial_x, 0.0)
finite("walker phia after 5 steps", walkers_x.phia)
finite("walker phib after 5 steps", walkers_x.phib)
finite("walker weights after 5 steps", walkers_x.weight)
assert walkers_x.phia.shape == (NW, na_orb, na)
assert walkers_x.phib.shape == (NW, nb_orb, nb_e)

trial_x.calc_greens_function(walkers_x)
system_x = Generic((na, nb_e))
e_x = numpy.asarray(local_energy(system_x, ham_x, walkers_x, trial_x))
finite("local energy (asymmetric)", e_x)
ef_x = frag_2body_thc_uhf(ham_x, trial_x._thc_Xocca, trial_x._thc_Xoccb,
                          walkers_x.Ghalfa - trial_x.Ghalf[0][None],
                          walkers_x.Ghalfb - trial_x.Ghalf[1][None],
                          min(NFRAG, na), min(NFRAG, nb_e))
finite("fragment estimator (asymmetric)", ef_x)
# per-(spin, fragment) wiring: alpha-only and beta-only pinned estimators
ef_a = frag_2body_thc_uhf(ham_x, trial_x._thc_Xocca, trial_x._thc_Xoccb,
                          walkers_x.Ghalfa, walkers_x.Ghalfb, min(NFRAG, na), None)
ef_b = frag_2body_thc_uhf(ham_x, trial_x._thc_Xocca, trial_x._thc_Xoccb,
                          walkers_x.Ghalfa, walkers_x.Ghalfb, None, min(NFRAG, nb_e))
finite("alpha-only fragment estimator", ef_a)
finite("beta-only fragment estimator", ef_b)

print()
if failures:
    print(f"GATE FAILED: {len(failures)} check(s): {failures}")
    sys.exit(1)
print("GATE PASSED: all checks green.")
