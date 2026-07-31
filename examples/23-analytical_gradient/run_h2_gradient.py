"""Analytical (forward-mode) gradient of the phaseless AFQMC energy for H2.

Computes dE/dlambda for H(lambda) = H + lambda*O with O = h1e on H2/STO-6G
using the numpy forward-mode addon (no automatic differentiation), and
compares against the FCI Hellmann-Feynman value Tr(gamma_FCI O) computed from
the same Cholesky-factorized integrals.  The HF orbital response to O = h1e
vanishes by g/u symmetry at the symmetric geometry, so the fixed-trial
gradient equals the relaxed-trial gradient for this observable.

The gradient estimator uses a single sub-block per AD block (the final-time
mixed-estimator derivative) without in-block reconfiguration; the sub-block-
averaged variant converges to the same answer more slowly in ad_block_size.
"""

import numpy as np
from pyscf import fci, gto, scf

from ipie.addons.analytical_gradient.hamiltonians.hamiltonian import (
    build_fixed_trial_tangent,
)
from ipie.addons.analytical_gradient.qmc.fwdgrad_afqmc import FwdGradAFQMC
from ipie.addons.analytical_gradient.trial_wavefunction.sdtrial import SDTrial
from ipie.utils.from_pyscf import generate_integrals

mol = gto.M(atom="H 0 0 0; H 0 0 1.4", basis="sto-6g", unit="bohr", verbose=0)
mf = scf.RHF(mol)
mf.kernel()
h1e, chol, enuc = generate_integrals(mol, mf.get_hcore(), mf.mo_coeff, chol_cut=1e-8)
nao = h1e.shape[0]
nocc = mol.nelec[0]

# FCI reference from the same factorized integrals.
eri = np.einsum("apq,ars->pqrs", chol, chol)
e_fci, ci0 = fci.direct_spin1.kernel(h1e, eri, nao, mol.nelec, ecore=enuc)
gamma = fci.direct_spin1.make_rdm1(ci0, nao, mol.nelec)
ref = np.sum(gamma * h1e)
print(f"# E_FCI = {e_fci:.8f}, dE/dlambda (FCI Hellmann-Feynman) = {ref:+.8f}")

ham = build_fixed_trial_tangent(nocc, nao, h1e, chol, enuc, obs_mat=h1e.copy())
trial = SDTrial(np.eye(nao), nocc)  # RHF determinant in the MO basis
trial.half_rot(ham)

driver = FwdGradAFQMC.build(
    num_walkers=200,
    num_steps_per_block=600,  # single sub-block: final-time gradient estimator
    ad_block_size=600,  # tau_block = 3 a.u.
    num_ad_blocks=20,
    timestep=0.005,
    stabilize_freq=5,
    pop_control_freq=10**9,  # no in-block reconfiguration
    pop_control_freq_eq=5,
    seed=7,
    num_eqlb_steps=2000,
)
energies, gradients, weights, wtsgrads = driver.run(ham, trial, obs_const=0.0, verbose=True)

wmean = np.sum(weights * gradients) / np.sum(weights)
neff = np.sum(weights) ** 2 / np.sum(weights**2)
err = np.sqrt(np.sum(weights * (gradients - wmean) ** 2) / np.sum(weights) / neff)
emean = np.sum(weights * energies) / np.sum(weights)
print(f"# E_AFQMC = {emean:.8f}")
print(f"# dE/dlambda (AFQMC) = {wmean:+.8f} +/- {err:.8f}")
print(f"# deviation from FCI = {wmean - ref:+.8f}")
