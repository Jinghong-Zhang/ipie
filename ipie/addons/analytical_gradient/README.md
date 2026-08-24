# Analytical gradient of the phaseless AFQMC energy

Hand-derived **forward-mode sensitivity** implementation of dE/dlambda for
H(lambda) = H0 + lambda*O (O a one-body observable) — no automatic
differentiation.  Every lambda-dependent quantity is carried as an explicit
(value, tangent) pair through the phaseless propagation: Frechet derivative of
the one-body exponential, force-bias (and cap) tangents, coupled Taylor
recursion for exp(V_HS), log-space weight-factor tangents with the phaseless
cosine projection, fixed-R gauge through QR reorthogonalization, and exact
stochastic-reconfiguration semantics.

The algorithm mirrors `ipie/addons/adafqmc` (PyTorch reverse-mode AD) step for
step: driven by identical auxiliary fields, the two implementations agree to
machine precision (see `qmc/tests/test_torch_parity.py`), which provides an
independent analytical verification of the AD addon.  Runtime dependencies are
numpy and scipy only; torch is imported solely inside the parity test.

## Workflows

- **Fixed trial**: `build_fixed_trial_tangent(...)` (dh1e = O, dchol = 0).
- **Relaxed trial** (orbital response): `build_relaxed_trial_tangent(...,
  rot_mat, drot_mat)` with the Hamiltonian rotated to the relaxed-orbital
  basis (trial = identity columns), exactly as adafqmc does; `(rot_mat,
  drot_mat)` can be produced by `torch.func.jvp` through
  `ipie.addons.adafqmc.utils.hf.hartree_fock` or any coupled-perturbed HF.

## Usage

See `examples/23-analytical_gradient/run_h2_gradient.py`.  Three estimator
modes:

- **`run_along_path` (path-continuous, recommended)**: forward mode needs no
  AD blocks (there is no computational graph), so tangents ride the whole
  trajectory, the energy-shift feedback is differentiated through, and the
  gradient is sampled at every measurement like the energy.  Its defining
  verification identity: with the same auxiliary fields and the SR walker map
  frozen (record/replay of the resampling indices — SR is not
  differentiated), the central finite difference of every measured E_i and
  W_i equals the analytical dE_i and dW_i for arbitrarily many blocks
  (`qmc/tests/test_fd_along_path.py`; ~2e-10 over a 500-step H2 path).
- `run` with `num_steps_per_block == ad_block_size`: block-detached
  final-time mixed-estimator derivative (converges exponentially in
  `ad_block_size * dt`).
- `run` with `num_steps_per_block < ad_block_size`: adafqmc-style sub-block
  averaging (kept for exact parity with the AD addon; converges more slowly,
  as early-imaginary-time sub-blocks contaminate the average).

The a.e.-derivative semantics at non-smooth operations (cosine projection,
force-bias/weight caps, reconfiguration, detached energy shifts) are identical
to what reverse-mode AD computes; `PROJECT_NOTES.md` in this directory
documents each tangent rule, the verification record, and the full project
history.

## Tests

```
conda run -n ipierun pytest ipie/addons/analytical_gradient -m unit
```
