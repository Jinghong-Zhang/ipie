# Analytical-Gradient AFQMC — Project Notes

Detailed catch-up document for anyone (human or AI) resuming this project.
Branch `analytical-gradient`, based on `develop` @ `b3f280e`.  All statements
below were verified by tests or numerical experiments; commit hashes and exact
numbers are given so claims can be re-checked.

This document is self-contained: everything needed to resume the project is in
the repo (this file, the addon README, the in-repo tests, and the example).
The off-repo validation studies (run scripts, raw data, and a full LaTeX
derivation) were kept outside the repository per the no-run-artifacts
convention; their protocols, parameters, and results are recorded in §6, §8,
and Appendix A so they can be regenerated from scratch on any machine.  §§2–4
summarize the derivation completely enough to reconstruct it.

## 1. Purpose

Hand-derive the analytical gradient dE/dλ of the **phaseless AFQMC** energy
for H(λ) = H₀ + λ·Ô (Ô a one-body observable), implement it as **forward-mode
tangent propagation** (no automatic differentiation), and use it to verify the
PyTorch reverse-mode AD addon `ipie/addons/adafqmc` (same repo).  Formalism
reference: Motta & Zhang, arXiv:1711.02242.  Everything is spin-restricted
(RHF), matching adafqmc.

Headline results:
- adafqmc's AD machinery is **correct to machine precision** (~1e-15 per-block
  gradient parity under identical auxiliary fields).
- The forward-mode implementation is now **1.5–3× faster** than torch
  forward+reverse AD, at ~2.2× the cost of a value-only step (theoretical
  floor ≈ 2×).
- Relaxed-trial dipoles agree with FCI within 1σ on H2, HeH⁺, NH3, H2O
  (minimal bases); deviations elsewhere were traced to *method* properties
  (phaseless constraint slope), not the gradient.

## 2. Algorithm conventions (must match adafqmc exactly for parity)

One propagation step (`adafqmc/propagation/propagator.py` ≡ our
`propagation/propagator.py`), walker φ ∈ C^(nao×nocc), weight w real:

- Green's function Θ = φS⁻¹, S = ψT†φ; force bias v_g = 2Tr(rchol_g Θ),
  x̄ = −√Δτ(i·v − mf_shift), mf_shift_g = 2i·Tr(L_g G_T) (purely imaginary);
  componentwise cap: |x̄|>1 → x̄/|x̄| (1e-13 guard).
- One-body half steps exp(−Δτ/2·H1mf), H1mf = h1e − v0 + Σ_g Im(mf)_g L_g,
  v0 = ½Σ_g L_g L_g; two-body exp(V_HS) with V_HS = i√Δτ Σ_g (ξ−x̄)_g L_g,
  applied by **Taylor series of fixed order 6** (differentiate the truncated
  series, not the exact exponential).
- Weight factor (hybrid, **no inter-step energy mixing**, unlike core ipie):
  log I = logratio + logfb + logmf + logct with
  logratio = 2[log det S′ − log det S] (RHF det **squared**),
  logfb = ξ·x̄ − ½x̄·x̄ (**unconjugated** dots), logmf = −√Δτ(ξ−x̄)·mf_shift,
  logct = Δτ(E_est − h0shift), h0shift = enuc − ½|Im mf|².
  Phase θ = Im(logratio + logmf) — **expfb and constant_term are excluded
  from the phase**.  w ← w·|I|·max(0, cos θ); then weight cap
  w ← min(w, 0.1·Σw_precap).
- QR reortho every `stabilize_freq` steps (R discarded); stochastic
  reconfiguration (SR) every `pop_control_freq` steps: renormalize to Σw=N,
  systematic comb with one uniform ζ, `searchsorted side='left'`, idx==N→0;
  **post-SR weights are identically equal ⇒ their λ-tangent is exactly zero**;
  state tangents are gathered by the (integer, undifferentiated) index map.
- Energy shift E_est: initialized to the trial energy — **NOT detached in
  adafqmc** (`propagator.py:165`), so its λ-tangent flows into constant_term
  during the first sub-block; each sub-block update *is* detached
  (`etot.detach()`), i.e. tangent zeroed.  Parity-critical.

## 3. Parameterizing λ — two equivalent trial-response routes

All λ-dependence enters through arrays carried with tangents:
`HamTangent(nelec0, nao, h1e, chol, enuc, dh1e, dchol)` plus a trial tangent
`SDTrial(psi, nelec0, dpsi=...)`.

1. **Rotated-basis (adafqmc convention)**: Hamiltonian rotated to relaxed
   orbitals U(λ); trial = identity columns (λ-independent);
   dh1e' = dU†hU + U†h dU + U†ÔU, dL' = dU†LU + U†L dU
   (`build_relaxed_trial_tangent`).
2. **Fixed-basis (preferred)**: integrals λ-independent except dh1e = Ô
   (dchol = 0 — the two-body interaction never changes for a one-body Ô);
   orbital response carried by dψ.  Verified pathwise-identical gradients to
   the rotated route on HeH⁺ (+1.074005(7183) both ways).  Cheaper (no chol
   rotation, packed-chol dchol term skipped) and much better conditioned for
   finite-δ correlated sampling (identical exp(V_HS) across ±δ runs).

Fixed trial = either route with zero response tangent.

**Getting dψ for real molecules**: use the occupied-projector finite
difference, dψ = dP·ψ₀ with P(λ) = C_occ(λ)C_occ(λ)† from tightly converged
SCF at h ± ε·Ô (Richardson over two ε), expressed in the λ=0 MO basis.  This
is gauge-clean (ψ†dψ = PdPP = 0) and **degeneracy-safe** — NH3's 1e orbitals
are degenerate to ~3e-8 and break jvp-through-eigh (1/(εi−εj) noise), while
the projector is smooth.  Sanity checks used: dP Richardson consistency
≤ 5e-7; HF Hellmann–Feynman closure FD(E_HF) = Tr(2P₀Ô) to ≤ 2e-10.

## 4. Forward-mode tangent equations (as implemented)

Per walker carry (φ, dφ, w, dw); every λ-dependent quantity is an explicit
(value, tangent) pair — deliberately no Dual/operator-overloading class.

- exp(−Δτ/2 H1mf) and its tangent via `scipy.linalg.expm_frechet` (once per
  propagator; use the tuple return so value/tangent share one Padé).
- dΘ = dφS⁻¹ − ΘdS S⁻¹ with dS = dψ†φ + ψ†dφ; dv from drchol and dΘ;
  force-bias cap tangent d(x/|x|) = dx/|x| − x·Re(x̄dx)/|x|³ (not holomorphic —
  complex-step differentiation is invalid).
- Coupled Taylor recursion, **tangent update consumes the pre-update term**:
  dT_n = (dV·T_{n−1} + V·dT_{n−1})/n before T_n = V·T_{n−1}/n.
- d log det via Jacobi (batched solve + trace); weight-factor tangent in log
  space: dfactor = [cosθ>0]·(factor·Re(dlogI) − |I|·sinθ·dθ),
  dθ = Im(dlogratio + dlogmf).
- Weight cap tangent: capped walkers get dw = 0.1·Σdw_precap (tie condition
  ¬(w < bound), matching torch.where).
- Estimator: dĒ = Σ[dw(E_L−Ē) + w·dE_L]/Σw (weights real, quotient rule
  before Re); block combination dÊ = Σ[dE_i W_i + (E_i−Ê)dW_i]/ΣW.
- **Reorthogonalization gauge (critical)**: φ←Q, dφ←(1−QQ†)dφR⁻¹.  The
  estimator is exactly invariant under in-span tangent shifts dφ→dφ−φM (the
  tangent flow maps gauge to gauge), so both the fixed-R choice and the
  projection are exact.  The projection is **mandatory for >2-electron
  systems**: the raw tangent grows ~e^(λ_L τ) with measured λ_L ≈ 13.3/a.u.
  on NH3 along an almost purely in-span direction; in float64 the roundoff
  residue 1e-16·|dφ| destroys the gradient (H2O per-block gradients of
  1e2–1e6, overflow at τ=6) even though exact arithmetic cancels it.  With
  projection |dφ| saturates ~1 indefinitely.  Reverse-mode AD through the QR
  Q-factor re-gauges automatically — why adafqmc never showed this.  Note
  tr(S⁻¹dS) alone is NOT gauge-invariant (shifts by −tr(M)); only step ratio
  differences are.
- SR: dφ gathered, dw ← 0 exactly (see §2); indices support record/replay
  (`sr_record` / `sr_replay`) for frozen-map verification.

## 5. Estimator modes and recommendations

`qmc/fwdgrad_afqmc.py` (`FwdGradAFQMC`), serial, RNG always injected via
`utils/fields.py` (`RandomFields` / `ScriptedFields`, strict consumption
contract: one (nw,nchol) normal per step, one uniform per SR event).

1. **`run_along_path` (recommended)**: tangents ride the whole trajectory
   (dφ never reset), eshift feedback differentiated through
   (`detach_eshift=False`), gradient sampled at every measurement like the
   energy.  Set `num_steps_per_block == pop_control_freq == window` so each
   sample sits right before an SR reset with the full **weight-response
   window τ_w = window·Δτ** (the τ_w bias decays ~e^(−Δ·τ_w) like a
   back-propagation time; τ_w ≳ 3/gap converged everything tested).  Discard
   ~3 burn-in measurements; use reblocked errors (samples correlated).
2. `run` with `num_steps_per_block == ad_block_size`, in-block SR off:
   final-time block estimator (tangent reset per block) — equivalent
   accuracy, wastes the per-block dφ transient.
3. `run` with sub-blocks (adafqmc aggregation) — kept ONLY for exact parity;
   **biased**: averaging sub-block gradients over early imaginary times
   plateaus ~4 mHa off FCI on H2 even at τ_block = 6.

## 6. Verification record (exact numbers)

Unit suite: `conda run -n ipierun pytest ipie/addons/analytical_gradient -m unit`
→ 32 passed.  Layers:

1. **Building-block FD**: every tangent (expm Fréchet, Θ, cap, Taylor, local
   energy, trial energy, weighted/block averages, single full step with caps
   off/on) vs Richardson CRN central differences.
2. **Exact-fields parity vs adafqmc** (`qmc/tests/test_torch_parity.py`):
   torch.randn/rand monkeypatched to replay one recorded stream;
   `ad_block_gradient` driven directly.  Gradient diffs 6.7e-16 (fixed) /
   3.1e-15 (relaxed) on a random system; ≤2.2e-15 on HeH⁺ with the true jvp
   response tangent and nonzero nuclear-dipole obs_const (protocol in
   Appendix A.2).  Also verified: adafqmc's
   `TrialwithTangent.backward` returning an (nao,nao) array for shape-(1,)
   coupling is CORRECT (torch sum-reduces broadcast gradients = chain rule).
3. **Frozen-map FD identity** (the defining pathwise check,
   `qmc/tests/test_fd_along_path.py`, 3 parameterization modes): same fields
   + SR index arrays recorded from the base run and replayed (SR is never
   differentiated) + common λ-independent initial walkers ⇒ FD(E_i) = dE_i
   and FD(W_i) = dW_i at EVERY measurement for arbitrarily many blocks.
   Long-path demo (Appendix A.3): 500 steps, 20 SR events, 100 QRs,
   self-consistent eshift → max |FD−dE_i| = 1.9e-10, no growth.  For the *block* estimator the detached sub-block eshift sequence
   must be frozen at λ=0 values in FD runs (`eshift_override`) — without it
   FD "disagrees" at ~7e-5 (that's the detached path, not a bug).
4. **Correlated sampling ≡ analytical** at δ=1e-5 (paired runs, frozen map):
   |g_CS − g_AN| = 0.000 mHa with identical SEs on H2 R=1.4/2.4, HeH⁺
   fixed/relaxed, NH3 (per-measurement 1e-9..1e-10 where no cosθ flips).
   Finite-δ CS deviates only via (a) O(δ²) curvature (measured scaling) and
   (b) one-branch cosθ=0 crossings (walker death → lineage decorrelation;
   e.g. HeH⁺ relaxed at δ=0.01: 18.2 mHa, 6 flips; δ≤1e-3: 0 flips).  The
   analytical gradient is the δ→0 limit computed without choosing δ.
5. **Physics vs FCI** (same Cholesky-factorized integrals, chol_cut 1e-8;
   pyscf FCI; path-continuous, τ_w=3, nw=300, dt=0.005):

   | system | Ô | trial ⟨Ô⟩ | FCI | AFQMC fixed | AFQMC relaxed |
   |---|---|---|---|---|---|
   | H2/STO-6G R=1.4 | ĥ₁ | −2.514147 | −2.494378 | −2.494904(720) | (response ≡ 0 by g/u symmetry) |
   | H2/STO-6G R=2.4 | ĥ₁ | −1.989327 | −1.940619 | −1.956569(2141) | (same) |
   | HeH⁺/STO-3G | μ_z | +1.113615 | +1.073441 | +1.058632(1027) | +1.074005(7183) |
   | NH3/STO-3G | μ_z | −0.703273 | −0.686650 | −0.683427(630) | **−0.686884(738)** |
   | H2O/STO-3G | μ_z | +0.678943 | +0.636028 | +0.625079(2146) | **+0.635538(2338)** |

   H2 R=2.4's −16 mHa residual equals the measured λ-slope of the phaseless
   constraint bias (CRN FD of E_ph(λ) at λ=±0.01: −1.956(34)) — a method
   property; the estimator exactly differentiates the phaseless surface.
6. **wtsgrad**: the weight-gradient term adafqmc collects but omits from its
   final average (adafqmc.py:466-487) is negligible: |Δ|/σ ≤ 0.03 worst case,
   ≲0.005 in production runs.

## 7. File map

```
ipie/addons/analytical_gradient/
├── hamiltonians/hamiltonian.py   HamTangent (h1e/chol + tangents, h1e_mod pair,
│                                 packed symmetric chol + fallback, has_dchol),
│                                 build_fixed_trial_tangent, build_relaxed_trial_tangent,
│                                 shifted_copy (value-only λ=ε copy for CRN FD)
├── trial_wavefunction/sdtrial.py SDTrial(psi, nelec0, dpsi): half-rotated (rh1, rchol)
│                                 + tangents (Hamiltonian AND trial terms), overlaps,
│                                 Θ, force bias, trial energy — all with tangents
├── walkers/rhf_walkers.py        GradWalkers(phi,dphi,weight,dweight), detached_copy,
│                                 reorthogonalize (gauge projection), SR (record/replay)
├── propagation/propagator.py     compute_exph1_with_tangent, force-bias cap,
│                                 construct_vhs_(packed_)with_tangent, Taylor,
│                                 GradPropagator.propagate_walkers (xbar_override for
│                                 frozen-force-bias CS gauges; debug diagnostics with
│                                 cap/cos masks, |dphi|, SR indices), propagate_block
│                                 (eshift_override, detach_eshift, sr queues)
├── estimators/estimator.py       local_energy_with_tangent, weighted/block averages
├── qmc/fwdgrad_afqmc.py          GradParams, FwdGradAFQMC: build/equilibrate_walkers/
│                                 ad_block/run/run_along_path
├── utils/fields.py               RandomFields / ScriptedFields (RNG injection)
├── utils/linalg.py               rmatmul (real/imag-split gemm), left_apply
│                                 (flattened one-body application), flatten_ghalf
├── utils/testing.py              random RHF test systems (gauge-fixed eigvecs)
└── */tests/                      32 unit tests (see §6)
examples/23-analytical_gradient/run_h2_gradient.py
```

## 8. Performance (repo c9cfe6b, f1bca39; benchmark protocol in Appendix A.7)

Single-thread, nw=300, ms per value+tangent step:

| system | einsum baseline | optimized | torch fwd-only | torch fwd+grad |
|---|---|---|---|---|
| NH3 (nao=8, nchol=36) | 9.3 | **2.8** | 2.2 | 8.5 |
| synthetic (nao=24, nchol=96) | 145 | **22.4** | 10.1 | 34.6 |

Optimizations (core-ipie kernel conventions): all hot einsums → flattened 2D
gemms with real/imag splitting (real integral tensors stay in dgemm); zero
dchol/drchol/dpsi terms short-circuited; Taylor stacks (T,dT) column-wise;
packed upper-triangle Cholesky VHS + ipie's numba `unpack_VHS_batch`
(automatic unpacked fallback for non-symmetric chol).  Now ~2.2× a value-only
step vs the ~2× intrinsic floor; profile: Taylor zgemms 43%, one-body+overlap
gemms 14%, VHS/force-bias gemms 12%, per-walker LAPACK ~9% (three redundant
LU factorizations of the same S per step — known ~10% headroom), python/copies
the rest.  Next real lever: cupy/GPU port (everything is batched gemm).

## 9. Gotchas for future work

- Never compare raw dφ across implementations/after QR — only gauge-invariant
  quantities; per-walker dφ is defined up to in-span shifts.
- FD verification stencils must not straddle kinks: assert identical
  force-bias-cap / weight-cap / sign(cosθ) masks and SR maps across ±ε runs
  (debug diagnostics exist for exactly this).  Continuous kinks do NOT bias
  the expected gradient (pathwise/IPA argument); only SR selection jumps do.
- cosθ, sinθ computed branch-cut-free from Im(log) sums, never np.angle
  differences; log-det via slogdet keeping the complex phase.
- The estimator is invariant under walker-uniform weight factors, so eshift
  choices cancel in dE_i but NOT in dW_i.
- obs_const (e.g. nuclear dipole) never enters propagation — added to the
  returned gradient only (matching adafqmc).
- Matrix-valued couplings (1rdm obs_type) would need nao² forward tangents —
  use reverse mode (adafqmc) there; this addon is for scalar couplings.
- Frozen-core is NOT implemented (needed beyond minimal bases).
- MPI is NOT implemented (serial by design; adafqmc's MPI is an orthogonal
  wrapper).

## 10. Commit map (develop..HEAD)

- c6c8366 scaffold: HamTangent + builders + shifted_copy + RNG injection
- d778427 trial + estimator tangents (FD unit tests)
- 98d8c61 walkers: reortho gauge + SR semantics
- 94b2e95 propagator value+tangent step (single-step FD tests)
- 841d967 driver + block-level CRN-FD validation (eshift_override)
- 3a19049 exact-fields torch parity test
- bd4e676 README + H2 example
- 88dd6ac eshift_override threading
- a09610e path-continuous mode + frozen-SR FD identity
- 748bd4f README: path-continuous recommended
- 9456938 fixed-basis trial-tangent (dpsi) parameterization
- 498db80 xbar_override (frozen-force-bias CS gauge; lockstep CS conditioning)
- 0be8d1f in-span gauge projection at reortho (many-electron conditioning fix)
- c9cfe6b BLAS kernel rewrite (6.3× on nao=24)
- f1bca39 packed-Cholesky VHS gemms

## Appendix A. Protocols of the off-repo validation studies

These studies were run as standalone scripts outside the repo; their results
are quoted in §6/§8.  Everything below can be regenerated from the public
addon API.  Common settings unless stated: nw=300, dt=0.005,
stabilize_freq=5, 2000 equilibration steps, seeds fixed, errors = weighted SE
with pair-reblocking to the plateau.

**A.1 FCI references.**  pyscf `gto.M` → RHF (conv_tol 1e-13) →
`ipie.utils.from_pyscf.generate_integrals(mol, hcore, mo_coeff, chol_cut=1e-8)`
(MO basis); reconstruct eri = einsum("apq,ars->pqrs", chol, chol) so the
reference shares the factorization; `fci.direct_spin1.kernel`;
dE/dλ = Tr(γ_FCI·Ô) cross-checked against Richardson FD of E_FCI(λ) (≤7e-13);
context values Tr(γ_HF·Ô) and the mixed ⟨HF|Ô|FCI⟩/⟨HF|FCI⟩ via `trans_rdm1`.
Geometries used: H2 R=1.4/2.4 bohr STO-6G; HeH⁺ R=1.46 bohr STO-3G (He at
origin, dipole origin at coordinate origin); NH3 r(NH)=1.0124 Å HNH=106.67°
(C3 axis = z); H2O r(OH)=0.9572 Å HOH=104.52° (C2 axis = z), both STO-3G.

**A.2 Torch parity on a real molecule.**  HeH⁺ integrals as in A.1; trial
response (U0, dU) from `torch.func.jvp` through
`adafqmc.utils.miscellaneous.get_hf_wgradient`; one AD block of 20 steps
(2 sub-blocks of 10), QR/SR every 5, nw=24, perturbed-identity initial
walkers with random weights in [0.5, 2]; torch.randn/torch.rand monkeypatched
to replay a shared numpy stream (assert both queues exhausted); compare
`ADAFQMC.ad_block_gradient` outputs (observable, wtsgrad, e_estimate, blkwts)
against `FwdGradAFQMC.ad_block` with `build_relaxed_trial_tangent(U0, dU)`.
Same harness as the in-repo `test_torch_parity.py`, just with pyscf inputs.

**A.3 Long-path frozen-map FD identity.**  H2 R=1.4, nw=50, 10 measurements
× 50 steps (500 total), SR every 25 (20 events), `run_along_path` from
trial-initialized walkers on `ScriptedFields`; base run records `sr_record`;
±ε value runs on `shifted_copy(ham, ±ε)` with `sr_replay`, ε = 1e-5 and 5e-6
+ Richardson; assert identical cap/cos masks across the stencil.  Result:
max_i |FD − dE_i| = 1.9e-10, max rel |FD − dW_i| = 6.9e-8.

**A.4 Estimator-bias scans (H2 R=1.4, O=ĥ₁, fixed trial).**
(a) Sub-block-averaged `run` (num_steps_per_block=50) at ad_block_size ∈
{150,300,600,1200} with in-block SR off (pop_control_freq=1e9) and on (5),
~45k sampling steps each: deviation from FCI plateaus at ≈ −4 mHa.
(b) Final-time estimator (num_steps_per_block == ad_block_size, SR off):
dev = −6.5 → −2.4 → +0.8 → −0.7(6) mHa at τ_block = 0.75/1.5/3/6 a.u.
(c) Phaseless-surface slope by CRN FD: two full value runs at λ = ±0.01
(same seed, standard SR5 sub-block runs), paired per-block differences:
dE_ph/dλ = −2.495290(1949) [R=1.4], −1.956(34) [R=2.4, larger λ noise],
HeH⁺ relaxed path +1.067403(2343) — each matching its gradient estimator.

**A.5 Correlated-sampling (CS) vs analytical.**  Protocol: equilibrate once
at λ=0; base `run_along_path` (window = num_steps_per_block =
pop_control_freq = 600, i.e. one measurement per SR period) records the SR
map; two full value runs at λ=±δ with same fields, same initial walkers,
replayed SR map; g_CS = ⟨(E₊ᵢ−E₋ᵢ)/2δ⟩ paired per measurement.  At δ=1e-5:
g_CS ≡ g_AN (0.000 mHa, identical SEs) on H2 R=1.4/2.4, HeH⁺ fixed/relaxed.
δ-scan on HeH⁺ relaxed: 18.2 mHa dev at δ=0.01 (6 one-branch cosθ flips),
3.0 at 0.005, 1.0 at 0.002, O(δ²) below; the flip mechanism is
parameterization-independent (control experiment in the fixed-basis dψ
gauge).  Frozen-force-bias variant (replay base x̄ via `xbar_override`):
conditions CS at δ=0.01 (flips 6–24 → ≤1) but measures a slightly different
sampling gauge (drops the dx̄ response; agrees in expectation within ~0.7σ).

**A.6 Dipole production runs (§6 table).**  Path-continuous, τ_w = 3 a.u.
(window 600), ~80–100 measurements, 3 burn-in discarded; trial response via
the occupied-projector FD of A.1 geometries (ε = 1e-3 and 5e-4, Richardson);
NH3 window check at τ_w=6 gave −0.68854(83) (consistent).  Tangent-growth
probe (pre gauge-fix diagnosis): continuous `run_along_path`, debug=True,
read `max_abs_dphi` from diagnostics — NH3 grew 4.6e-3 → 2.6e+55 over 10 a.u.
(λ_L ≈ 13.3/a.u.) without the projection; saturates at ~1 with it.

**A.7 Speed benchmark.**  Single thread (OMP_NUM_THREADS=1,
torch.set_num_threads(1)); NH3 integrals (nao=8, nchol=36) and a synthetic
system (nao=24, nocc=12, nchol=96, random symmetric chol scaled 0.3/√nchol);
nw=300; time `GradPropagator.propagate_walkers` over 200/60 steps after one
warmup step; torch reference: `adafqmc.Propagator.propagate_walkers` under
no_grad (forward-only) and a full graph over the same steps + one
autograd.grad w.r.t. a coupling in h1e (forward+reverse, amortized).
