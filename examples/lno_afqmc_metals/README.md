# No-TRS metal LNO-AFQMC on the complex THC export

This is the runner that consumes qcpbc's **complex** THC export
(`cthc_iao_<F>_<T>.h5`, written by `postscf_lno.C`) and runs LNO-AFQMC on it.

The `ipie` side of the complex path (`GenericComplexTHC`, `ipie/lno_thc_cx.py`,
`ipie/lno_thc.py`) has been on this branch for a while; **this driver was the
missing piece** — it lived only in a FASRC scratch directory, so nothing in the
repo actually read a `cthc_iao` file. That is what this directory fixes.

## Prerequisites

* This branch of ipie installed (`pip install -e .` from the repo root) — the
  runner imports `ipie.hamiltonians.thc.GenericComplexTHC`,
  `ipie.lno_thc_cx`, `ipie.lno_thc`.
* A `cthc_iao_<iF>_<ithresh>.h5` produced by qcpbc's metals driver
  (`postscf_method = lno`, complex/no-TRS path, `lno_thc_save = true`).

## Run

```bash
# 1) ALWAYS do this first on a new export: non-circular consistency check
python run_cthc_metal.py cthc_iao_0_0.h5 --validate

# 2) production
python run_cthc_metal.py cthc_iao_0_0.h5 <nwalkers> <nblocks> <dt>
#    e.g.  python run_cthc_metal.py cthc_iao_0_0.h5 1024 500 0.002
```

`SEED` is read from the environment (default 7).

## What `--validate` proves, and why it matters

At **full space** (every occupied orbital inside the cluster) the check is
*non-circular*: the bare core must satisfy

    hcore + vHF == diag(f_diag + xi)

which tests `vHF` itself rather than re-deriving it. For **truncated** clusters
the bare core legitimately misses the environment mean field — that is exactly
why `h_eff` is constructed rather than used raw. So `--validate` is meaningful
on a full-space export and is *expected* to show a difference on a truncated
one; do not read a truncated mismatch as a failure.

## Conventions you must not "fix"

The two-body convention is

    (pq|rs) = sum_{mu,nu} X[p,mu] conj(X[q,mu]) M[mu,nu] conj(X[r,nu]) X[s,nu]

and the exchange index pattern is **(pr|qs)**. Using `(pr|sq)` instead yields a
NON-Hermitian `K`; the runner checks `|K - K^H|` and will catch it. This class
of error is invisible for real orbitals and wrong for complex ones, which is
precisely the metals case.

The C++ writes only the BARE core (`T + V_pp`) on purpose; the Fock-consistent
one-body is built here:

    h_eff = diag(f_diag + xi*P_occ) - vHF,   vHF = J - K/2

## Gotchas carried over from the FASRC runs

* `dt <= 0.002` — larger timesteps bias the metals runs.
* Reblock the `EFragCorr` errors; do not quote raw block standard errors.
* Truncated fragments drift (non-stationary trial); that is expected, not a bug.
* ipie without `mpi4py` silently falls back to `FakeComm`, so `srun -n N` runs N
  redundant copies clobbering one estimates file. Build `mpi4py` against the
  system MPI and launch with `--mpi=pmi2`, or run single-rank.
