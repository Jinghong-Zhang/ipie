#!/usr/bin/env python
"""Map the low-k vs dense-k (large-k) exx algorithm regimes, old vs new kernels.

For each (nk, nbsf, nocc, nw) cell, times four variants of the KptISDF exx
kernel on GPU:
  old_lowk / old_largek : frozen 7cb863c kernels with the branch forced
  new_lowk / new_largek : optimized kernels with algo= forced
checks parity across all computed variants, reports which algorithm the new
FLOP model would select and which is actually fastest, and emits one
``BENCH_RESULT {json}`` line per cell.

Variants whose estimated cost exceeds --max-cell-seconds are skipped (the
frozen low-k kernel re-copies the full Green's function per ISDF chunk pair,
which becomes intractable at large nk*nbsf).
"""

import argparse
import json
import os
import time
from math import ceil

import numpy as np

if os.environ.get("IPIE_USE_GPU") != "1":
    raise RuntimeError("Set IPIE_USE_GPU=1 before running this benchmark.")

from ipie.config import config

config.update_option("use_gpu", True)

import cupy

from ipie.utils.backend import arraylib as xp
from ipie.utils.kpt_conv import find_Qplus, find_self_inverse_set
from ipie.hamiltonians.kpt_hamiltonian import construct_kpq
import ipie.estimators.local_energy_kpt_sd as le

import legacy_energy_kernels as legacy

KMESH = {8: (2, 2, 2), 16: (4, 2, 2), 32: (4, 4, 2), 64: (4, 4, 4), 125: (5, 5, 5)}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--nisdf-factor", type=int, default=18)
parser.add_argument("--repeats", type=int, default=2)
parser.add_argument("--max-cell-seconds", type=float, default=2400.0)
parser.add_argument("--seed", type=int, default=114514)
parser.add_argument(
    "--cells",
    default=None,
    help="comma-separated nk:nbsf:nocc:nw cells; default is the built-in grid",
)
args = parser.parse_args()

EFF_FLOPS = 40e12  # effective fp64 throughput (real flops)
EFF_BW = 1.5e12  # effective bandwidth (bytes/s)


def default_cells():
    cells = []
    grid = {
        8: (20, 50, 100, 200, 500, 1000),
        16: (20, 50, 100, 200, 500, 1000),
        32: (20, 50, 100, 200, 500),
        64: (20, 50, 100, 200),
        125: (20, 50, 100),
    }
    # main 2-d grid at nw=8, nocc ~ 9% of nbsf
    for nk, nbsfs in grid.items():
        for nbsf in nbsfs:
            cells.append((nk, nbsf, max(2, round(0.09 * nbsf)), 8))
    # walker-count axis (crossover moves with nk*nw)
    for nk, nbsf in ((8, 200), (16, 500), (32, 100)):
        for nw in (4, 16, 64):
            cells.append((nk, nbsf, max(2, round(0.09 * nbsf)), nw))
    # occupancy axis
    for nocc in (4, 36):
        cells.append((16, 200, nocc, 8))
    return cells


def lowk_chunk_old(nisdf, nk, nw):
    intermediate = nisdf * nisdf * nk**2 * nw * 3 * 16 / 1024**3
    num_chunks = max(1, ceil(intermediate / 8.0))
    per_dim = ceil(num_chunks**0.5)
    return ceil(nisdf / per_dim)


def estimate_seconds(variant, nk, nbsf, nocc, nw, nisdf, nq):
    g_bytes = nw * nk * nk * nocc * nbsf * 16
    if variant.endswith("largek"):
        cflops = nq * (
            nk * nocc * nbsf * nisdf**2 + 3 * nk**2 * nw * nocc * nbsf * nisdf
        )
        nbytes = nq * 4 * g_bytes
    else:
        if variant == "old_lowk":
            nchunk = lowk_chunk_old(nisdf, nk, nw)
            npair = ceil(nisdf / nchunk) ** 2
            copies = nq * npair * 2 * g_bytes  # full-G transpose per chunk pair
            stage1_factor = max(1, nisdf // nchunk)
        else:
            copies = nq * 2 * g_bytes
            stage1_factor = max(1, 4)  # stage-1 recompute bound when cache disabled
        cflops = nq * (
            2 * nk**2 * nw * nocc * nisdf**2
            + 2 * stage1_factor * nk**2 * nw * nocc * nbsf * nisdf
        )
        nbytes = copies + nq * 3 * nw * nk**2 * nisdf**2 * 16
    return 8 * cflops / EFF_FLOPS + nbytes / EFF_BW


def device_complex_rand(shape, scale):
    out = cupy.empty(shape, dtype=cupy.complex128)
    for i in range(shape[0]):
        re = cupy.random.standard_normal(shape[1:])
        im = cupy.random.standard_normal(shape[1:])
        out[i] = scale * (re + 1j * im)
        del re, im
    return out


def frac_grid(n):
    c = np.fft.fftfreq(n)
    c[c == -0.5] = 0.5
    return c


def run_variant(fn, est_s, repeats):
    if est_s < 5.0:
        fn()  # warmup only when cheap
    cupy.cuda.Stream.null.synchronize()
    t0 = time.time()
    out = fn()
    cupy.cuda.Stream.null.synchronize()
    best = time.time() - t0
    if best < 10.0:
        for _ in range(max(0, repeats - 1)):
            t0 = time.time()
            out = fn()
            cupy.cuda.Stream.null.synchronize()
            best = min(best, time.time() - t0)
    return out, best


if args.cells:
    cells = [tuple(int(x) for x in c.split(":")) for c in args.cells.split(",")]
else:
    cells = default_cells()
# run cheap cells first so partial output is useful
cells.sort(key=lambda c: sum(
    estimate_seconds(v, c[0], c[1], c[2], c[3], args.nisdf_factor * c[1], c[0])
    for v in ("old_lowk", "old_largek", "new_lowk", "new_largek")
))

dev_total = cupy.cuda.Device().mem_info[1]
pool = cupy.get_default_memory_pool()

for nk, nbsf, nocc, nw in cells:
    mesh = KMESH[nk]
    nisdf = args.nisdf_factor * nbsf
    kpoints = (
        np.array(np.meshgrid(*[frac_grid(n) for n in mesh], indexing="ij"))
        .reshape(3, -1)
        .T.copy()
    )
    kpq_mat = xp.asarray(construct_kpq(kpoints))
    Sset = find_self_inverse_set(kpoints)
    Qplus = find_Qplus(kpoints)
    nq = len(Sset) + len(Qplus)

    result = {
        "event": "exx_algo_sweep",
        "nk": nk,
        "nbsf": nbsf,
        "nocc": nocc,
        "nwalkers": nw,
        "nisdf": nisdf,
        "nq": nq,
    }

    need = (
        nq * nisdf**2 * 16
        + nk * nisdf * (nbsf + nocc) * 16
        + 8 * nw * nk * nk * nocc * nbsf * 16
        + 12 * 1024**3
    )
    if need > 0.9 * dev_total:
        result["skipped"] = f"memory estimate {need / 1024**3:.0f} GB"
        print("BENCH_RESULT " + json.dumps(result), flush=True)
        continue

    cupy.random.seed(args.seed + nk * 100003 + nbsf * 17 + nw)
    MPQ = device_complex_rand((nq, nisdf, nisdf), 1.0 / nisdf)
    cgto = device_complex_rand((nk, nisdf, nbsf), 1.0 / np.sqrt(nisdf))
    rcgto = device_complex_rand((nk, nisdf, nocc), 1.0 / np.sqrt(nisdf))
    Ga = device_complex_rand((nw, nk, nocc, nk, nbsf), 1.0 / np.sqrt(nk * nbsf))

    # what the live FLOP model in the new kernel would pick
    flops_largek = nk * nocc * nbsf * nisdf**2 + 3 * nk**2 * nw * nocc * nbsf * nisdf
    flops_lowk = 2 * nk**2 * nw * nocc * nisdf**2 + 2 * nk**2 * nw * nocc * nbsf * nisdf
    result["model_algo"] = "largek" if flops_largek <= flops_lowk else "lowk"
    result["model_flop_ratio"] = flops_largek / flops_lowk
    result["old_heuristic_algo"] = "lowk" if nbsf > 8 * nk else "largek"

    variants = {
        "old_lowk": lambda: legacy.kpt_isdf_exx_kernel_gpu_forced(
            MPQ, rcgto, cgto, Ga, kpq_mat, Sset, Qplus, "lowk"
        ),
        "old_largek": lambda: legacy.kpt_isdf_exx_kernel_gpu_forced(
            MPQ, rcgto, cgto, Ga, kpq_mat, Sset, Qplus, "largek"
        ),
        "new_lowk": lambda: le.kpt_isdf_exx_kernel_gpu(
            MPQ, rcgto, cgto, Ga, kpq_mat, Sset, Qplus, algo="lowk"
        ),
        "new_largek": lambda: le.kpt_isdf_exx_kernel_gpu(
            MPQ, rcgto, cgto, Ga, kpq_mat, Sset, Qplus, algo="largek"
        ),
    }

    outputs = {}
    for name, fn in variants.items():
        est = estimate_seconds(name, nk, nbsf, nocc, nw, nisdf, nq)
        result[f"est_{name}_s"] = round(est, 1)
        if est > args.max_cell_seconds:
            result[f"t_{name}_s"] = None
            result[f"skip_{name}"] = f"estimated {est:.0f}s"
            continue
        try:
            out, t = run_variant(fn, est, args.repeats)
            outputs[name] = out
            result[f"t_{name}_s"] = t
        except cupy.cuda.memory.OutOfMemoryError:
            result[f"t_{name}_s"] = None
            result[f"skip_{name}"] = "oom"
            pool.free_all_blocks()

    if len(outputs) >= 2:
        names = list(outputs)
        ref = outputs[names[0]]
        scale = float(xp.max(xp.abs(ref)))
        max_diff = max(
            float(xp.max(xp.abs(outputs[n] - ref))) for n in names[1:]
        )
        result["parity_max_rel_diff"] = max_diff / scale
        result["parity_ok"] = bool(max_diff <= 1e-8 * scale)

    new_times = {a: result.get(f"t_new_{a}_s") for a in ("lowk", "largek")}
    new_times = {a: t for a, t in new_times.items() if t is not None}
    if new_times:
        result["best_new_algo"] = min(new_times, key=new_times.get)
        result["model_matches_best"] = result["best_new_algo"] == result["model_algo"]

    print("BENCH_RESULT " + json.dumps(result), flush=True)

    del MPQ, cgto, rcgto, Ga, outputs
    pool.free_all_blocks()
