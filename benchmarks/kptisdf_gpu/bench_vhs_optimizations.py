#!/usr/bin/env python
"""Isolated GPU benchmarks for the e3487cc KptISDF VHS optimizations.

Compares, per case:
  - kernel: apply_VHS_to_phi_batch old (cbaa246: per-chunk phi transpose,
    2-buffer chunk estimate) vs new (hoisted phi copy, 4-buffer estimate)
  - taylor: the exp_nmax-term Taylor expansion of exp(VHS) applied to both
    spins: old kernel + separate spin loops vs new kernel + spins fused along
    the occupied index, plus new kernel + separate loops to isolate the fusion
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass

if os.environ.get("IPIE_USE_GPU") != "1":
    raise RuntimeError("Set IPIE_USE_GPU=1 before importing ipie GPU kernels.")

import cupy as cp

from ipie.propagation.phaseless_kpt import apply_VHS_to_phi_batch

from legacy_kernels import apply_VHS_to_phi_batch_old


@dataclass(frozen=True)
class Case:
    nk: int
    nbasis: int
    nocc: int
    nwalkers: int
    nisdf_factor: int = 18
    nq: int | None = None

    @property
    def nisdf(self) -> int:
        return self.nisdf_factor * self.nbasis

    @classmethod
    def parse(cls, text: str) -> "Case":
        fields = [int(x) for x in text.split(",")]
        if len(fields) not in (4, 5, 6):
            raise ValueError("case must be nk,nbasis,nocc,nwalkers[,nisdf_factor[,nq]]")
        nk, nbasis, nocc, nwalkers = fields[:4]
        nisdf_factor = fields[4] if len(fields) >= 5 else 18
        nq = fields[5] if len(fields) == 6 else None
        return cls(nk, nbasis, nocc, nwalkers, nisdf_factor, nq)


def complex_rand(shape, scale=1.0):
    return scale * (
        cp.random.standard_normal(shape, dtype=cp.float64)
        + 1.0j * cp.random.standard_normal(shape, dtype=cp.float64)
    )


def kpoint_maps(nk):
    q = cp.arange(nk)[:, None]
    k = cp.arange(nk)[None, :]
    return (k + q) % nk, (k - q) % nk


def elapsed_ms(func, args, repeats, warmups):
    cp.cuda.Stream.null.synchronize()
    out = None
    for _ in range(warmups):
        out = func(*args)
    cp.cuda.Stream.null.synchronize()
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    for _ in range(repeats):
        out = func(*args)
    end.record()
    end.synchronize()
    return cp.cuda.get_elapsed_time(start, end) / repeats, out


def rel_error(a, b):
    denom = max(float(cp.linalg.norm(b).get()), 1.0)
    return float((cp.linalg.norm(a - b) / denom).get())


def memory_snapshot():
    free_bytes, total_bytes = cp.cuda.Device().mem_info
    pool = cp.get_default_memory_pool()
    return {
        "gpu_used_gb": (total_bytes - free_bytes) / 1024**3,
        "gpu_total_gb": total_bytes / 1024**3,
        "pool_used_gb": pool.used_bytes() / 1024**3,
        "pool_total_gb": pool.total_bytes() / 1024**3,
    }


def make_inputs(case, seed):
    cp.random.seed(seed)
    nq = case.nq or (case.nk // 2 + 1)
    kpq_mat, kmq_mat = kpoint_maps(case.nk)
    unique_qs = cp.arange(nq, dtype=cp.int64)
    # scale so repeated VHS applications stay numerically tame
    scale = 1.0 / (case.nbasis * case.nisdf) ** 0.25
    cgto = complex_rand((case.nk, case.nisdf, case.nbasis), scale)
    Lx = complex_rand((case.nwalkers, nq, case.nisdf), scale)
    Lconjx = complex_rand((case.nwalkers, nq, case.nisdf), scale)
    return nq, kpq_mat, kmq_mat, unique_qs, cgto, Lx, Lconjx


def taylor_separate(kernel, cgto, Lx, Lconjx, phia, phib, kpq, kmq, uq, exp_nmax):
    acc_a = phia.copy()
    Temp = phia.copy()
    for n in range(1, exp_nmax + 1):
        Temp = kernel(cgto, Lx, Lconjx, Temp, kpq, kmq, uq) / n
        acc_a += Temp
    acc_b = phib.copy()
    Temp = phib.copy()
    for n in range(1, exp_nmax + 1):
        Temp = kernel(cgto, Lx, Lconjx, Temp, kpq, kmq, uq) / n
        acc_b += Temp
    return acc_a, acc_b


def taylor_fused(kernel, cgto, Lx, Lconjx, phia, phib, kpq, kmq, uq, exp_nmax):
    nw = phia.shape[0]
    nkbsf = phia.shape[1]
    nk = kpq.shape[1]
    nocca = phia.shape[-1] // nk
    noccb = phib.shape[-1] // nk
    acc_a = phia.copy()
    acc_b = phib.copy()
    Temp = cp.concatenate(
        (
            phia.reshape(nw, nkbsf, nk, nocca),
            phib.reshape(nw, nkbsf, nk, noccb),
        ),
        axis=3,
    ).reshape(nw, nkbsf, nk * (nocca + noccb))
    for n in range(1, exp_nmax + 1):
        Temp = kernel(cgto, Lx, Lconjx, Temp, kpq, kmq, uq) / n
        Temp_split = Temp.reshape(nw, nkbsf, nk, nocca + noccb)
        acc_a += Temp_split[:, :, :, :nocca].reshape(nw, nkbsf, nk * nocca)
        acc_b += Temp_split[:, :, :, nocca:].reshape(nw, nkbsf, nk * noccb)
    return acc_a, acc_b


def run_kernel_case(case, args):
    nq, kpq, kmq, uq, cgto, Lx, Lconjx = make_inputs(case, args.seed)
    scale = 1.0 / (case.nbasis * case.nisdf) ** 0.25
    phi = complex_rand((case.nwalkers, case.nk * case.nbasis, case.nk * case.nocc), scale)
    cp.get_default_memory_pool().free_all_blocks()
    old_ms, old = elapsed_ms(
        apply_VHS_to_phi_batch_old, (cgto, Lx, Lconjx, phi, kpq, kmq, uq), args.repeats, args.warmups
    )
    new_ms, new = elapsed_ms(
        apply_VHS_to_phi_batch, (cgto, Lx, Lconjx, phi, kpq, kmq, uq), args.repeats, args.warmups
    )
    err = None if args.no_check else rel_error(new, old)
    return {
        "mode": "kernel",
        "case": asdict(case),
        "nq": nq,
        "old_ms": old_ms,
        "new_ms": new_ms,
        "speedup": old_ms / new_ms,
        "rel_error": err,
        "memory": memory_snapshot(),
    }


def run_taylor_case(case, args):
    nq, kpq, kmq, uq, cgto, Lx, Lconjx = make_inputs(case, args.seed)
    scale = 1.0 / (case.nbasis * case.nisdf) ** 0.25
    phia = complex_rand((case.nwalkers, case.nk * case.nbasis, case.nk * case.nocc), scale)
    phib = complex_rand((case.nwalkers, case.nk * case.nbasis, case.nk * case.nocc), scale)
    common = (cgto, Lx, Lconjx, phia, phib, kpq, kmq, uq, args.exp_nmax)
    cp.get_default_memory_pool().free_all_blocks()
    old_sep_ms, old_out = elapsed_ms(
        taylor_separate, (apply_VHS_to_phi_batch_old,) + common, args.repeats, args.warmups
    )
    new_sep_ms, _ = elapsed_ms(
        taylor_separate, (apply_VHS_to_phi_batch,) + common, args.repeats, args.warmups
    )
    fused_ms, fused_out = elapsed_ms(
        taylor_fused, (apply_VHS_to_phi_batch,) + common, args.repeats, args.warmups
    )
    err = None
    if not args.no_check:
        err = max(rel_error(fused_out[0], old_out[0]), rel_error(fused_out[1], old_out[1]))
    return {
        "mode": "taylor",
        "case": asdict(case),
        "nq": nq,
        "exp_nmax": args.exp_nmax,
        "old_kernel_separate_ms": old_sep_ms,
        "new_kernel_separate_ms": new_sep_ms,
        "new_kernel_fused_ms": fused_ms,
        "kernel_only_speedup": old_sep_ms / new_sep_ms,
        "fusion_only_speedup": new_sep_ms / fused_ms,
        "total_speedup": old_sep_ms / fused_ms,
        "rel_error": err,
        "memory": memory_snapshot(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--mode", choices=("kernel", "taylor", "both"), default="both")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--exp-nmax", type=int, default=6)
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args()
    dev = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
    print(json.dumps({"event": "device", "device_id": dev.id, "name": name, "memory": memory_snapshot()}), flush=True)
    for case_text in args.case:
        case = Case.parse(case_text)
        if args.mode in ("kernel", "both"):
            print(json.dumps(run_kernel_case(case, args)), flush=True)
        if args.mode in ("taylor", "both"):
            print(json.dumps(run_taylor_case(case, args)), flush=True)
        cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
