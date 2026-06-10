#!/usr/bin/env python
"""Isolated old-vs-new KptISDF local-energy kernel benchmark (GPU).

Builds random ISDF tensors at production-like sizes, verifies on-GPU parity of
the frozen 7cb863c kernels against the optimized ones, then times the ecoul
and exx kernels separately. Emits one ``BENCH_RESULT {json}`` line.
"""

import argparse
import json
import os
import time

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

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--kmesh", default="2,2,2")
parser.add_argument("--nbasis", type=int, default=100)
parser.add_argument("--nocc", type=int, default=9)
parser.add_argument("--nwalkers", type=int, default=32)
parser.add_argument("--nisdf-factor", type=int, default=18)
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--seed", type=int, default=114514)
parser.add_argument("--skip-old", action="store_true",
                    help="skip the legacy kernels (e.g. when they would OOM)")
args = parser.parse_args()

mesh = [int(x) for x in args.kmesh.split(",")]
nk = int(np.prod(mesh))
nbasis = args.nbasis
nocc = args.nocc
nwalkers = args.nwalkers
nisdf = args.nisdf_factor * nbasis

rng = np.random.default_rng(args.seed)


def complex_rand(shape, scale=1.0):
    return scale * (rng.standard_normal(shape) + 1.0j * rng.standard_normal(shape))


# fractional k-point grid in (-0.5, 0.5] as required by BZ_to_1BZ matching
def frac_grid(n):
    c = np.fft.fftfreq(n)
    c[c == -0.5] = 0.5
    return c


kpoints = (
    np.array(np.meshgrid(*[frac_grid(n) for n in mesh], indexing="ij"))
    .reshape(3, -1)
    .T.copy()
)
kpq_mat = construct_kpq(kpoints)
Sset = find_self_inverse_set(kpoints)
Qplus = find_Qplus(kpoints)
nq = len(Sset) + len(Qplus)

print(
    f"# bench_local_energy: nk={nk} nbasis={nbasis} nisdf={nisdf} nocc={nocc} "
    f"nwalkers={nwalkers} nq={nq} (|Sset|={len(Sset)}, |Qplus|={len(Qplus)})",
    flush=True,
)

MPQ = xp.asarray(complex_rand((nq, nisdf, nisdf), 1.0 / nisdf))
cgto = xp.asarray(complex_rand((nk, nisdf, nbasis), 1.0 / np.sqrt(nisdf)))
rcgtoa = xp.asarray(complex_rand((nk, nisdf, nocc), 1.0 / np.sqrt(nisdf)))
rcgtob = xp.asarray(complex_rand((nk, nisdf, nocc), 1.0 / np.sqrt(nisdf)))
Ga = xp.asarray(complex_rand((nwalkers, nk, nocc, nk, nbasis), 1.0 / np.sqrt(nk * nbasis)))
Gb = xp.asarray(complex_rand((nwalkers, nk, nocc, nk, nbasis), 1.0 / np.sqrt(nk * nbasis)))


def timed(fn, repeats):
    fn()  # warmup (kernel compilation, pool growth)
    cupy.cuda.Stream.null.synchronize()
    times = []
    for _ in range(repeats):
        t0 = time.time()
        out = fn()
        cupy.cuda.Stream.null.synchronize()
        times.append(time.time() - t0)
    return out, min(times)


result = {
    "event": "local_energy_kernel_timing",
    "nk": nk,
    "nbasis": nbasis,
    "nisdf": nisdf,
    "nocc": nocc,
    "nwalkers": nwalkers,
    "nq": nq,
}

ecoul_new, t_ecoul_new = timed(
    lambda: le.kpt_isdf_ecoul_kernel_gpu(
        MPQ, rcgtoa, rcgtob, cgto, Ga, Gb, kpq_mat, Sset, Qplus
    ),
    args.repeats,
)
result["t_ecoul_new_s"] = t_ecoul_new

exx_new_a, t_exx_new = timed(
    lambda: le.kpt_isdf_exx_kernel_gpu(MPQ, rcgtoa, cgto, Ga, kpq_mat, Sset, Qplus),
    args.repeats,
)
result["t_exx_new_s"] = t_exx_new

if not args.skip_old:
    ecoul_old, t_ecoul_old = timed(
        lambda: legacy.kpt_isdf_ecoul_kernel_gpu(
            MPQ, rcgtoa, rcgtob, cgto, Ga, Gb, kpq_mat, Sset, Qplus
        ),
        args.repeats,
    )
    exx_old_a, t_exx_old = timed(
        lambda: legacy.kpt_isdf_exx_kernel_gpu(MPQ, rcgtoa, cgto, Ga, kpq_mat, Sset, Qplus),
        args.repeats,
    )
    ecoul_diff = float(xp.max(xp.abs(ecoul_new - ecoul_old)))
    exx_diff = float(xp.max(xp.abs(exx_new_a - exx_old_a)))
    scale_ecoul = float(xp.max(xp.abs(ecoul_old)))
    scale_exx = float(xp.max(xp.abs(exx_old_a)))
    result.update(
        t_ecoul_old_s=t_ecoul_old,
        t_exx_old_s=t_exx_old,
        ecoul_speedup=t_ecoul_old / t_ecoul_new,
        exx_speedup=t_exx_old / t_exx_new,
        ecoul_rel_diff=ecoul_diff / scale_ecoul,
        exx_rel_diff=exx_diff / scale_exx,
    )
    assert ecoul_diff <= 1e-8 * scale_ecoul, f"ecoul mismatch: {ecoul_diff} vs scale {scale_ecoul}"
    assert exx_diff <= 1e-8 * scale_exx, f"exx mismatch: {exx_diff} vs scale {scale_exx}"
    print("# GPU parity old-vs-new: OK", flush=True)

print("BENCH_RESULT " + json.dumps(result), flush=True)
