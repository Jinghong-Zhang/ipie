#!/usr/bin/env python
"""Timing-only AFQMC run on a fake KptISDF Hamiltonian (Nisdf = 18 * Nbasis).

Builds random ISDF vectors and cholM factors, an identity HF trial, and runs a
short real AFQMC calculation on GPU purely for timing. ``--propagation old``
monkeypatches the cbaa246 VHS path (per-chunk phi transpose, spin-separate
Taylor loops) so the e3487cc optimizations can be measured end to end.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

if os.environ.get("IPIE_USE_GPU") != "1":
    raise RuntimeError("Set IPIE_USE_GPU=1 before running this benchmark.")

from ipie.config import config

config.update_option("use_gpu", True)

import cupy
from mpi4py import MPI

from ipie.hamiltonians.kpt_hamiltonian import KptISDF
from ipie.qmc.afqmc import AFQMC
from ipie.systems.generic import Generic
from ipie.trial_wavefunction.single_det_kpt import KptSingleDet
from ipie.utils.backend import arraylib as xp
from ipie.utils.mpi import MPIHandler
from ipie.walkers.uhf_walkers import UHFWalkers

import legacy_kernels

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--kmesh", default="2,2,2")
parser.add_argument("--nbasis", type=int, default=100)
parser.add_argument("--nocc", type=int, default=9)
parser.add_argument("--nwalkers", type=int, default=32)
parser.add_argument("--nisdf-factor", type=int, default=18)
parser.add_argument("--naux-factor", type=int, default=4)
parser.add_argument("--nblocks", type=int, default=3)
parser.add_argument("--nsteps", type=int, default=5)
parser.add_argument("--timestep", type=float, default=0.005)
parser.add_argument("--seed", type=int, default=114514)
parser.add_argument("--propagation", choices=("old", "new"), default="new")
parser.add_argument("--scratch", default="/n/netscratch/joonholee_lab/Lab/jhzhang/benchmarks_kptisdf_amd")
args = parser.parse_args()

gpu_number_per_node = max(cupy.cuda.runtime.getDeviceCount(), 1)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
xp.cuda.Device(rank % gpu_number_per_node).use()

if args.propagation == "old":
    legacy_kernels.patch_old_propagation()

mesh = [int(x) for x in args.kmesh.split(",")]
nk = int(np.prod(mesh))
nbasis = args.nbasis
nocc = args.nocc
nisdf = args.nisdf_factor * nbasis
naux = args.naux_factor * nbasis
occ_ratio = 2 * nocc / nbasis
assert occ_ratio < 0.2, (
    f"occ_ratio={occ_ratio:.3f} >= 0.2 would select the dense-VHS branch; "
    "lower --nocc so the optimized Taylor path is exercised"
)

scratch_dir = os.path.join(
    args.scratch, f"fake_nk{nk}_nbsf{nbasis}_no{nocc}_nw{args.nwalkers}_{args.propagation}"
)
os.makedirs(scratch_dir, exist_ok=True)
os.chdir(scratch_dir)

rng = np.random.default_rng(args.seed)


def complex_rand(shape, scale):
    return scale * (rng.standard_normal(shape) + 1.0j * rng.standard_normal(shape))


# fractional k-point grid; find_self_inverse_set/find_Qplus operate on these
kpoints = (
    np.array(np.meshgrid(*[np.arange(n) / n for n in mesh], indexing="ij"))
    .reshape(3, -1)
    .T.copy()
)

hcore = complex_rand((nk, nbasis, nbasis), 0.1)
hcore = 0.5 * (hcore + hcore.conj().transpose(0, 2, 1))
hcore += np.diag(np.linspace(-1.0, 1.0, nbasis))[None]

cgto = complex_rand((nk, nisdf, nbasis), 1.0 / np.sqrt(nisdf))

from ipie.utils.kpt_conv import find_Qplus, find_self_inverse_set

Sset = find_self_inverse_set(kpoints)
Qplus = find_Qplus(kpoints)
nq = len(Sset) + len(Qplus)

cholM = complex_rand((nq, nisdf, naux), 1.0 / np.sqrt(naux))
# MPQ = cholM cholM^dagger, formed on GPU to keep setup fast
cholM_d = cupy.asarray(cholM)
MPQ = cupy.asnumpy(cupy.matmul(cholM_d, cholM_d.conj().transpose(0, 2, 1)))
del cholM_d
cupy.get_default_memory_pool().free_all_blocks()

if rank == 0:
    print(f"# fake KptISDF: nk={nk} nbasis={nbasis} nisdf={nisdf} naux={naux} "
          f"nocc={nocc} nwalkers={args.nwalkers} nq={nq} propagation={args.propagation}")

handler = MPIHandler(nmembers=1)
system = Generic(nelec=(nocc, nocc))
ham = KptISDF(
    np.array([hcore, hcore]),
    MPQ,
    cholM,
    cgto,
    kpoints,
    0.0,
    h1e_mod=np.zeros((2, nk, nbasis, nbasis), dtype=np.complex128),
)

noccs = np.full(nk, nocc, dtype=np.int64)
psi_a = np.zeros((nk, nbasis, nocc), dtype=np.complex128)
psi_b = np.zeros((nk, nbasis, nocc), dtype=np.complex128)
phi_a = np.zeros((nk, nbasis, nk, nocc), dtype=np.complex128)
phi_b = np.zeros((nk, nbasis, nk, nocc), dtype=np.complex128)
for ik in range(nk):
    psi_a[ik, :, :nocc] = np.eye(nbasis, nocc, dtype=np.complex128)
    psi_b[ik, :, :nocc] = np.eye(nbasis, nocc, dtype=np.complex128)
    phi_a[ik, :, ik, :nocc] = np.eye(nbasis, nocc, dtype=np.complex128)
    phi_b[ik, :, ik, :nocc] = np.eye(nbasis, nocc, dtype=np.complex128)
phia = phi_a.reshape(nk * nbasis, nk * nocc)
phib = phi_b.reshape(nk * nbasis, nk * nocc)

trial = KptSingleDet(
    np.concatenate([psi_a, psi_b], axis=2),
    nk,
    (nocc, nocc),
    nbasis,
    handler=handler,
    noccas=noccs,
)
trial.build()
trial.half_rotate(ham)

walkers = UHFWalkers(
    np.hstack([phia, phib]),
    nk * system.nup,
    nk * system.ndown,
    nk * ham.nbasis,
    args.nwalkers,
    mpi_handler=handler,
)
walkers.build(trial)
walkers.rhf = False  # propagate both spins so the fused Taylor path is exercised

afqmc = AFQMC.build(
    (nocc, nocc),
    ham,
    trial,
    walkers,
    args.nwalkers,
    args.seed,
    args.nsteps,
    args.nblocks,
    args.timestep,
    eq_timestep=args.timestep,
    eq_num_steps_per_block=2,
    num_eq_blocks=1,
    stabilize_freq=5,
    pop_control_freq=5,
    mpi_handler=handler,
    verbose=(rank == 0),
)

cupy.cuda.Stream.null.synchronize()
t0 = time.time()
afqmc.run(estimator_filename=os.path.join(scratch_dir, "estimates.h5"))
cupy.cuda.Stream.null.synchronize()
total_s = time.time() - t0

timer = afqmc.propagator.timer
result = {
    "event": "afqmc_timing",
    "propagation": args.propagation,
    "nk": nk,
    "nbasis": nbasis,
    "nisdf": nisdf,
    "naux": naux,
    "nocc": nocc,
    "nwalkers": args.nwalkers,
    "nblocks": args.nblocks,
    "nsteps_per_block": args.nsteps,
    "total_run_s": total_s,
    "tvhs_s": timer.tvhs,
    "tgemm_s": timer.tgemm,
    "tfbias_s": timer.tfbias,
    "tgf_s": timer.tgf,
    "tovlp_s": timer.tovlp,
    "tupdate_s": timer.tupdate,
}
if rank == 0:
    print("BENCH_RESULT " + json.dumps(result), flush=True)
afqmc.finalise(verbose=(rank == 0))
