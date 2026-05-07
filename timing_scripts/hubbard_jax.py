"""Small parity/timing probe for the JAX Hubbard single-site backend.

Example
-------
conda run -n ipierun env IPIE_USE_GPU=1 python timing_scripts/hubbard_jax.py
"""

import argparse
import time
from types import SimpleNamespace

import numpy

from ipie.hamiltonians.hubbard import Hubbard
from ipie.propagation.hubbard_generic import HubbardSingleSite
from ipie.trial_wavefunction.single_det import SingleDet
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import synchronize, to_host
from ipie.utils.mpi import MPIHandler
from ipie.walkers.walkers_dispatch import UHFWalkersTrial


def build_case(nbasis, nup, ndown, nwalkers, dt):
    h1 = numpy.zeros((nbasis, nbasis), dtype=numpy.float64)
    for i in range(nbasis - 1):
        h1[i, i + 1] = h1[i + 1, i] = -1.0
    h1[0, -1] = h1[-1, 0] = -1.0

    psi = numpy.zeros((nbasis, nup + ndown), dtype=numpy.complex128)
    psi[:, :nup] = numpy.eye(nbasis, dtype=numpy.complex128)[:, :nup]
    psi[:, nup:] = numpy.eye(nbasis, dtype=numpy.complex128)[:, -ndown:]

    hamiltonian = Hubbard(numpy.array([h1, h1]), U=4.0)
    trial = SingleDet(psi, (nup, ndown), nbasis)
    trial.build()
    trial.half_rotate(hamiltonian)
    walkers = UHFWalkersTrial(trial, psi, nup, ndown, nbasis, nwalkers, MPIHandler())
    walkers.build(trial)
    propagator = HubbardSingleSite(dt)
    propagator.build(hamiltonian, trial, walkers)
    return hamiltonian, trial, walkers, propagator


def numpy_walkers(walkers):
    return SimpleNamespace(
        nwalkers=walkers.nwalkers,
        nup=walkers.nup,
        ndown=walkers.ndown,
        nbasis=walkers.nbasis,
        rhf=walkers.rhf,
        phia=numpy.asarray(to_host(walkers.phia)).copy(),
        phib=numpy.asarray(to_host(walkers.phib)).copy(),
        inv_ovlp_a=numpy.asarray(to_host(walkers.inv_ovlp_a)).copy(),
        inv_ovlp_b=numpy.asarray(to_host(walkers.inv_ovlp_b)).copy(),
        weight=numpy.asarray(to_host(walkers.weight)).copy(),
        ovlp=numpy.asarray(to_host(walkers.ovlp)).copy(),
    )


def clone_walkers(walkers):
    return SimpleNamespace(
        nwalkers=walkers.nwalkers,
        nup=walkers.nup,
        ndown=walkers.ndown,
        nbasis=walkers.nbasis,
        rhf=walkers.rhf,
        phia=walkers.phia.copy(),
        phib=walkers.phib.copy(),
        inv_ovlp_a=walkers.inv_ovlp_a.copy(),
        inv_ovlp_b=walkers.inv_ovlp_b.copy(),
        weight=walkers.weight.copy(),
        ovlp=walkers.ovlp.copy(),
    )


def snapshot(walkers):
    names = ("phia", "phib", "inv_ovlp_a", "inv_ovlp_b", "weight", "ovlp")
    return {name: numpy.asarray(to_host(getattr(walkers, name))).copy() for name in names}


def max_error(actual, reference):
    err = 0.0
    for name, ref in reference.items():
        err = max(err, float(numpy.max(numpy.abs(actual[name] - ref))))
    return err


def time_kernel(label, fn, repeat):
    fn()
    synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    synchronize()
    elapsed = (time.perf_counter() - start) / repeat
    print(f"{label:>12s}: {elapsed:.6e} s/call")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nbasis", type=int, default=64)
    parser.add_argument("--nwalkers", type=int, default=256)
    parser.add_argument("--nup", type=int, default=32)
    parser.add_argument("--ndown", type=int, default=32)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--dt", type=float, default=0.01)
    args = parser.parse_args()

    hamiltonian, trial, walkers, propagator = build_case(
        args.nbasis, args.nup, args.ndown, args.nwalkers, args.dt
    )
    random_fields_host = numpy.random.default_rng(7).random((args.nbasis, args.nwalkers))

    cpu_ref = numpy_walkers(walkers)
    propagator._cpu_numba_two_body(cpu_ref, hamiltonian, trial, random_fields_host)
    ref = snapshot(cpu_ref)

    jax_prop = HubbardSingleSite(args.dt, backend="jax")
    jax_prop.build(hamiltonian, trial, walkers)
    jax_walkers = numpy_walkers(walkers)
    jax_prop._jax_two_body(jax_walkers, hamiltonian, trial, random_fields_host)
    print(f"{'jax error':>12s}: {max_error(snapshot(jax_walkers), ref):.6e}")

    if hasattr(xp, "RawKernel"):
        cuda_walkers = clone_walkers(walkers)
        random_fields_dev = xp.asarray(random_fields_host, dtype=xp.float64)
        propagator._cuda_two_body(cuda_walkers, hamiltonian, trial, random_fields_dev)
        print(f"{'cuda error':>12s}: {max_error(snapshot(cuda_walkers), ref):.6e}")

    time_kernel(
        "cpu_numba",
        lambda: propagator._cpu_numba_two_body(
            numpy_walkers(walkers), hamiltonian, trial, random_fields_host
        ),
        args.repeat,
    )
    time_kernel(
        "jax",
        lambda: jax_prop._jax_two_body(
            numpy_walkers(walkers), hamiltonian, trial, random_fields_host
        ),
        args.repeat,
    )
    if hasattr(xp, "RawKernel"):
        random_fields_dev = xp.asarray(random_fields_host, dtype=xp.float64)
        time_kernel(
            "cuda",
            lambda: propagator._cuda_two_body(
                clone_walkers(walkers), hamiltonian, trial, random_fields_dev
            ),
            args.repeat,
        )


if __name__ == "__main__":
    main()
