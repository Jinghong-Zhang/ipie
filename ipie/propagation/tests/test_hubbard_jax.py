import numpy
import pytest

jax = pytest.importorskip("jax")

from ipie.config import config
from ipie.hamiltonians.hubbard import Hubbard
from ipie.propagation.hubbard_generic import HubbardSingleSite
from ipie.qmc.afqmc import AFQMC
from ipie.trial_wavefunction.single_det import SingleDet
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import to_host
from ipie.utils.mpi import MPIHandler
from ipie.walkers.walkers_dispatch import UHFWalkersTrial


def _hubbard_inputs(nelec=(3, 2), nbasis=6, nwalkers=5, rhf=False):
    if rhf:
        assert nelec[0] == nelec[1]
    h1 = numpy.zeros((nbasis, nbasis), dtype=numpy.float64)
    for i in range(nbasis - 1):
        h1[i, i + 1] = h1[i + 1, i] = -1.0
    h1[0, -1] = h1[-1, 0] = -1.0
    h1 += numpy.diag(numpy.linspace(-0.2, 0.2, nbasis))

    psi = numpy.zeros((nbasis, sum(nelec)), dtype=numpy.complex128)
    psi[:, : nelec[0]] = numpy.eye(nbasis, dtype=numpy.complex128)[:, : nelec[0]]
    if rhf:
        psi[:, nelec[0] :] = psi[:, : nelec[0]]
    else:
        psi[:, nelec[0] :] = numpy.eye(nbasis, dtype=numpy.complex128)[:, -nelec[1] :]

    ham = Hubbard(numpy.array([h1, h1]), U=4.0)
    trial = SingleDet(psi, nelec, nbasis)
    trial.build()
    trial.half_rotate(ham)
    return ham, trial, psi, nwalkers


def _new_walkers(trial, initial_walker, nelec, nbasis, nwalkers, rhf=False, dead=False):
    walkers = UHFWalkersTrial(
        trial, initial_walker, nelec[0], nelec[1], nbasis, nwalkers, MPIHandler()
    )
    walkers.build(trial)
    walkers.rhf = rhf
    if dead and nwalkers > 1:
        walkers.weight[1] = 0.0
    return walkers


def _snapshot(walkers):
    fields = ("phia", "phib", "inv_ovlp_a", "inv_ovlp_b", "weight", "ovlp")
    snap = {}
    for name in fields:
        value = getattr(walkers, name)
        snap[name] = None if value is None else numpy.asarray(to_host(value)).copy()
    if getattr(walkers, "hybrid_energy", None) is not None:
        snap["hybrid_energy"] = numpy.asarray(to_host(walkers.hybrid_energy)).copy()
    return snap


def _assert_snapshots_close(actual, expected, atol=1e-10, rtol=1e-10):
    for key, ref in expected.items():
        if ref is None:
            assert actual[key] is None
        else:
            numpy.testing.assert_allclose(actual[key], ref, atol=atol, rtol=rtol, err_msg=key)


def _cpu_full_step(prop, walkers, hamiltonian, trial, eshift, random_fields):
    if walkers.ovlp is None or len(walkers.ovlp) != walkers.nwalkers:
        walkers.ovlp = trial.calc_overlap(walkers)
    ovlp = walkers.ovlp.copy()
    prop.kinetic_importance_sampling(walkers, trial)
    prop._cpu_numba_two_body(walkers, hamiltonian, trial, random_fields)
    prop.kinetic_importance_sampling(walkers, trial)
    walkers.hybrid_energy = (-(xp.log(walkers.ovlp / ovlp)) / prop.dt).real
    walkers.weight *= numpy.exp(prop.dt * eshift)


@pytest.mark.unit
@pytest.mark.parametrize("spin_decomp", [True, False])
@pytest.mark.parametrize("rhf", [False, True])
@pytest.mark.parametrize("dead", [False, True])
def test_jax_two_body_matches_cpu_paths(spin_decomp, rhf, dead):
    nelec = (3, 3) if rhf else (3, 2)
    ham, trial, initial_walker, nwalkers = _hubbard_inputs(nelec=nelec, rhf=rhf)
    random_fields = numpy.random.default_rng(11).random((ham.nbasis, nwalkers))

    refs = {}
    for path in ("legacy", "numba", "einsum", "jax"):
        walkers = _new_walkers(
            trial, initial_walker, nelec, ham.nbasis, nwalkers, rhf=rhf, dead=dead
        )
        prop = HubbardSingleSite(
            0.01, spin_decomp=spin_decomp, backend="jax" if path == "jax" else "auto"
        )
        prop.build(ham, trial, walkers)
        if path == "legacy":
            prop._legacy_two_body(walkers, ham, trial, random_fields=random_fields)
        elif path == "numba":
            prop._cpu_numba_two_body(walkers, ham, trial, random_fields)
        elif path == "einsum":
            prop._einsum_two_body(walkers, ham, trial, random_fields)
        else:
            prop._jax_two_body(walkers, ham, trial, random_fields=random_fields)
        refs[path] = _snapshot(walkers)

    _assert_snapshots_close(refs["numba"], refs["legacy"])
    _assert_snapshots_close(refs["einsum"], refs["legacy"])
    _assert_snapshots_close(refs["jax"], refs["legacy"])


@pytest.mark.unit
@pytest.mark.parametrize("spin_decomp", [True, False])
@pytest.mark.parametrize("dead", [False, True])
def test_jax_full_step_matches_cpu_full_step(spin_decomp, dead):
    nelec = (3, 2)
    ham, trial, initial_walker, nwalkers = _hubbard_inputs(nelec=nelec)
    random_fields = numpy.random.default_rng(17).random((ham.nbasis, nwalkers))
    eshift = 0.25

    walkers_ref = _new_walkers(trial, initial_walker, nelec, ham.nbasis, nwalkers, dead=dead)
    prop_ref = HubbardSingleSite(0.01, spin_decomp=spin_decomp)
    prop_ref.build(ham, trial, walkers_ref)
    _cpu_full_step(prop_ref, walkers_ref, ham, trial, eshift, random_fields)

    walkers_jax = _new_walkers(trial, initial_walker, nelec, ham.nbasis, nwalkers, dead=dead)
    prop_jax = HubbardSingleSite(0.01, spin_decomp=spin_decomp, backend="jax")
    prop_jax.build(ham, trial, walkers_jax)
    prop_jax._jax_full_step(walkers_jax, ham, trial, eshift, random_fields=random_fields)

    _assert_snapshots_close(_snapshot(walkers_jax), _snapshot(walkers_ref), atol=5e-10, rtol=5e-10)


@pytest.mark.unit
def test_afqmc_build_accepts_hubbard_jax_backend():
    nelec = (2, 2)
    ham, trial, _, _ = _hubbard_inputs(nelec=nelec, nbasis=4, nwalkers=3)
    afqmc = AFQMC.build(
        nelec,
        ham,
        trial,
        num_walkers=3,
        num_steps_per_block=1,
        num_blocks=1,
        timestep=0.01,
        seed=7,
        propagator_backend="jax",
        verbose=False,
    )
    assert afqmc.propagator.backend == "jax"


@pytest.mark.unit
def test_jax_propagate_walkers_smoke():
    nelec = (2, 2)
    ham, trial, initial_walker, nwalkers = _hubbard_inputs(nelec=nelec, nbasis=4, nwalkers=3)
    walkers = _new_walkers(trial, initial_walker, nelec, ham.nbasis, nwalkers)
    prop = HubbardSingleSite(0.01, backend="jax")
    prop.build(ham, trial, walkers)
    prop.propagate_walkers(walkers, ham, trial, 0.0)

    assert numpy.all(numpy.isfinite(numpy.asarray(to_host(walkers.weight))))
    assert numpy.all(numpy.isfinite(numpy.asarray(to_host(walkers.hybrid_energy))))


@pytest.mark.unit
def test_jax_propagate_walkers_leaves_writable_buffers_for_reorthogonalisation():
    nelec = (2, 2)
    ham, trial, initial_walker, nwalkers = _hubbard_inputs(nelec=nelec, nbasis=4, nwalkers=3)
    walkers = _new_walkers(trial, initial_walker, nelec, ham.nbasis, nwalkers)
    prop = HubbardSingleSite(0.01, backend="jax")
    prop.build(ham, trial, walkers)

    prop.propagate_walkers(walkers, ham, trial, 0.0)
    walkers.orthogonalise()

    assert walkers.phia.flags.writeable
    assert walkers.phib.flags.writeable
    assert numpy.all(numpy.isfinite(numpy.asarray(to_host(walkers.detR))))


@pytest.mark.gpu
def test_jax_two_body_matches_cuda_path():
    if not config.get_option("use_gpu") or not hasattr(xp, "RawKernel"):
        pytest.skip("CuPy CUDA backend is not active.")

    nelec = (3, 2)
    ham, trial, initial_walker, nwalkers = _hubbard_inputs(nelec=nelec)
    random_fields = xp.asarray(
        numpy.random.default_rng(23).random((ham.nbasis, nwalkers)), dtype=xp.float64
    )

    walkers_cuda = _new_walkers(trial, initial_walker, nelec, ham.nbasis, nwalkers)
    prop_cuda = HubbardSingleSite(0.01)
    prop_cuda.build(ham, trial, walkers_cuda)
    prop_cuda._cuda_two_body(walkers_cuda, ham, trial, random_fields)

    walkers_jax = _new_walkers(trial, initial_walker, nelec, ham.nbasis, nwalkers)
    prop_jax = HubbardSingleSite(0.01, backend="jax")
    prop_jax.build(ham, trial, walkers_jax)
    prop_jax._jax_two_body(walkers_jax, ham, trial, random_fields=random_fields)

    _assert_snapshots_close(_snapshot(walkers_jax), _snapshot(walkers_cuda), atol=5e-10, rtol=5e-10)
