import numpy as np
import pytest

from ipie.config import MPI
from ipie.estimators.energy import local_energy
from ipie.propagation.force_bias import construct_force_bias_batch_single_det
from ipie.systems.generic import Generic
from ipie.trial_wavefunction.cisd import CISD
from ipie.trial_wavefunction.single_det import SingleDet
from ipie.utils.mpi import MPIHandler
from ipie.utils.testing import generate_hamiltonian
from ipie.walkers.uhf_walkers import UHFWalkers


def random_orthonormal(nbasis, nocc, seed):
    rng = np.random.default_rng(seed)
    mat = rng.standard_normal((nbasis, nocc))
    q, _ = np.linalg.qr(mat)
    return q[:, :nocc]


def build_real_hamiltonian(nbasis, nelec, seed):
    np.random.seed(seed)
    h1e, chol, ecore, _ = generate_hamiltonian(nbasis, nelec, cplx=False)
    h1 = np.stack([h1e, h1e])
    chol = chol.transpose(1, 2, 0).reshape(nbasis * nbasis, -1)
    system = Generic(nelec)
    return system, h1, chol, ecore


def build_walkers(initial_wfn, nelec, nbasis, rhf=False):
    walkers = UHFWalkers(
        initial_wfn,
        nelec[0],
        nelec[1],
        nbasis,
        1,
        mpi_handler=MPIHandler(),
    )
    walkers.rhf = rhf
    return walkers


@pytest.mark.unit
def test_cisd_half_rotate_shapes_restricted():
    nbasis = 8
    nelec = (3, 3)
    system, h1, chol, ecore = build_real_hamiltonian(nbasis, nelec, seed=11)

    psi0 = random_orthonormal(nbasis, nelec[0], 13)
    ref = np.hstack([psi0, psi0])
    rdm1 = np.stack([psi0 @ psi0.T.conj(), psi0 @ psi0.T.conj()])

    trial = CISD(
        ref,
        nelec,
        nbasis,
        c1a=np.zeros((nelec[0], nbasis - nelec[0])),
        c2aa=np.zeros((nelec[0], nbasis - nelec[0], nelec[0], nbasis - nelec[0])),
        rdm1=rdm1,
    )
    from ipie.hamiltonians.generic import GenericRealChol

    hamiltonian = GenericRealChol(h1, chol, ecore)
    trial.build()
    trial.half_rotate(hamiltonian, comm=MPI.COMM_WORLD)

    assert trial._rH1a.shape == (nelec[0], nbasis)
    assert trial._rH1b.shape == (nelec[1], nbasis)
    assert trial._rchola.shape == (hamiltonian.nchol, nbasis * nelec[0])
    assert trial._rcholb.shape == (hamiltonian.nchol, nbasis * nelec[1])
    assert trial.rchola.shape == (nelec[0], nbasis, hamiltonian.nchol)
    assert trial.rcholb.shape == (nelec[1], nbasis, hamiltonian.nchol)


@pytest.mark.unit
def test_cisd_zero_amplitudes_matches_single_det_restricted():
    from ipie.hamiltonians.generic import GenericRealChol

    nbasis = 7
    nelec = (3, 3)
    system, h1, chol, ecore = build_real_hamiltonian(nbasis, nelec, seed=17)
    hamiltonian = GenericRealChol(h1, chol, ecore)

    psi0 = random_orthonormal(nbasis, nelec[0], 19)
    walker_occ = random_orthonormal(nbasis, nelec[0], 23)
    ref = np.hstack([psi0, psi0])
    walker_ref = np.hstack([walker_occ, walker_occ])
    rdm1 = np.stack([psi0 @ psi0.T.conj(), psi0 @ psi0.T.conj()])

    trial_sd = SingleDet(ref, nelec, nbasis)
    trial_sd.build()
    trial_sd.half_rotate(hamiltonian, comm=MPI.COMM_WORLD)

    trial_cisd = CISD(
        ref,
        nelec,
        nbasis,
        c1a=np.zeros((nelec[0], nbasis - nelec[0])),
        c2aa=np.zeros((nelec[0], nbasis - nelec[0], nelec[0], nbasis - nelec[0])),
        rdm1=rdm1,
    )
    trial_cisd.build()
    trial_cisd.half_rotate(hamiltonian, comm=MPI.COMM_WORLD)

    walkers_sd = build_walkers(walker_ref, nelec, nbasis, rhf=True)
    walkers_cisd = build_walkers(walker_ref, nelec, nbasis, rhf=True)

    ovlp_sd = trial_sd.calc_greens_function(walkers_sd)
    ovlp_cisd = trial_cisd.calc_greens_function(walkers_cisd)
    fb_sd = construct_force_bias_batch_single_det(
        hamiltonian, walkers_sd, trial_sd._rchola, trial_sd._rcholb
    )
    fb_cisd = trial_cisd.calc_force_bias(hamiltonian, walkers_cisd, MPI)
    energy_sd = local_energy(system, hamiltonian, walkers_sd, trial_sd)
    energy_cisd = local_energy(system, hamiltonian, walkers_cisd, trial_cisd)

    np.testing.assert_allclose(ovlp_cisd, ovlp_sd, atol=1e-10)
    np.testing.assert_allclose(fb_cisd, fb_sd, atol=1e-10)
    np.testing.assert_allclose(energy_cisd, energy_sd, atol=1e-10)


@pytest.mark.unit
def test_cisd_zero_amplitudes_matches_single_det_unrestricted():
    from ipie.hamiltonians.generic import GenericRealChol

    nbasis = 8
    nelec = (3, 2)
    system, h1, chol, ecore = build_real_hamiltonian(nbasis, nelec, seed=29)
    hamiltonian = GenericRealChol(h1, chol, ecore)

    psi0a = random_orthonormal(nbasis, nelec[0], 31)
    mo_coeffb = random_orthonormal(nbasis, nbasis, 37)
    psi0b = mo_coeffb[:, : nelec[1]]
    walker_a = random_orthonormal(nbasis, nelec[0], 41)
    walker_b = mo_coeffb @ random_orthonormal(nbasis, nelec[1], 43)

    ref = np.hstack([psi0a, psi0b])
    walker_ref = np.hstack([walker_a, walker_b])
    rdm1 = np.stack([psi0a @ psi0a.T.conj(), psi0b @ psi0b.T.conj()])

    trial_sd = SingleDet(ref, nelec, nbasis)
    trial_sd.build()
    trial_sd.half_rotate(hamiltonian, comm=MPI.COMM_WORLD)

    trial_cisd = CISD(
        ref,
        nelec,
        nbasis,
        c1a=np.zeros((nelec[0], nbasis - nelec[0])),
        c2aa=np.zeros((nelec[0], nbasis - nelec[0], nelec[0], nbasis - nelec[0])),
        c1b=np.zeros((nelec[1], nbasis - nelec[1])),
        c2ab=np.zeros((nelec[0], nbasis - nelec[0], nelec[1], nbasis - nelec[1])),
        c2bb=np.zeros((nelec[1], nbasis - nelec[1], nelec[1], nbasis - nelec[1])),
        mo_coeffb=mo_coeffb,
        rdm1=rdm1,
    )
    trial_cisd.build()
    trial_cisd.half_rotate(hamiltonian, comm=MPI.COMM_WORLD)

    walkers_sd = build_walkers(walker_ref, nelec, nbasis, rhf=False)
    walkers_cisd = build_walkers(walker_ref, nelec, nbasis, rhf=False)

    ovlp_sd = trial_sd.calc_greens_function(walkers_sd)
    ovlp_cisd = trial_cisd.calc_greens_function(walkers_cisd)
    fb_sd = construct_force_bias_batch_single_det(
        hamiltonian, walkers_sd, trial_sd._rchola, trial_sd._rcholb
    )
    fb_cisd = trial_cisd.calc_force_bias(hamiltonian, walkers_cisd, MPI)
    energy_sd = local_energy(system, hamiltonian, walkers_sd, trial_sd)
    energy_cisd = local_energy(system, hamiltonian, walkers_cisd, trial_cisd)

    np.testing.assert_allclose(ovlp_cisd, ovlp_sd, atol=1e-10)
    np.testing.assert_allclose(fb_cisd, fb_sd, atol=1e-10)
    np.testing.assert_allclose(energy_cisd, energy_sd, atol=1e-10)