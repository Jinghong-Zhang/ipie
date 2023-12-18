import numpy
import pytest

from ipie.config import MPI
from ipie.trial_wavefunction.single_det import SingleDet
from ipie.trial_wavefunction.single_det_ghf import SingleDetGHF
from ipie.utils.testing import get_random_nomsd
from ipie.hamiltonians.generic import GenericComplexChol
from ipie.hamiltonians.sparse import SparseNonHermitian

@pytest.mark.unit
def test_single_det_ghf():
    nbasis = 10
    naux = 5 * nbasis
    nalpha, nbeta = (5, 7)
    numpy.random.seed(7)

    h1e, chol, nuc, eri = generate_hamiltonian(nbasis, (nalpha, nbeta), cplx=True, sparse=True, sym=4)

    wavefunction = get_random_nomsd(nalpha, nbeta, nbasis, ndet=1)
    trial_cmplx = SingleDet(
        wavefunction[1][0],
        (nalpha, nbeta),
        nbasis,
    )
    trial_sp = SingleDet(
        wavefunction[1][0],
        (nalpha, nbeta),
        nbasis,
    )
    sys = Generic(nelec=(nalpha, nbeta))
    #manipulating the cholesky vectors
    A = .5 * (chol + chol.transpose(0, 2, 1).conj())
    A = A.reshape((nchol, nmo * nmo)).T.copy()
    B = .5j * (chol - chol.transpose(0, 2, 1).conj())
    B = B.reshape((nchol, nmo * nmo)).T.copy()
    Asp = scipy.sparse.csc_matrix(A)
    Bsp = scipy.sparse.csc_matrix(B)   
    chol = chol.reshape((nchol, nmo**2)).T.copy()

    hamcomp = GenericComplexChol(numpy.array([h1e, h1e], dtype=h1e.dtype), chol, nuc, verbose=True)
    hamsparse = SparseNonHermitian(numpy.array([h1e, h1e], dtype=h1e.dtype), Asp, Bsp, nuc, verbose=True)

    # results for the complex hamiltonian
    trial_cmplx.half_rotate(ham)
    rAa_cmplx = self._rAa
    rBa_cmplx = self._rBa
    rAb_cmplx = self._rAb
    rBb_cmplx = self._rBb
    trial_cmplx.calculate_energy(sys, ham)
    e_cmplx = self.energy

    #results for the sparse hamiltonian
    trial_sp.half_rotate()

    energy_ref = trial.energy

    psi0 = numpy.zeros((2 * nbasis, nalpha + nbeta), dtype=trial.psi0a.dtype)

    # no rotation is applied
    trial.psi0a, _ = numpy.linalg.qr(trial.psi0a)
    trial.psi0b, _ = numpy.linalg.qr(trial.psi0b)
    psi0[:nbasis, :nalpha] = trial.psi0a.copy()
    psi0[nbasis:, nalpha:] = trial.psi0b.copy()

    trial = SingleDetGHF(
        psi0,
        (nalpha, nbeta),
        nbasis,
    )
    trial.calculate_energy(sys, ham)
