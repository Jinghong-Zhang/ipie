import numpy
import pytest
import scipy.sparse
from ipie.propagation.phaseless_generic import PhaselessGeneric
from ipie.hamiltonians import Generic as HamGeneric
from ipie.hamiltonians import Sparse as HamSparse
from ipie.systems import Generic
from ipie.utils.testing import shaped_normal

#Test with the complex cholesky case: the result should be the same using the sparse case

@pytest.mark.unit
def test_VHS_nonHermitian(nalpha, nbeta, nmo, naux, nwalkers, cmplx = True):
    sys = Generic(nelec=(nalpha, nbeta))
    chol = shaped_normal((naux, nmo, nmo), cmplx=cmplx)
    h1e = shaped_normal((nmo, nmo), cmplx=cmplx)
    hamgen = HamGeneric(
        h1e=numpy.array([h1e, h1e]),
        chol=chol.reshape((naux, nmo * nmo)).T.copy(),
        # h1e_mod=h1e.copy(),
        ecore=0,
        verbose=False,
    )
    A = .5 * (chol + chol.transpose(0, 2, 1).conj())
    A = A.reshape((naux, nmo * nmo)).T.copy()
    B = .5j * (chol - chol.transpose(0, 2, 1).conj())
    B = B.reshape((naux, nmo * nmo)).T.copy()
    Asp = scipy.sparse.csc_matrix(A)
    Bsp = scipy.sparse.csc_matrix(B)
    hamsparse = HamSparse(
        h1e=numpy.array([h1e, h1e]),
        chol=None,
        ecore=0,
        A = Asp,
        B = Bsp,
        verbose=False,
    )
    propagator = PhaselessGeneric(.01)
    xshifted = shaped_normal((2 * naux, nwalkers), cmplx= True)
    VHScomplex = propagator.construct_VHS(hamgen, xshifted)
    VHSsparse = propagator.construct_VHS(hamsparse, xshifted)
    assert numpy.allclose(VHScomplex, VHSsparse)


if __name__ == '__main__':
    test_VHS_nonHermitian(2, 2, 5, 10, 3)