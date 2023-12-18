import pytest
import numpy
import scipy.sparse
from ipie.utils.linalg import modified_cholesky
from ipie.utils.testing import generate_hamiltonian
from ipie.hamiltonians.generic import GenericComplexChol
from ipie.hamiltonians.sparse import SparseComplexChol, SparseRealChol, SparseNonHermitian
from itertools import product

# TODO: write tests with sparse hamiltonian.
@pytest.mark.unit
def test_sparse_complex_hamiltonian():
    numpy.random.seed(7)
    nmo = 13
    nelec = (4, 3)
    h1e, chol, nuc, eri = generate_hamiltonian(nmo, nelec, cplx=True, sparse=True, sym=4)
    nchol = chol.shape[0]
    chol = chol.reshape((nchol, nmo**2)).T.copy()
    ham = SparseComplexChol(numpy.array([h1e, h1e], dtype=h1e.dtype), chol, nuc, verbose=True)

@pytest.mark.unit
def test_sparse_h1emod():
    numpy.random.seed(7)
    nmo = 13
    nelec = (4, 3)
    h1e, chol, nuc, eri = generate_hamiltonian(nmo, nelec, cplx=True, sparse=False, sym=4)
    nchol = chol.shape[0]
    A = .5 * (chol + chol.transpose(0, 2, 1).conj())
    A = A.reshape((nchol, nmo * nmo)).T.copy()
    B = .5j * (chol - chol.transpose(0, 2, 1).conj())
    B = B.reshape((nchol, nmo * nmo)).T.copy()
    Asp = scipy.sparse.csc_matrix(A)
    Bsp = scipy.sparse.csc_matrix(B)   
    chol = chol.reshape((nchol, nmo**2)).T.copy()
    hamcomp = GenericComplexChol(numpy.array([h1e, h1e], dtype=h1e.dtype), chol, nuc, verbose=True)
    hamsparse = SparseNonHermitian(numpy.array([h1e, h1e], dtype=h1e.dtype), Asp, Bsp, nuc, verbose=True)
    assert numpy.allclose(hamcomp.h1e_mod, hamcomp.h1e_mod_cmplx)
    '''
    delta_hijkl = numpy.zeros_like(eri, dtype=numpy.complex128)
    for i, j, k, l in product(range(nmo), repeat=4):
        delta_hijkl[i,j,k,l] = hamcomp.hijkl(i,j,k,l) - hamcomp.hijkl_cmplx(i,j,k,l)
    delta_eri = numpy.zeros_like(eri, dtype=eri.dtype)
    for i, j, k, l in product(range(nmo), repeat=4):
        delta_eri[i,j,k,l] = hamcomp.hijkl(i,j,k,l) - eri[i,k,j,l]
    '''
    #print(numpy.allclose(delta_hijkl, numpy.zeros_like(delta_hijkl)))
    #print(numpy.max(delta_eri))
    #print(numpy.allclose(delta_eri, numpy.zeros_like(delta_eri)))
    #print(numpy.abs(hamcomp.hijkl(0, 1, 2, 3) - hamcomp.hijkl(1, 0, 3, 2)) < 1e-10)
    #print(hamcomp.h1e_mod - hamsparse.h1e_mod)
    assert numpy.allclose(hamcomp.h1e_mod, hamsparse.h1e_mod)
    return
    



@pytest.mark.unit
def test_sparse_real_hamiltonian():
    numpy.random.seed(7)
    nmo = 13
    nelec = (4, 3)
    h1e, chol, nuc, eri = generate_hamiltonian(nmo, nelec, cplx=False, sparse=True, sym=8)
    nchol = chol.shape[0]
    chol = chol.reshape((nchol, nmo**2)).T.copy()
    ham = SparseRealChol(numpy.array([h1e, h1e], dtype=h1e.dtype), chol, nuc, verbose=True)


if __name__ == "__main__":
    test_sparse_complex_hamiltonian()
    test_sparse_h1emod()
    test_sparse_real_hamiltonian()
