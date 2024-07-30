import numpy

from ipie.hamiltonians.generic_base import GenericBase
from ipie.utils.backend import arraylib as xp
from ipie.utils.kpt_conv import find_gamma_pt, find_inverted_index_batched, find_translated_index_batched

from ipie.utils.io import (
    from_qmcpack_dense,
    from_qmcpack_sparse,
    read_hamiltonian,
)

def construct_kpq(kpts_frac):
    nk = kpts_frac.shape[0]
    idx_kpq_mat = numpy.zeros((nk, nk), dtype=numpy.int64)
    for iq in range(nk):
        qvec = kpts_frac[iq]
        idx_kpq = find_translated_index_batched(kpts_frac, qvec)
        idx_kpq_mat[iq] = idx_kpq
    assert numpy.allclose(idx_kpq_mat, idx_kpq_mat.T)
    return idx_kpq_mat

def construct_mq(kpts_frac):
    return find_inverted_index_batched(kpts_frac)

def construct_h1e_mod(chol, h1e, ikpq_mat, imq_vec, h1e_mod):
    nk, nbasis = h1e.shape[1], h1e.shape[2]
    self_int = numpy.zeros((nk, nbasis, nbasis), dtype=numpy.complex128)
    for ik in range(nk):
        for iq in range(nk):
            ikpq = ikpq_mat[ik, iq]
            imq = imq_vec[iq]
            self_int[ik] += .5 * numpy.einsum('gpr, grq -> pq', chol[:, ik, :, iq, :], chol[:, ikpq, :, imq, :])
    h1e_mod = h1e - self_int


class KptComplexChol(GenericBase):
    """Class for ab-initio k-point Hamiltonian with 4-fold complex symmetric integrals.
    Can be created by passing the one and two electron integrals directly.
    """

    def __init__(self, h1e, chol, kpts, ecore=0.0, verbose=False):
        assert h1e.shape[0] == 2
        assert len(h1e.shape) == 4 # shape = nspin, nk, nbasis, nbasis
        super().__init__(h1e, ecore, verbose)

        self.chol = numpy.array(chol, dtype=numpy.complex128)  # [nchol, Nk, M, Nk, M] (gamma, k, p, q, r)
        self.kpts = kpts
        self.ikpq_mat = construct_kpq(self.kpts)
        self.imq_vec = construct_mq(self.kpts)
        self.igamma = find_gamma_pt(self.kpts)
        self.nk = self.kpts.shape[0]
        self.nchol = self.chol.shape[0]

        self.chunked = False

        # this is the one-body part that comes out of re-ordering the 2-body operators
        h1e_mod = numpy.zeros(self.H1.shape, dtype=self.H1.dtype)
        construct_h1e_mod(self.chol, self.H1, self.ikpq_mat, self.imq_vec, h1e_mod)
        self.h1e_mod = xp.array(h1e_mod)

        if verbose:
            mem = self.A.nbytes / (1024.0**3) * 3
            print("# Number of orbitals: %d" % self.nbasis)
            print(f"# Approximate memory required by Cholesky + A&B vectors {mem:f} GB")
            print("# Number of Cholesky vectors: %d" % (self.nchol))
            print("# Finished setting up KptComplexChol object.")
