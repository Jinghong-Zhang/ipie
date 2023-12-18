# Copyright 2022 The ipie Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Joonho Lee
#
# TODO: finish implementing.

import sys
import numpy
import math
import scipy.sparse
from ipie.hamiltonians.generic_base import GenericBase
from ipie.utils.pack_numba import pack_cholesky

from ipie.utils.io import (
    from_qmcpack_dense,
    from_qmcpack_sparse,
    read_hamiltonian,
)


def construct_h1e_mod(chol, h1e, h1e_mod):
    # Subtract one-body bit following reordering of 2-body operators.
    # Eqn (17) of [Motta17].
    nbasis = h1e.shape[-1]
    nchol = chol.shape[-1]
    chol_view = chol.reshape((nbasis, nbasis * nchol))
    # assert chol_view.__array_interface__['data'][0] == chol.__array_interface__['data'][0]
    v0 = 0.5 * numpy.dot(
        chol_view,
        chol_view.T.conj(),  # conjugate added to account for complex integrals
    )  # einsum('ikn,jkn->ij', chol_3, chol_3, optimize=True)
    h1e_mod[0, :, :] = h1e[0] - v0
    h1e_mod[1, :, :] = h1e[1] - v0

def construct_h1e_mod_cmplx(A, B, h1e, h1e_mod):
    # Subtract one-body bit following reordering of 2-body operators.
    # Eqn (17) of [Motta17].
    nbasis = h1e.shape[-1]
    nchol = A.shape[-1]
    A_view = A.reshape((nbasis, nbasis * nchol))
    AT_view = A.transpose(1, 0, 2).reshape((nbasis, nbasis * nchol))
    B_view = B.reshape((nbasis, nbasis * nchol))
    BT_view = B.transpose(1, 0, 2).reshape((nbasis, nbasis * nchol))
    # assert chol_view.__array_interface__['data'][0] == chol.__array_interface__['data'][0]
    v0 = 0.5 * (numpy.dot(A_view, AT_view.T) + numpy.dot(B_view, BT_view.T))
    h1e_mod[0, :, :] = h1e[0] - v0
    h1e_mod[1, :, :] = h1e[1] - v0



def get_sparse_AT_prime_from_A_prime(A_prime_csc):
    '''
    Function to directly obtain the reshaped transposed 3d tensor from the reshaped tensor, i.e. assume A is a 3d tensor with shape (n,n,x), AT is the transpose of A in the first 2 indices, AT = A.transpose(1,0,2). Both A an AT are reshaped into arrays A' and AT' with shape (n**2, x), respectively. 
    The function provides a way to directly calculate AT' from A' without reshaping back to 3d, since it is impossible to do that for sparse matrices.
    '''
    # Get the shape (a^2, b) and deduce a
    a_squared, b = A_prime_csc.shape
    a = int(math.sqrt(a_squared))
    
    rows_AT_prime = []
    cols_AT_prime = []
    data_AT_prime = []
    
    # Iterate over all non-zero elements of A'
    cx = A_prime_csc.tocoo()
    for i, j, v in zip(cx.row, cx.col, cx.data):
        # Calculate the original (i, j, k) indices in the 3D array A
        original_i = i // a
        original_j = i % a
        original_k = j
        
        # Calculate the new row index after the transpose of A
        transposed_row = original_j * a + original_i
        
        # Append the new position in the transposed matrix
        rows_AT_prime.append(transposed_row)
        cols_AT_prime.append(original_k)
        data_AT_prime.append(v)
    
    # Create a new CSC matrix for AT' from the new rows, cols, and data
    AT_prime_csc = scipy.sparse.csc_matrix((data_AT_prime, (rows_AT_prime, cols_AT_prime)), shape=(a_squared, b))
    
    return AT_prime_csc

def construct_h1e_mod_trsUEG(A, B, h1e, h1e_mod):
    '''
    Subtract one-body part following reordering of 2-body operators for non-Hermitian A&B matrices. 
    The one body part can be written as .5 * \sum_{k1, q} [A0_{k, k1}(q)A0_{k1, k'}(q) + B0_{k, k1}(q)B0_{k1, k'}(q)]
    '''
    nbasis = h1e.shape[-1]
    nchol0 = A.shape[-1] // 3
    A0 = A[:,:nchol0]
    B0 = B[:,:nchol0]
    A0_view = A0.reshape((nbasis, nbasis * nchol0))
    B0_view = B0.reshape((nbasis, nbasis * nchol0))
    A0_viewT = get_sparse_AT_prime_from_A_prime(A0).reshape((nbasis, nbasis * nchol0))
    B0_viewT = get_sparse_AT_prime_from_A_prime(B0).reshape((nbasis, nbasis * nchol0))
    modH = .5 * (A0_view.dot(A0_viewT.T) + B0_view.dot(B0_viewT.T))
    h1e_mod[0, :, :] = h1e[0] - modH
    h1e_mod[1, :, :] = h1e[1] - modH

def construct_h1e_mod_nonHermitian(A, B, h1e, h1e_mod):
    '''
    Subtract one-body part following reordering of 2-body operators for non-Hermitian A&B matrices. 
    The one body part can be written as .5 * \sum_{k1, q} [A0_{k, k1}(q)A0_{k1, k'}(q) + B0_{k, k1}(q)B0_{k1, k'}(q)]
    '''
    nbasis = h1e.shape[-1]
    nchol0 = A.shape[-1]
    A_view = A.reshape((nbasis, nbasis * nchol0))
    B_view = B.reshape((nbasis, nbasis * nchol0))
    A_viewT = get_sparse_AT_prime_from_A_prime(A).reshape((nbasis, nbasis * nchol0))
    B_viewT = get_sparse_AT_prime_from_A_prime(B).reshape((nbasis, nbasis * nchol0))
    modH = .5 * (A_view.dot(A_viewT.T) + B_view.dot(B_viewT.T))
    h1e_mod[0, :, :] = h1e[0] - modH
    h1e_mod[1, :, :] = h1e[1] - modH

class SparseRealChol(GenericBase):
    """Similar to GenericRealChol but stores sparse arrays.
    """

    def __init__(self, h1e, chol, ecore=0.0, verbose=False):
        assert (
            h1e.shape[0] == 2
        )  # assuming each spin component is given. this should be fixed for GHF...?
        super().__init__(h1e, ecore, verbose)

        assert chol.dtype == numpy.dtype("float64")

        self.chol = scipy.sparse.csc_array(chol)  # [M^2, nchol]
        self.nchol = self.chol.shape[-1]
        self.nfields = self.nchol
        assert self.nbasis**2 == chol.shape[0]

        chol = chol.reshape((self.nbasis, self.nbasis, self.nchol))
        self.sym_idx = numpy.triu_indices(self.nbasis)
        cp_shape = (self.nbasis * (self.nbasis + 1) // 2, chol.shape[-1])
        self.chol_packed = numpy.zeros(cp_shape, dtype=chol.dtype)
        pack_cholesky(self.sym_idx[0], self.sym_idx[1], self.chol_packed, chol)
        self.chol_packed = scipy.sparse.csc_array(self.chol_packed)
        #chol = chol.reshape((self.nbasis * self.nbasis, self.nchol))

        self.chunked = False

        # this is the one-body part that comes out of re-ordering the 2-body operators
        h1e_mod = numpy.zeros(self.H1.shape, dtype=self.H1.dtype)
        construct_h1e_mod(chol, self.H1, h1e_mod)
        self.h1e_mod = h1e_mod

        if verbose:
            # TODO: Is this way to approximate mem usage correct?
            #mem = (self.chol.data.nbytes + self.chol.indptr.nbytes + self.chol.indices.nbytes) / (1024.0**3)
            #mem_packed = (self.chol_packed.data.nbytes + self.chol_packed.indptr.nbytes + self.chol_packed.indices.nbytes) \
            #                / (1024.0**3)
            mem = self.chol.data.nbytes / (1024.0**3)
            mem_packed = self.chol_packed.data.nbytes / (1024.0**3)
            print("# Number of orbitals: %d" % self.nbasis)
            print(f"# Approximate memory required by Cholesky vectors {mem:f} GB")
            print(f"# Approximate memory required by packed Cholesky vectors {mem_packed:f} GB")
            print(f"# Approximate memory required total {mem_packed + mem:f} GB")
            print("# Number of Cholesky vectors: %d" % (self.nchol))
            print("# Number of fields: %d" % (self.nchol))
            print("# Finished setting up SparseRealChol object.")

    def hijkl(self, i, j, k, l):  # (ik|jl) somehow physicist notation - terrible!!
        ik = i * self.nbasis + k
        jl = j * self.nbasis + l
        return numpy.dot(self.chol[ik], self.chol[jl])


class SparseComplexChol(GenericBase):
    """Similar to GenericComplexChol but stores sparse arrays.
    """

    def __init__(self, h1e, chol, ecore=0.0, verbose=False):
        assert h1e.shape[0] == 2
        super().__init__(h1e, ecore, verbose)

        self.chol = scipy.sparse.csc_array(chol, dtype=numpy.complex128)  # [M^2, nchol]
        self.nchol = self.chol.shape[-1]
        self.nfields = self.nchol * 2
        assert self.nbasis**2 == chol.shape[0]

        self.chunked = False

        # This is the one-body part that comes out of re-ordering the 2-body operators
        h1e_mod = numpy.zeros(self.H1.shape, dtype=self.H1.dtype)
        construct_h1e_mod(chol, self.H1, h1e_mod) # Pass in dense matrices.
        self.h1e_mod = h1e_mod

        # We need to store A and B integrals. Working with full arrays first.
        chol = chol.reshape((self.nbasis, self.nbasis, self.nchol))
        A = numpy.zeros(chol.shape, dtype=chol.dtype)
        B = numpy.zeros(chol.shape, dtype=chol.dtype)

        for x in range(self.nchol):
            A[:, :, x] = chol[:, :, x] + chol[:, :, x].T.conj()
            B[:, :, x] = 1.0j * (chol[:, :, x] - chol[:, :, x].T.conj())
        A /= 2.0
        B /= 2.0

        #chol = self.chol.reshape((self.nbasis * self.nbasis, self.nchol))
        self.A = scipy.sparse.csc_array(A.reshape((self.nbasis * self.nbasis, self.nchol)))
        self.B = scipy.sparse.csc_array(B.reshape((self.nbasis * self.nbasis, self.nchol)))

        if verbose:
            # TODO: Is this way to approximate mem usage correct?
            #mem = (self.A.data.nbytes + self.A.indptr.nbytes + self.A.indices.nbytes) / (1024.0**3) * 3
            mem = self.A.data.nbytes / (1024.0**3) * 3
            print("# Number of orbitals: %d" % self.nbasis)
            print(f"# Approximate memory required by Cholesky + A&B vectors {mem:f} GB")
            print("# Number of Cholesky vectors: %d" % (self.nchol))
            print("# Number of fields: %d" % (self.nchol * 2))
            print("# Finished setting up SparseComplexChol object.")

    def hijkl(self, i, j, k, l):  # (ik|jl) somehow physicist notation - terrible!!
        ik = i * self.nbasis + k
        lj = l * self.nbasis + j
        chol_ik = 0.5 * (self.A[ik] + self.B[ik] / 1.0j)
        chol_lj = 0.5 * (self.A[lj] + self.B[lj] / 1.0j)
        return numpy.dot(chol_ik, chol_lj.conj())

class SparseNonHermitian(GenericBase):
    """
    Designed for transcorrelated Hamiltonian since the A and B tensors for H_TC is not Hermitian, thus there is no suitable chol_vec that can be defined as A[:,:,x] = .5 * (chol[:,:,x] + chol[:,:,x].T.conj()). Here we directly manipulate the A and B tensors.
    The A and B tensors are stored as sparse CSC matrix, with shape (nbasis * nbasis, nchol). 
    """
    def __init__(self, h1e, A, B, ecore=0.0, verbose=False):
        assert h1e.shape[0] == 2
        super().__init__(h1e, ecore, verbose)
        assert A is not None and B is not None
        self.A = scipy.sparse.csc_matrix(A, dtype = numpy.complex128)
        self.B = scipy.sparse.csc_matrix(B, dtype = numpy.complex128)
        self.nchol = self.A.shape[-1]
        self.nfields = self.nchol * 2
        assert self.nbasis**2 == A.shape[0] and self.nbasis**2 == B.shape[0]

        self.chunked = False

        # This is the one-body part that comes out of re-ordering the 2-body operators
        h1e_mod = numpy.zeros(self.H1.shape, dtype=self.H1.dtype)
        construct_h1e_mod_nonHermitian(self.A, self.B, self.H1, h1e_mod) # Pass in dense matrices.
        self.h1e_mod = h1e_mod

        if verbose:
            # TODO: Is this way to approximate mem usage correct?
            #mem = (self.A.data.nbytes + self.A.indptr.nbytes + self.A.indices.nbytes) / (1024.0**3) * 3
            mem = self.A.data.nbytes / (1024.0**3) * 3
            print("# Number of orbitals: %d" % self.nbasis)
            print(f"# Approximate memory required by Cholesky + A&B vectors {mem:f} GB")
            print("# Number of Cholesky vectors: %d" % (self.nchol))
            print("# Number of fields: %d" % (self.nchol * 2))
            print("# Finished setting up SparseComplexChol object for non-Hermitian A and B matrices.")
            
    def hijkl(self, i, j, k, l):  # (ik|jl) somehow physicist notation - terrible!!
        ik = i * self.nbasis + k
        jl = j * self.nbasis + l
        hijkl = (self.A[ik].dot(self.A[jl]) + self.B[ik].dot(self.B[jl]))
        return hijkl

def Sparse(h1e, chol, ecore=0.0, A=None, B=None, verbose=False):
    if chol is None:
        return SparseNonHermitian(h1e, A, B, ecore, verbose)
    else:
        if chol.dtype == numpy.dtype("complex128"):
            return SparseComplexChol(h1e, chol, ecore, verbose)
        elif chol.dtype == numpy.dtype("float64"):
            return SparseRealChol(h1e, chol, ecore, verbose)


def read_integrals(integral_file):
    try:
        (h1e, schol_vecs, ecore, _, _, _) = from_qmcpack_sparse(integral_file)
        chol_vecs = schol_vecs.toarray()
        return h1e, chol_vecs, ecore
    except KeyError:
        pass
    try:
        (h1e, chol_vecs, ecore, _, _, _) = from_qmcpack_dense(integral_file)
        return h1e, chol_vecs, ecore
    except KeyError:
        pass
    try:
        (h1e, chol_vecs, ecore) = read_hamiltonian(integral_file)
        naux = chol_vecs.shape[0]
        nbsf = chol_vecs.shape[-1]
        return h1e, chol_vecs.T.reshape((nbsf, nbsf, naux)), ecore
    except KeyError:
        return None
