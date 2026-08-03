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
# Authors: Jinghong Zhang <jinghongzhang@fas.harvard.edu>
#
"""BLAS-friendly kernels replacing the hot numpy.einsum contractions.

Follows the core-ipie kernel conventions (see e.g. propagation/
phaseless_generic.py construct_VHS and estimators/local_energy_sd.py):
flatten batched contractions to single 2D gemms, and split complex operands
into real/imag parts so real integral tensors stay in dgemm instead of being
promoted to complex.
"""

import numpy as np


def rmatmul(x, mat):
    """x @ mat with real/imag splitting when mat is real and x is complex.

    x: (..., k) 2D array (real or complex); mat: (k, n) real or complex.
    """
    if np.iscomplexobj(mat) or not np.iscomplexobj(x):
        return x @ mat
    return np.ascontiguousarray(x.real) @ mat + 1j * (np.ascontiguousarray(x.imag) @ mat)


def left_apply(mat, states):
    """einsum('pq,wqr->wpr', mat, states) as one flattened gemm.

    mat: (m, nq) real or complex; states: (nw, nq, nr) complex.
    """
    nw, nq, nr = states.shape
    flat = states.transpose(1, 0, 2).reshape(nq, nw * nr)  # copy (non-contiguous view)
    if np.iscomplexobj(flat) and not np.iscomplexobj(mat):
        out = mat @ np.ascontiguousarray(flat.real) + 1j * (
            mat @ np.ascontiguousarray(flat.imag)
        )
    else:
        out = mat @ flat
    return out.reshape(mat.shape[0], nw, nr).transpose(1, 0, 2)


def flatten_ghalf(Ghalf):
    """(nw, nao, nocc) -> contiguous (nw, nocc*nao) in (occ, ao) index order,
    matching rchol.reshape(nchol, nocc*nao)."""
    nw = Ghalf.shape[0]
    return Ghalf.transpose(0, 2, 1).reshape(nw, -1)  # copy
