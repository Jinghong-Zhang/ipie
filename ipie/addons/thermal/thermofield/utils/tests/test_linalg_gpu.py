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

import numpy
import pytest

from ipie.addons.thermal.thermofield.utils.linalg_gpu import householder_qr

cupy = pytest.importorskip("cupy")


@pytest.mark.gpu
@pytest.mark.parametrize("n,span", [(6, 0.0), (16, 300.0), (24, 600.0), (64, 600.0)])
def test_householder_qr_graded_stack(n, span):
    numpy.random.seed(5)
    A = numpy.random.standard_normal((5, 2, n, n)) + 1j * numpy.random.standard_normal((5, 2, n, n))
    # Column scales as in Q e^{D - max D} of a stabilization at large beta.
    A *= numpy.exp(-numpy.linspace(0.0, span, n))
    # A zero column stays zero.
    A[0, 1, :, n // 2] = 0.0
    Q, R = (cupy.asnumpy(x) for x in householder_qr(cupy.asarray(A)))
    eye = numpy.broadcast_to(numpy.eye(n), Q.shape)
    numpy.testing.assert_allclose(Q.conj().swapaxes(-1, -2) @ Q, eye, atol=1e-13)
    assert numpy.all(numpy.tril(R, -1) == 0.0)
    # Every column is reproduced to its own scale.
    scale = numpy.maximum(numpy.abs(A).max(axis=-2), 1e-300)
    assert (numpy.abs(Q @ R - A).max(axis=-2) / scale).max() < 1e-13
    d = numpy.abs(numpy.diagonal(R, axis1=-2, axis2=-1))
    d_ref = numpy.abs(numpy.diagonal(numpy.linalg.qr(A)[1], axis1=-2, axis2=-1))
    assert d[0, 1, n // 2] == 0.0
    # The scales agree with LAPACK where the factorization is unique (full rank).
    d[0, 1] = d_ref[0, 1] = 1.0
    numpy.testing.assert_allclose(d, d_ref, rtol=1e-12)


if __name__ == "__main__":
    test_householder_qr_graded_stack(16, 300.0)
