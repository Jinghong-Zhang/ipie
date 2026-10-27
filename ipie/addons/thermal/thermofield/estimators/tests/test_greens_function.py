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

from ipie.addons.thermal.thermofield.estimators.greens_function import (
    thermofield_greens_function,
    thermofield_log_overlap,
    thermofield_one_rdm,
    thermofield_overlap,
    thermofield_wick_normal_ordered_square,
)
from ipie.addons.thermal.thermofield.utils.testing import (
    creation_operators,
    thermofield_state_vector,
)


def random_complex(shape):
    return numpy.random.standard_normal(shape) + 1j * numpy.random.standard_normal(shape)


@pytest.mark.unit
def test_overlap_and_greens_function_vs_fock_space():
    for nbasis in (2, 3):
        numpy.random.seed(7 + nbasis)
        shape = (nbasis, nbasis)
        Lmat = random_complex(shape)
        Delta = random_complex(shape)
        Lambda = Lmat.conj().T

        # Brute-force states in the doubled Fock space.
        vL = thermofield_state_vector(Lmat)
        vR = thermofield_state_vector(Delta)
        S_brute = vL.conj() @ vR

        S = thermofield_overlap(Lambda, Delta)
        assert S == pytest.approx(S_brute, abs=1e-11)
        log_S = thermofield_log_overlap(Lambda, Delta)
        assert numpy.exp(log_S) == pytest.approx(S_brute, abs=1e-11)

        # One-body density matrix: <c_i^dagger c_j> / <1> = G[j, i].
        cre = creation_operators(2 * nbasis)
        ann = [c.conj().T for c in cre]
        G = thermofield_greens_function(Lambda, Delta)
        P = thermofield_one_rdm(Lambda, Delta)
        for i in range(nbasis):
            for j in range(nbasis):
                brute = (vL.conj() @ (cre[i] @ ann[j] @ vR)) / S_brute
                assert G[j, i] == pytest.approx(brute, abs=1e-11)
                assert P[i, j] == pytest.approx(brute, abs=1e-11)

        # Tr(A G) = <c^dagger A c> for a random A.
        A = random_complex(shape)
        opA = sum(A[i, j] * (cre[i] @ ann[j]) for i in range(nbasis) for j in range(nbasis))
        brute = (vL.conj() @ (opA @ vR)) / S_brute
        assert numpy.trace(A @ G) == pytest.approx(brute, abs=1e-11)


@pytest.mark.unit
def test_wick_normal_ordered_square_vs_fock_space():
    for nbasis in (2, 3):
        numpy.random.seed(17 + nbasis)
        shape = (nbasis, nbasis)
        Lmat = random_complex(shape)
        Delta = random_complex(shape)
        Lambda = Lmat.conj().T
        L = random_complex(shape)

        vL = thermofield_state_vector(Lmat)
        vR = thermofield_state_vector(Delta)
        S_brute = vL.conj() @ vR

        # Normal ordering: :c_i^dagger c_k c_j^dagger c_l: = c_i^dagger c_j^dagger c_l c_k.
        cre = creation_operators(2 * nbasis)
        ann = [c.conj().T for c in cre]
        op = numpy.zeros_like(cre[0], dtype=numpy.complex128)
        for i in range(nbasis):
            for k in range(nbasis):
                for j in range(nbasis):
                    for l in range(nbasis):
                        op += L[i, k] * L[j, l] * (cre[i] @ cre[j] @ ann[l] @ ann[k])
        brute = (vL.conj() @ (op @ vR)) / S_brute

        G = thermofield_greens_function(Lambda, Delta)
        wick = thermofield_wick_normal_ordered_square(L, G)
        assert wick == pytest.approx(brute, abs=1e-10)


@pytest.mark.unit
def test_log_scale_branches_consistent():
    # System params.
    nbasis = 4
    seed = 3
    numpy.random.seed(seed)

    # The scaled overlap and Green's function agree with direct evaluation.
    shape = (nbasis, nbasis)
    Lambda = random_complex(shape)
    Delta = random_complex(shape)
    for log_scale in (-1.5, 0.0, 2.5):
        scale = numpy.exp(log_scale)
        ref_S = thermofield_overlap(Lambda, scale * Delta)
        ref_G = thermofield_greens_function(Lambda, scale * Delta)
        log_S = thermofield_log_overlap(Lambda, Delta, log_scale=log_scale)
        assert numpy.exp(log_S) == pytest.approx(ref_S, rel=1e-12)
        numpy.testing.assert_allclose(
            thermofield_greens_function(Lambda, Delta, log_scale=log_scale), ref_G, atol=1e-12
        )


if __name__ == "__main__":
    test_overlap_and_greens_function_vs_fock_space()
    test_wick_normal_ordered_square_vs_fock_space()
    test_log_scale_branches_consistent()
