# Copyright 2026 The ipie Developers. All Rights Reserved.
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
# Author: Jinghong Zhang <jinghongzhang@fas.harvard.edu>
#
"""Test 1: thermofield Gaussian algebra against brute-force Fock space."""

import numpy
import pytest

from ipie.addons.thermal.thermofield.gaussian import (
    factored_pair_greens_function,
    factored_pair_log_overlap,
    thermofield_greens_function,
    thermofield_log_overlap,
    thermofield_one_rdm,
    thermofield_overlap,
    thermofield_wick_normal_ordered_square,
)
from ipie.addons.thermal.thermofield.tests.ed_utils import (
    creation_operators,
    thermofield_state_vector,
)


@pytest.mark.unit
@pytest.mark.parametrize("M", [2, 3])
def test_overlap_and_greens_function_vs_fock_space(M):
    rng = numpy.random.default_rng(7 + M)
    Lmat = rng.standard_normal((M, M)) + 1j * rng.standard_normal((M, M))
    Delta = rng.standard_normal((M, M)) + 1j * rng.standard_normal((M, M))
    Lambda = Lmat.conj().T

    # Brute-force states in the doubled Fock space.
    vL = thermofield_state_vector(Lmat)
    vR = thermofield_state_vector(Delta)
    S_brute = vL.conj() @ vR

    S = thermofield_overlap(Lambda, Delta)
    assert S == pytest.approx(S_brute, abs=1e-11)
    log_S = thermofield_log_overlap(Lambda, Delta)
    assert numpy.exp(log_S) == pytest.approx(S_brute, abs=1e-11)

    # One-body density: <c_i^dagger c_j> / <1> = G[j, i].
    cre = creation_operators(2 * M)
    ann = [c.conj().T for c in cre]
    G = thermofield_greens_function(Lambda, Delta)
    P = thermofield_one_rdm(Lambda, Delta)
    for i in range(M):
        for j in range(M):
            brute = (vL.conj() @ (cre[i] @ ann[j] @ vR)) / S_brute
            assert G[j, i] == pytest.approx(brute, abs=1e-11)
            assert P[i, j] == pytest.approx(brute, abs=1e-11)

    # Tr(A G) = <c^dagger A c> for a random A.
    A = rng.standard_normal((M, M)) + 1j * rng.standard_normal((M, M))
    opA = sum(A[i, j] * (cre[i] @ ann[j]) for i in range(M) for j in range(M))
    assert numpy.trace(A @ G) == pytest.approx((vL.conj() @ (opA @ vR)) / S_brute, abs=1e-11)


@pytest.mark.unit
@pytest.mark.parametrize("M", [2, 3])
def test_wick_normal_ordered_square_vs_fock_space(M):
    rng = numpy.random.default_rng(17 + M)
    Lmat = rng.standard_normal((M, M)) + 1j * rng.standard_normal((M, M))
    Delta = rng.standard_normal((M, M)) + 1j * rng.standard_normal((M, M))
    Lambda = Lmat.conj().T
    L = rng.standard_normal((M, M)) + 1j * rng.standard_normal((M, M))

    vL = thermofield_state_vector(Lmat)
    vR = thermofield_state_vector(Delta)
    S_brute = vL.conj() @ vR

    # : c_i^dag c_k c_j^dag c_l : = c_i^dag c_j^dag c_l c_k.
    cre = creation_operators(2 * M)
    ann = [c.conj().T for c in cre]
    op = numpy.zeros_like(cre[0], dtype=numpy.complex128)
    for i in range(M):
        for k in range(M):
            for j in range(M):
                for l in range(M):
                    op += L[i, k] * L[j, l] * (cre[i] @ cre[j] @ ann[l] @ ann[k])
    brute = (vL.conj() @ (op @ vR)) / S_brute

    G = thermofield_greens_function(Lambda, Delta)
    wick = thermofield_wick_normal_ordered_square(L, G)
    assert wick == pytest.approx(brute, abs=1e-10)


@pytest.mark.unit
def test_factored_pair_formulas_match_plain():
    """QDT-factored overlap/Green's function agree with the plain formulas."""
    M = 4
    rng = numpy.random.default_rng(9)

    def random_qdt():
        Q = rng.standard_normal((M, M)) + 1j * rng.standard_normal((M, M))
        d = rng.uniform(-2.0, 2.0, M)
        T = rng.standard_normal((M, M)) + 1j * rng.standard_normal((M, M))
        Delta = Q @ (numpy.exp(d)[:, None] * T)
        return Q, d, T, Delta

    QL, dL, TL, Delta_L = random_qdt()
    QR, dR, TR, Delta_R = random_qdt()
    Lambda = Delta_L.conj().T

    log_S = factored_pair_log_overlap(QL, dL, TL, QR, dR, TR)
    assert numpy.exp(log_S) == pytest.approx(thermofield_overlap(Lambda, Delta_R), rel=1e-11)
    G = factored_pair_greens_function(QL, dL, TL, QR, dR, TR)
    numpy.testing.assert_allclose(G, thermofield_greens_function(Lambda, Delta_R), atol=1e-11)


@pytest.mark.unit
def test_log_scale_branches_consistent():
    """Scaled overlap/Green's function agree with direct evaluation."""
    M = 4
    rng = numpy.random.default_rng(3)
    Lambda = rng.standard_normal((M, M)) + 1j * rng.standard_normal((M, M))
    Delta = rng.standard_normal((M, M)) + 1j * rng.standard_normal((M, M))
    for log_scale in (-1.5, 0.0, 2.5):
        scale = numpy.exp(log_scale)
        ref_S = thermofield_overlap(Lambda, scale * Delta)
        ref_G = thermofield_greens_function(Lambda, scale * Delta)
        assert numpy.exp(
            thermofield_log_overlap(Lambda, Delta, log_scale=log_scale)
        ) == pytest.approx(ref_S, rel=1e-12)
        numpy.testing.assert_allclose(
            thermofield_greens_function(Lambda, Delta, log_scale=log_scale), ref_G, atol=1e-12
        )
