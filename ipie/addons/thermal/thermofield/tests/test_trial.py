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
"""Trial tests: beta = 0 identity limit and the low-temperature projector
limit of the guide overlap (the guide must reduce to the ground-state AFQMC
trial overlap, NOT the closed-trace det(I + B))."""

import numpy
import pytest

from ipie.addons.thermal.thermofield.tests.ed_utils import build_noninteracting_hamiltonian
from ipie.addons.thermal.thermofield.trial import ThermofieldThermalTrial


@pytest.mark.unit
def test_beta_zero_identity_purification():
    """Test 3a: at beta = 0 the guide is D_T = I and G_T(I) = I / 2."""
    M = 4
    rng = numpy.random.default_rng(11)
    h = rng.standard_normal((M, M))
    h = 0.5 * (h + h.T)
    hamiltonian = build_noninteracting_hamiltonian(h)
    trial = ThermofieldThermalTrial(hamiltonian, beta=0.0, mu=0.0)

    numpy.testing.assert_allclose(trial.dmat, numpy.array([numpy.eye(M)] * 2), atol=1e-12)
    numpy.testing.assert_allclose(trial.P, numpy.array([0.5 * numpy.eye(M)] * 2), atol=1e-12)
    assert trial.nav == pytest.approx(M, abs=1e-12)

    Delta = numpy.array([numpy.eye(M)] * 2, dtype=numpy.complex128)
    log_ovlp = trial.calc_log_overlap(Delta)
    assert log_ovlp == pytest.approx(2 * M * numpy.log(2.0), abs=1e-12)
    G = trial.calc_greens_function(Delta)
    numpy.testing.assert_allclose(G, numpy.array([0.5 * numpy.eye(M)] * 2), atol=1e-12)


@pytest.mark.unit
def test_projector_limit_of_guide_overlap():
    """Test 6: as beta -> inf, S_T(B) reduces to the zero-T trial overlap.

    With k_T = diag(eps), occupied eps < 0,

        S_T(B) / prod_occ exp(-beta eps_i / 2) -> det(C_occ^dag B C_occ),
        G_T(B) -> B C_occ (C_occ^dag B C_occ)^{-1} C_occ^dag,

    and det(I + B) does NOT reproduce the limit.
    """
    M = 4
    eps = numpy.array([-1.0, -0.7, 0.4, 1.3])
    nocc = 2
    hamiltonian = build_noninteracting_hamiltonian(numpy.diag(eps))
    C_occ = numpy.eye(M)[:, :nocc]

    rng = numpy.random.default_rng(23)
    B = rng.standard_normal((M, M)) + 1j * rng.standard_normal((M, M))
    B += 2.0 * numpy.eye(M)  # Keep B well conditioned.
    Delta = numpy.array([B, B])

    log_S_ZT = numpy.log(numpy.linalg.det(C_occ.T @ B @ C_occ))
    G_ZT = B @ C_occ @ numpy.linalg.inv(C_occ.T @ B @ C_occ) @ C_occ.T

    # The projector limit is approached as exp(-beta eps_min / 2) with
    # eps_min = 0.4 here, so machine precision needs beta ~ 160.
    errors_S = []
    errors_G = []
    for beta in (40.0, 80.0, 160.0):
        trial = ThermofieldThermalTrial(
            hamiltonian, beta=beta, mu=0.0, k_trial=numpy.array([numpy.diag(eps)] * 2)
        )
        # Both spins: log S_T - 2 sum_occ (-beta eps_i / 2) -> 2 log S_ZT.
        # (The complex log phase is only defined mod 2 pi, so compare exps.)
        log_norm = trial.calc_log_overlap(Delta) + beta * numpy.sum(eps[:nocc])
        diff = numpy.exp(log_norm - 2.0 * log_S_ZT) - 1.0
        errors_S.append(abs(diff))

        G = trial.calc_greens_function(Delta)
        errors_G.append(numpy.max(numpy.abs(G[0] - G_ZT)))

    # Errors decrease with beta and reach near machine precision at beta = 80.
    assert errors_S[0] > errors_S[1] > errors_S[2]
    assert errors_G[0] > errors_G[1] > errors_G[2]
    assert errors_S[2] < 1e-12
    assert errors_G[2] < 1e-12

    # The closed-trace determinant is NOT the limiting guide overlap.
    det_trace = numpy.linalg.det(numpy.eye(M) + B)
    assert abs(det_trace - numpy.exp(log_S_ZT)) > 1.0


@pytest.mark.unit
def test_guide_matches_thermal_occupations():
    """trial.P equals the Fermi function of k_T and trial.G is its hole form."""
    M = 5
    rng = numpy.random.default_rng(5)
    h = rng.standard_normal((M, M))
    h = 0.5 * (h + h.T)
    mu = 0.3
    beta = 1.7
    hamiltonian = build_noninteracting_hamiltonian(h)
    trial = ThermofieldThermalTrial(hamiltonian, beta=beta, mu=mu)

    import scipy.linalg

    f = numpy.linalg.inv(numpy.eye(M) + scipy.linalg.expm(beta * (h - mu * numpy.eye(M))))
    numpy.testing.assert_allclose(trial.P[0], f, atol=1e-12)
    numpy.testing.assert_allclose(trial.G[0], numpy.eye(M) - f.T, atol=1e-12)

    # The mixed Green's function at Delta = D_T is the thermal occupation.
    G = trial.calc_greens_function(numpy.array(trial.dmat, dtype=numpy.complex128))
    numpy.testing.assert_allclose(G[0], f, atol=1e-11)
