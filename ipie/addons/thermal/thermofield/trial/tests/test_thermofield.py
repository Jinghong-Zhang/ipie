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
import scipy.linalg

from ipie.addons.thermal.thermofield.trial.thermofield import ThermofieldThermalTrial
from ipie.addons.thermal.thermofield.utils.testing import build_noninteracting_hamiltonian


def random_complex(shape):
    return numpy.random.standard_normal(shape) + 1j * numpy.random.standard_normal(shape)


@pytest.mark.unit
def test_beta_zero_identity_purification():
    # System params.
    nbasis = 4

    # Thermofield AFQMC params.
    mu = 0.0
    beta = 0.0
    seed = 11
    numpy.random.seed(seed)

    h1e = numpy.random.standard_normal((nbasis, nbasis))
    h1e = 0.5 * (h1e + h1e.T)
    hamiltonian = build_noninteracting_hamiltonian(h1e)
    trial = ThermofieldThermalTrial(hamiltonian, beta=beta, mu=mu)

    # At beta = 0 the guide is D_T = I and G_T(I) = I / 2.
    eye = numpy.eye(nbasis)
    numpy.testing.assert_allclose(trial.dmat, numpy.array([eye, eye]), atol=1e-12)
    numpy.testing.assert_allclose(trial.P, numpy.array([0.5 * eye, 0.5 * eye]), atol=1e-12)
    assert trial.nav == pytest.approx(nbasis, abs=1e-12)

    Delta = numpy.array([eye, eye], dtype=numpy.complex128)
    log_ovlp = trial.calc_log_overlap(Delta)
    assert log_ovlp == pytest.approx(2 * nbasis * numpy.log(2.0), abs=1e-12)
    G = trial.calc_greens_function(Delta)
    numpy.testing.assert_allclose(G, numpy.array([0.5 * eye, 0.5 * eye]), atol=1e-12)


@pytest.mark.unit
def test_projector_limit_of_guide_overlap():
    # System params.
    eps = numpy.array([-1.0, -0.7, 0.4, 1.3])
    nbasis = len(eps)
    nocc = 2

    # Thermofield AFQMC params.
    mu = 0.0
    seed = 23
    numpy.random.seed(seed)

    # As beta grows, S_T(B) reduces to the zero-temperature trial overlap. With
    # k_T = diag(eps) and the occupied levels eps < 0,
    # S_T(B) / prod_occ exp(-beta eps_i / 2) -> det(C_occ^dagger B C_occ) and
    # G_T(B) -> B C_occ (C_occ^dagger B C_occ)^{-1} C_occ^dagger,
    # which det(I + B) does not reproduce.
    hamiltonian = build_noninteracting_hamiltonian(numpy.diag(eps))
    k_trial = numpy.array([numpy.diag(eps)] * 2)
    C_occ = numpy.eye(nbasis)[:, :nocc]
    B = random_complex((nbasis, nbasis))
    B += 2.0 * numpy.eye(nbasis)  # Keep B well conditioned.
    Delta = numpy.array([B, B])

    log_S_ZT = numpy.log(numpy.linalg.det(C_occ.T @ B @ C_occ))
    G_ZT = B @ C_occ @ numpy.linalg.inv(C_occ.T @ B @ C_occ) @ C_occ.T

    # The limit is approached as exp(-beta eps_min / 2) with eps_min = 0.4, so
    # machine precision needs beta of about 160.
    errors_S = []
    errors_G = []
    for beta in (40.0, 80.0, 160.0):
        trial = ThermofieldThermalTrial(hamiltonian, beta=beta, mu=mu, k_trial=k_trial)
        # Both spins: log S_T - 2 sum_occ (-beta eps_i / 2) -> 2 log S_ZT. The
        # phase of a complex log is only defined modulo 2 pi, so compare exponentials.
        log_norm = trial.calc_log_overlap(Delta) + beta * numpy.sum(eps[:nocc])
        diff = numpy.exp(log_norm - 2.0 * log_S_ZT) - 1.0
        errors_S.append(abs(diff))

        G = trial.calc_greens_function(Delta)
        errors_G.append(numpy.max(numpy.abs(G[0] - G_ZT)))

    # The errors decrease with beta and reach machine precision at beta = 160.
    assert errors_S[0] > errors_S[1] > errors_S[2]
    assert errors_G[0] > errors_G[1] > errors_G[2]
    assert errors_S[2] < 1e-12
    assert errors_G[2] < 1e-12

    # The closed-trace determinant is not the limiting guide overlap.
    det_trace = numpy.linalg.det(numpy.eye(nbasis) + B)
    assert abs(det_trace - numpy.exp(log_S_ZT)) > 1.0


@pytest.mark.unit
def test_guide_matches_thermal_occupations():
    # System params.
    nbasis = 5

    # Thermofield AFQMC params.
    mu = 0.3
    beta = 1.7
    seed = 5
    numpy.random.seed(seed)

    h1e = numpy.random.standard_normal((nbasis, nbasis))
    h1e = 0.5 * (h1e + h1e.T)
    hamiltonian = build_noninteracting_hamiltonian(h1e)
    trial = ThermofieldThermalTrial(hamiltonian, beta=beta, mu=mu)
    assert trial.guide_time == pytest.approx(0.5 * beta)

    # The guide density matrix P is the Fermi function of k_T and G its hole form.
    eye = numpy.eye(nbasis)
    f = numpy.linalg.inv(eye + scipy.linalg.expm(beta * (h1e - mu * eye)))
    numpy.testing.assert_allclose(trial.P[0], f, atol=1e-12)
    numpy.testing.assert_allclose(trial.G[0], eye - f.T, atol=1e-12)

    # The mixed Green's function at Delta = D_T is the thermal occupation.
    G = trial.calc_greens_function(numpy.array(trial.dmat, dtype=numpy.complex128))
    numpy.testing.assert_allclose(G[0], f, atol=1e-11)


@pytest.mark.unit
def test_explicit_guide_time_and_trace_endpoint():
    # System params.
    eps = numpy.array([-0.8, 0.2, 1.1])
    nbasis = len(eps)

    # Thermofield AFQMC params.
    beta = 2.4
    guide_time = beta
    nwalkers = 4
    seed = 41
    numpy.random.seed(seed)

    # The fixed guide time is independent of beta, and zero gives the trace.
    hamiltonian = build_noninteracting_hamiltonian(numpy.diag(eps))
    k_trial = numpy.array([numpy.diag(eps)] * 2)
    trial = ThermofieldThermalTrial(hamiltonian, beta=beta, k_trial=k_trial, guide_time=guide_time)

    eye = numpy.eye(nbasis)
    Lambda = numpy.diag(numpy.exp(-guide_time * eps))
    numpy.testing.assert_allclose(trial.dmat, numpy.array([Lambda, Lambda]), atol=1e-12)

    Delta = numpy.array(
        [
            eye + 0.1 * numpy.random.standard_normal((nbasis, nbasis)),
            eye + 0.1 * numpy.random.standard_normal((nbasis, nbasis)),
        ],
        dtype=numpy.complex128,
    )
    log_ovlp, G = trial.calc_log_overlap_and_greens_function(Delta)
    overlap_ref = numpy.prod([numpy.linalg.det(eye + Lambda @ Delta[s]) for s in range(2)])
    numpy.testing.assert_allclose(numpy.exp(log_ovlp), overlap_ref, atol=1e-11)
    for s in range(2):
        G_ref = Delta[s] @ numpy.linalg.inv(eye + Lambda @ Delta[s]) @ Lambda
        numpy.testing.assert_allclose(G[s], G_ref, atol=1e-11)

    log_trace, G_trace = trial.calc_log_overlap_and_greens_function(Delta, guide_time=0.0)
    trace_ref = numpy.prod([numpy.linalg.det(eye + Delta[s]) for s in range(2)])
    numpy.testing.assert_allclose(numpy.exp(log_trace), trace_ref, atol=1e-11)
    for s in range(2):
        G_ref = Delta[s] @ numpy.linalg.inv(eye + Delta[s])
        numpy.testing.assert_allclose(G_trace[s], G_ref, atol=1e-11)


@pytest.mark.unit
def test_batched_overlap_and_greens_function_matches_scalar():
    # System params.
    nbasis = 6

    # Thermofield AFQMC params.
    beta = 3.1
    nwalkers = 5
    seed = 77

    # The walker-batched entry point reproduces the per-walker path bit for bit
    # for every guide time, and flags singular walkers with non-finite output
    # instead of raising.
    numpy.random.seed(seed)
    k = random_complex((nbasis, nbasis))
    k = 0.5 * (k + k.conj().T)
    hamiltonian = build_noninteracting_hamiltonian(k.real)
    trial = ThermofieldThermalTrial(hamiltonian, beta=beta, k_trial=numpy.array([k, k]))

    shape = (nwalkers, 2, nbasis, nbasis)
    Qmat = random_complex(shape)
    Tmat = random_complex(shape)
    log_d = numpy.random.uniform(-40.0, 40.0, size=(nwalkers, 2, nbasis))

    for guide_time in (0.0, beta):
        lo_b, G_b = trial.calc_log_overlap_and_greens_function_batched(
            Qmat, log_d, Tmat, guide_time=guide_time
        )
        assert lo_b.shape == (nwalkers,) and G_b.shape == (nwalkers, 2, nbasis, nbasis)
        for iw in range(nwalkers):
            lo, G = trial.calc_log_overlap_and_greens_function(
                Qmat[iw], log_d[iw], Tmat[iw], guide_time=guide_time
            )
            assert lo_b[iw] == lo
            assert numpy.array_equal(G_b[iw], G)

    # Default guide time, as used for the walker cache.
    lo_b, G_b = trial.calc_log_overlap_and_greens_function_batched(Qmat, log_d, Tmat)
    for iw in range(nwalkers):
        lo, G = trial.calc_log_overlap_and_greens_function(Qmat[iw], log_d[iw], Tmat[iw])
        assert lo_b[iw] == lo
        assert numpy.array_equal(G_b[iw], G)

    # A pathological walker must not raise: it is flagged non-finite and the
    # others are unchanged. The one-sided kernel inverts no walker factor and
    # gives the finite det(I + 0) = 1 for Q = 0, so it gets a non-finite Q entry.
    bad = 2
    Qs, Ts = Qmat.copy(), Tmat.copy()
    Qs[bad, 0, 0, 0] = numpy.nan
    lo_s, G_s = trial.calc_log_overlap_and_greens_function_batched(Qs, log_d, Ts)
    assert not (numpy.isfinite(lo_s[bad]) and numpy.all(numpy.isfinite(G_s[bad])))
    for iw in range(nwalkers):
        if iw == bad:
            continue
        assert lo_s[iw] == lo_b[iw]
        assert numpy.array_equal(G_s[iw], G_b[iw])


@pytest.mark.unit
def test_negative_guide_time_rejected():
    hamiltonian = build_noninteracting_hamiltonian(numpy.eye(2))
    with pytest.raises(ValueError, match="guide_time"):
        ThermofieldThermalTrial(hamiltonian, beta=1.0, guide_time=-0.1)


if __name__ == "__main__":
    test_beta_zero_identity_purification()
    test_projector_limit_of_guide_overlap()
    test_guide_matches_thermal_occupations()
    test_explicit_guide_time_and_trace_endpoint()
    test_batched_overlap_and_greens_function_matches_scalar()
    test_negative_guide_time_rejected()
