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

# Exact-bridge dressed importance on a two-site model. Every conditional field
# probability, the path probability, the weight identity and the dressing cache
# are recomputed from explicit determinants of the dressed importance
# I = (1 - lambda) S + lambda / (2M) sum_{k, x'} c(x') S_bridge, without the
# rank-two formulas or Green's functions used by the propagator.

from itertools import product
from types import SimpleNamespace

import numpy
import pytest
from scipy.linalg import expm

from ipie.addons.thermal.thermofield.hamiltonians.hubbard import build_hubbard_hamiltonian
from ipie.addons.thermal.thermofield.propagation.hubbard_discrete import (
    ThermofieldHubbardCharge,
    ThermofieldHubbardDiscrete,
)
from ipie.addons.thermal.thermofield.qmc.hubbard_afqmc import propagate_population_cp
from ipie.addons.thermal.thermofield.trial.hubbard import ThermofieldCPTrial
from ipie.addons.thermal.thermofield.utils.testing import creation_operators
from ipie.addons.thermal.thermofield.walkers.pop_controller import ThermofieldPopController
from ipie.addons.thermal.thermofield.walkers.uhf_walkers import ThermofieldWalkers
from ipie.utils.mpi import MPIHandler


def build_test_case(hs, guide, lam, theta=1.0, U=8, dt=0.1, beta=0.3, nw=64, cache=True):
    # Open two-site chain at half filling with the noninteracting guide.
    ham, hop, _ = build_hubbard_hamiltonian(2, 1, U, 0, pbc_x=False, pbc_y=False)
    trial = ThermofieldCPTrial(ham, beta, k_trial=numpy.array([hop, hop]), guide_time=beta)
    mpi = MPIHandler()
    w = ThermofieldWalkers(trial, 2, nw, mpi)
    Prop = ThermofieldHubbardCharge if hs == "charge" else ThermofieldHubbardDiscrete
    prop = Prop(
        numpy.array([hop, hop]),
        U,
        dt,
        beta,
        record_fields=True,
        cache_qinv=cache,
        guide_mode=guide,
        fixed_guide_time=beta if guide == "fixed" else None,
        bridge_lambda=lam,
        bridge_theta=theta,
    )
    return ham, hop, trial, mpi, w, prop


def field_factors(prop, hs, x):
    # Per-spin diagonal factor of one HS field value and its scalar, including
    # the factor 1/2 of the identity.
    if hs == "charge":
        return numpy.exp(1j * prop.gamma * x) * numpy.ones(2), 0.5 * numpy.exp(
            0.5 * prop.dt * prop.U - 1j * prop.gamma * x
        )
    return numpy.exp(prop.gamma * x * numpy.array([1.0, -1.0])), 0.5


def bridge_factors(hs, theta, x):
    if hs == "charge":
        return numpy.exp(1j * theta * x) * numpy.ones(2), numpy.exp(-1j * theta * x)
    return numpy.exp(theta * x * numpy.array([1.0, -1.0])), 1.0


@pytest.mark.unit
@pytest.mark.parametrize("lam", [0.5, 1.0])
@pytest.mark.parametrize("cache", [True, False])
@pytest.mark.parametrize("guide", ["completion", "fixed"])
@pytest.mark.parametrize("hs", ["charge", "spin"])
def test_bridge_all_paths_against_explicit_importance(hs, guide, cache, lam):
    U, dt, nslices, theta, nsite = 8, 0.1, 3, 1.0, 2
    ham, hop, trial, mpi, w, p = build_test_case(hs, guide, lam, theta=theta, cache=cache)
    histories = numpy.array(list(product([1, -1], repeat=nsite * nslices))).reshape(
        64, nslices, nsite
    )
    w.field_hist = numpy.zeros_like(histories, dtype=float)
    p.init_walker_cache(w, trial)
    if cache:
        p.seed_qinv_cache(w)
    K = expm(-dt * hop / 2)

    def importance(closure, D, prev):
        # Dressed importance at the interaction cut for the diagonal factors D
        # (2 spins, sites) and the earlier product prev (2 spins, M, M).
        S = numpy.prod(
            [
                numpy.linalg.det(numpy.eye(nsite) + closure @ K @ numpy.diag(D[s]) @ K @ prev[s])
                for s in range(2)
            ]
        )
        bridged = 0.0
        for k in range(nsite):
            for xb in (1, -1):
                v, c = bridge_factors(hs, theta, xb)
                Dk = D.copy()
                Dk[:, k] *= v
                bridged += c * numpy.prod(
                    [
                        numpy.linalg.det(
                            numpy.eye(nsite) + closure @ K @ numpy.diag(Dk[s]) @ K @ prev[s]
                        )
                        for s in range(2)
                    ]
                )
        return S, (1 - lam) * S + lam * bridged / (2 * nsite)

    pdf = numpy.ones(64)
    # HS scalars of the fields already applied on each path.
    accumulated = numpy.ones(64, dtype=complex)
    expected_dress = numpy.zeros(64)
    prev = numpy.repeat(numpy.eye(nsite, dtype=complex)[None, None], 64, axis=0).repeat(2, axis=1)
    for step in range(nslices):
        closure = expm(-(trial.beta if guide == "fixed" else trial.beta - (step + 1) * dt) * hop)
        for iw, fields in enumerate(histories[:, step]):
            D = numpy.ones((2, nsite), dtype=complex)
            for site, selected in enumerate(fields):
                weights = []
                for x in (1, -1):
                    f, scal = field_factors(p, hs, x)
                    Dx = D.copy()
                    Dx[:, site] *= f
                    weights.append(scal * importance(closure, Dx, prev[iw])[1])
                weights = accumulated[iw] * numpy.array(weights)
                numpy.testing.assert_allclose(
                    weights.imag, 0, atol=1e-10 * numpy.abs(weights).max()
                )
                assert numpy.all(weights.real > 0)
                pick = 0 if selected == 1 else 1
                pdf[iw] *= weights[pick].real / weights.real.sum()
                D[:, site] *= field_factors(p, hs, selected)[0]
                accumulated[iw] *= field_factors(p, hs, selected)[1]
            if step < nslices - 1:
                S, I = importance(closure, D, prev[iw])
                expected_dress[iw] = numpy.log((I / S).real)
            for s in range(2):
                prev[iw, s] = K @ numpy.diag(D[s]) @ K @ prev[iw, s]
        uniforms = numpy.where(histories[:, step] == 1, 0.0, numpy.nextafter(1.0, 0.0))
        p._advance(w, trial, step, numpy.arange(64), uniforms)
        if step < nslices - 1:
            numpy.testing.assert_allclose(w.log_dress.real, expected_dress, atol=2e-11)
        if step == 1:
            w.stabilize()
            if cache:
                p.refresh_qinv_after_stabilize(w)
    assert p.kill_count == p.forbidden_field_count == p.invalid_weight_count == 0
    numpy.testing.assert_array_equal(w.field_hist, histories)
    # The dressing is removed after the last slice.
    numpy.testing.assert_allclose(w.log_dress, 0, atol=0)
    numpy.testing.assert_allclose(pdf.sum(), 1, atol=2e-13)
    trace, green = trial.calc_log_overlap_and_greens_function_scaled_batched(
        w.Qmat, w.log_d, w.Tmat, 0.0
    )
    q = numpy.exp(trace - w.log_ovlp)
    numpy.testing.assert_allclose(q.imag, 0, atol=2e-12)
    # Physical path mass: HS scalars times the Fock trace of the walker.
    scalars = numpy.ones(64, dtype=complex)
    for iw in range(64):
        for x in histories[iw].ravel():
            scalars[iw] *= field_factors(p, hs, x)[1]
    target_mass = scalars * numpy.prod(
        [numpy.linalg.det(numpy.eye(nsite) + prev[:, s]) for s in range(2)], axis=0
    )
    numpy.testing.assert_allclose(target_mass.imag, 0, atol=2e-12)
    initial_Z = numpy.linalg.det(numpy.eye(nsite) + expm(-trial.beta * hop)) ** 2
    numpy.testing.assert_allclose(initial_Z * pdf * w.weight * q, target_mass, rtol=2e-10, atol=0)
    # The trace-weighted energy equals the Fock-space thermal value of the same Trotter product.
    cre = creation_operators(4)
    number = [c @ c.T for c in cre]
    Hk = sum(
        hop[i, j] * cre[2 * s + i] @ cre[2 * s + j].T
        for s in range(2)
        for i in range(2)
        for j in range(2)
    )
    double = sum(number[i] @ number[2 + i] for i in range(2))
    Hi = U * (double - sum(number) / 2)
    rho = numpy.linalg.matrix_power(expm(-dt * Hk / 2) @ expm(-dt * Hi) @ expm(-dt * Hk / 2), 3)
    numpy.testing.assert_allclose(target_mass.sum(), numpy.trace(rho), rtol=5e-11)
    density = numpy.diagonal(green, axis1=-2, axis2=-1)
    energy = numpy.einsum("ij,nsji->n", hop, green) + U * numpy.sum(
        density[:, 0] * density[:, 1], axis=1
    )
    # Expectation over the proposal: sum_paths pdf w Q E / sum_paths pdf w Q.
    exact = numpy.trace(rho @ (Hk + U * double)) / numpy.trace(rho)
    numpy.testing.assert_allclose(
        numpy.sum(pdf * w.weight * q * energy) / numpy.sum(pdf * w.weight * q), exact, atol=5e-12
    )


@pytest.mark.unit
@pytest.mark.parametrize("hs", ["charge", "spin"])
def test_bridge_small_lambda_matches_undressed(hs):
    # A vanishing lambda recovers the undressed propagation on identical random fields.
    runs = []
    for lam in (0.0, 1e-12):
        _, _, trial, mpi, w, p = build_test_case(hs, "completion", lam, beta=1, dt=0.05, nw=16)
        w.field_hist = numpy.zeros((w.nwalkers, 20, 2))
        w.reset(trial, compute_cache=False)
        p.init_walker_cache(w, trial)
        numpy.random.seed(7)
        for t in range(20):
            p.propagate_walkers(w, trial, t)
        runs.append((w.weight.copy(), w.log_ovlp.copy(), w.field_hist.copy()))
    numpy.testing.assert_array_equal(runs[0][2], runs[1][2])
    numpy.testing.assert_allclose(runs[1][0], runs[0][0], rtol=1e-9)
    numpy.testing.assert_allclose(runs[1][1], runs[0][1], atol=1e-9)


@pytest.mark.unit
@pytest.mark.parametrize("guide", ["completion", "fixed"])
@pytest.mark.parametrize("hs", ["charge", "spin"])
def test_bridge_population_cache_and_endpoint(hs, guide):
    # Branching and stabilization carry the dressing; it is removed at the endpoint.
    _, _, trial, mpi, w, p = build_test_case(hs, guide, 0.5, beta=1, dt=0.05, nw=8)
    driver = SimpleNamespace(
        mpi_handler=mpi, params=SimpleNamespace(num_stblz=2, pop_control_freq=2)
    )
    pc = ThermofieldPopController(w.nwalkers, 20, mpi, "pair_branch", verbose=False)
    pc.min_weight = 0.9
    pc.max_weight = 1.1
    numpy.random.seed(92324)
    propagate_population_cp(driver, w, trial, p, 20, pc)
    fresh, _ = trial.calc_log_overlap_and_greens_function_scaled_batched(
        w.Qmat, w.log_d, w.Tmat, p.guide_scale(19)
    )
    numpy.testing.assert_allclose(numpy.exp(fresh - w.log_ovlp), 1, atol=2e-10)
    numpy.testing.assert_array_equal(w.log_dress, 0)
    assert p.kill_count == p.forbidden_field_count == p.invalid_weight_count == 0
    assert "log_dress" in w.buff_names


@pytest.mark.unit
def test_bridge_rejects_unsupported_options():
    _, hop, trial, _, w, p = build_test_case("spin", "completion", 0.0)
    h_kin = numpy.array([hop, hop])
    with pytest.raises(ValueError, match="bridge_lambda"):
        ThermofieldHubbardDiscrete(h_kin, 8, 0.1, 0.3, bridge_lambda=1.5)
    with pytest.raises(ValueError, match="bridge_theta"):
        ThermofieldHubbardDiscrete(h_kin, 8, 0.1, 0.3, bridge_lambda=0.5, bridge_theta=4.0)
    with pytest.raises(ValueError, match="ordinary guide branching"):
        ThermofieldHubbardDiscrete(h_kin, 8, 0.1, 0.3, bridge_lambda=0.5, trace_branching=True)
    with pytest.raises(ValueError, match="whole number of slices"):
        ThermofieldHubbardDiscrete(h_kin, 8, 0.07, 0.3, bridge_lambda=0.5)


@pytest.mark.mpi
@pytest.mark.parametrize("guide", ["completion", "fixed"])
@pytest.mark.parametrize("hs", ["charge", "spin"])
def test_bridge_mpi_population_transport(hs, guide):
    # Walkers exchanged between ranks carry a dressing equal to the one of their own state.
    _, _, trial, mpi, w, p = build_test_case(hs, guide, 0.5, beta=1, dt=0.05, nw=8)
    if mpi.comm.size < 2:
        pytest.skip("Requires two MPI ranks")
    driver = SimpleNamespace(
        mpi_handler=mpi, params=SimpleNamespace(num_stblz=2, pop_control_freq=1)
    )
    pc = ThermofieldPopController(w.nwalkers, 19, mpi, "pair_branch", verbose=False)
    pc.min_weight = 0.9
    pc.max_weight = 1.1
    numpy.random.seed(100 + mpi.comm.rank)
    # One slice short of the endpoint, so the dressing is still live.
    propagate_population_cp(driver, w, trial, p, 19, pc)
    alive = w.weight != 0
    front = numpy.stack([numpy.eye(2) - w.Ga, numpy.eye(2) - w.Gb], axis=1)
    fresh = numpy.log(p._bridge_dress(p.BK2inv @ front @ p.BK2).real)
    numpy.testing.assert_allclose(w.log_dress.real[alive], fresh[alive], atol=1e-9)
    # The dressing is live and nontrivial.
    assert numpy.abs(w.log_dress.real[alive]).max() > 1e-3
    assert p.kill_count == p.forbidden_field_count == p.invalid_weight_count == 0


if __name__ == "__main__":
    test_bridge_all_paths_against_explicit_importance("spin", "fixed", True, 0.5)
    test_bridge_small_lambda_matches_undressed("charge")
    test_bridge_population_cache_and_endpoint("spin", "completion")
    test_bridge_rejects_unsupported_options()
    test_bridge_mpi_population_transport("charge", "completion")
