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

from itertools import product
from types import SimpleNamespace

import numpy
import pytest
from scipy.linalg import expm

from ipie.addons.thermal.thermofield.estimators.cyclic_average import cyclic_average_hubbard
from ipie.addons.thermal.thermofield.hamiltonians.hubbard import build_hubbard_hamiltonian
from ipie.addons.thermal.thermofield.propagation.hubbard_discrete import ThermofieldHubbardCharge
from ipie.addons.thermal.thermofield.qmc.hubbard_afqmc import propagate_population_cp
from ipie.addons.thermal.thermofield.trial.hubbard import ThermofieldCPTrial
from ipie.addons.thermal.thermofield.utils.testing import creation_operators
from ipie.addons.thermal.thermofield.walkers.pop_controller import ThermofieldPopController
from ipie.addons.thermal.thermofield.walkers.uhf_walkers import ThermofieldWalkers
from ipie.utils.mpi import MPIHandler


def build_test_case(U=8, dt=0.1, beta=0.3, nw=64, cache=True, guide="completion"):
    # Open two-site chain at half filling with the noninteracting guide.
    ham, hop, _ = build_hubbard_hamiltonian(2, 1, U, 0, pbc_x=False, pbc_y=False)
    trial = ThermofieldCPTrial(ham, beta, k_trial=numpy.array([hop, hop]), guide_time=beta)
    mpi = MPIHandler()
    w = ThermofieldWalkers(trial, 2, nw, mpi)
    prop = ThermofieldHubbardCharge(
        numpy.array([hop, hop]),
        U,
        dt,
        beta,
        record_fields=True,
        cache_qinv=cache,
        guide_mode=guide,
        fixed_guide_time=beta if guide == "fixed" else None,
    )
    return ham, hop, trial, mpi, w, prop


@pytest.mark.unit
@pytest.mark.parametrize("U,dt", [(0, 0.05), (4, 0.01), (8, 0.05)])
def test_charge_local_identity(U, dt):
    alpha = numpy.arccos(numpy.exp(-dt * U / 2))
    for up, down in product([0, 1], repeat=2):
        decoupled = (
            sum(numpy.exp(dt * U / 2 + 1j * alpha * x * (up + down - 1)) for x in [-1, 1]) / 2
        )
        numpy.testing.assert_allclose(
            decoupled, numpy.exp(-dt * U * (up * down - (up + down) / 2)), atol=3e-15
        )


@pytest.mark.unit
@pytest.mark.parametrize("guide", ["completion", "fixed"])
@pytest.mark.parametrize("cache", [True, False])
def test_charge_all_paths_against_fock_trace(cache, guide):
    # All 64 paths are enumerated to check both the proposal weights and the observables.
    U, dt, nslices = 8, 0.1, 3
    ham, hop, trial, mpi, w, p = build_test_case(cache=cache, guide=guide)
    histories = numpy.array(list(product([1, -1], repeat=6))).reshape(64, 3, 2)
    w.field_hist = numpy.zeros_like(histories, dtype=float)
    p.init_walker_cache(w, trial)
    if cache:
        p.seed_qinv_cache(w)
    K = expm(-dt * hop / 2)
    pdf = numpy.ones(64)
    matrices = numpy.repeat(numpy.eye(2, dtype=complex)[None], 64, axis=0)
    for step in range(nslices):
        closure = expm(-(trial.beta if guide == "fixed" else trial.beta - (step + 1) * dt) * hop)
        for iw, fields in enumerate(histories[:, step]):
            factors = numpy.ones(2, complex)
            old = numpy.linalg.det(numpy.eye(2) + closure @ K @ K @ matrices[iw]) ** 2
            for site, selected in enumerate(fields):
                ratios = []
                determinants = []
                for x in [1, -1]:
                    new = factors.copy()
                    new[site] *= numpy.exp(1j * p.gamma * x)
                    det = (
                        numpy.linalg.det(
                            numpy.eye(2) + closure @ K @ numpy.diag(new) @ K @ matrices[iw]
                        )
                        ** 2
                    )
                    ratios.append(0.5 * numpy.exp(dt * U / 2 - 1j * p.gamma * x) * det / old)
                    determinants.append(det)
                numpy.testing.assert_allclose(numpy.imag(ratios), 0, atol=2e-12)
                ratios = numpy.real(ratios)
                assert numpy.all(ratios > 0)
                pick = 0 if selected == 1 else 1
                pdf[iw] *= ratios[pick] / sum(ratios)
                factors[site] *= numpy.exp(1j * p.gamma * selected)
                old = determinants[pick]
            matrices[iw] = K @ numpy.diag(factors) @ K @ matrices[iw]
        uniforms = numpy.where(histories[:, step] == 1, 0.0, numpy.nextafter(1.0, 0.0))
        p._advance(w, trial, step, numpy.arange(64), uniforms)
        if step == 1:
            w.stabilize()
            if cache:
                p.refresh_qinv_after_stabilize(w)
    assert p.kill_count == p.forbidden_field_count == p.invalid_weight_count == 0
    numpy.testing.assert_array_equal(w.field_hist, histories)
    trace, green = trial.calc_log_overlap_and_greens_function_scaled_batched(
        w.Qmat, w.log_d, w.Tmat, 0.0
    )
    # The endpoint correction Q = Tr(B) / Tr(Lambda B) is one for the
    # completion guide and real and nonnegative for a fixed guide.
    q = numpy.exp(trace - w.log_ovlp)
    numpy.testing.assert_allclose(q.imag, 0, atol=2e-12)
    assert numpy.all(q.real >= 0)
    if guide == "completion":
        numpy.testing.assert_allclose(q, 1, atol=2e-12)
    numpy.testing.assert_allclose(green[:, 0], green[:, 1], atol=2e-12)
    det = numpy.linalg.det(numpy.eye(2) + matrices) ** 2
    scalar = numpy.exp(6 * dt * U / 2 - 1j * p.gamma * histories.sum(axis=(1, 2)))
    target_mass = scalar * det / 64
    numpy.testing.assert_allclose(target_mass.imag, 0, atol=2e-12)
    initial_Z = numpy.linalg.det(numpy.eye(2) + expm(-trial.beta * hop)) ** 2
    numpy.testing.assert_allclose(initial_Z * pdf * w.weight * q, target_mass, atol=3e-11)
    numpy.testing.assert_allclose(pdf.sum(), 1, atol=2e-13)

    cre = creation_operators(4)
    number = [c @ c.T for c in cre]
    Hk = sum(
        hop[i, j] * cre[2 * s + i] @ cre[2 * s + j].T
        for s in range(2)
        for i in range(2)
        for j in range(2)
    )
    double = sum(number[i] @ number[2 + i] for i in range(2))
    Hphys = Hk + U * double
    Hi = U * (double - sum(number) / 2)
    rho = numpy.linalg.matrix_power(expm(-dt * Hk / 2) @ expm(-dt * Hi) @ expm(-dt * Hk / 2), 3)
    numpy.testing.assert_allclose(target_mass.sum(), numpy.trace(rho), atol=5e-11)
    density = numpy.diagonal(green, axis1=-2, axis2=-1)
    energy = numpy.einsum("ij,nsji->n", hop, green) + U * numpy.sum(
        density[:, 0] * density[:, 1], axis=1
    )
    czz = -2 * green[:, 0, 0, 1] * green[:, 0, 1, 0]
    spin_pair = (number[0] - number[2]) @ (number[1] - number[3])
    numpy.testing.assert_allclose(
        numpy.sum(target_mass * energy) / target_mass.sum(),
        numpy.trace(rho @ Hphys) / numpy.trace(rho),
        atol=5e-12,
    )
    numpy.testing.assert_allclose(
        numpy.sum(target_mass * czz) / target_mass.sum(),
        numpy.trace(rho @ spin_pair) / numpy.trace(rho),
        atol=5e-12,
    )
    cuts = []
    dev, cut0 = cyclic_average_hubbard(
        w, trial, p, 3, 1, lambda iw, G, cut: cuts.append((iw, G, cut)), stab_every=1
    )
    assert dev < 1e-10 and len(cuts) == 192
    numpy.testing.assert_allclose(numpy.exp(cut0 - trace), 1, atol=2e-12)
    for iw, G, cut in cuts:
        ordered = [K @ numpy.diag(numpy.exp(1j * p.gamma * x)) @ K for x in histories[iw]]
        rotation = ordered[cut:] + ordered[:cut]
        direct = numpy.eye(2, dtype=complex)
        for B in rotation:
            direct = B @ direct
        numpy.testing.assert_allclose(
            G[0], numpy.eye(2) - numpy.linalg.inv(numpy.eye(2) + direct), atol=3e-12
        )


@pytest.mark.unit
def test_charge_population_history_and_endpoint():
    _, _, trial, mpi, w, p = build_test_case(beta=1, dt=0.05, nw=8)
    driver = SimpleNamespace(
        mpi_handler=mpi, params=SimpleNamespace(num_stblz=2, pop_control_freq=2)
    )
    pc = ThermofieldPopController(w.nwalkers, 20, mpi, "pair_branch", verbose=False)
    pc.min_weight = 0.9
    pc.max_weight = 1.1
    numpy.random.seed(92324)
    propagate_population_cp(driver, w, trial, p, 20, pc)
    trace = trial.calc_log_trace_batched(w.Qmat, w.log_d, w.Tmat)
    dev, cut0 = cyclic_average_hubbard(w, trial, p, 20, 2, lambda *args: None, stab_every=2)
    numpy.testing.assert_allclose(numpy.exp(trace - w.log_ovlp), 1, atol=2e-10)
    numpy.testing.assert_allclose(numpy.exp(cut0 - trace), 1, atol=2e-10)
    assert dev < 1e-9 and p.kill_count == p.forbidden_field_count == 0


@pytest.mark.unit
def test_charge_fixed_guide_population_endpoint():
    _, _, trial, mpi, w, p = build_test_case(beta=1, dt=0.05, nw=8, guide="fixed")
    driver = SimpleNamespace(
        mpi_handler=mpi, params=SimpleNamespace(num_stblz=2, pop_control_freq=2)
    )
    pc = ThermofieldPopController(w.nwalkers, 20, mpi, "pair_branch", verbose=False)
    pc.min_weight = 0.9
    pc.max_weight = 1.1
    numpy.random.seed(92324)
    propagate_population_cp(driver, w, trial, p, 20, pc)
    trace = trial.calc_log_trace_batched(w.Qmat, w.log_d, w.Tmat)
    fresh, _ = trial.calc_log_overlap_and_greens_function_scaled_batched(
        w.Qmat, w.log_d, w.Tmat, trial.beta
    )
    q = numpy.exp(trace - w.log_ovlp)
    numpy.testing.assert_allclose(numpy.exp(fresh - w.log_ovlp), 1, atol=2e-10)
    numpy.testing.assert_allclose(q.imag, 0, atol=2e-10)
    assert numpy.all(q.real > 0)
    assert p.kill_count == p.forbidden_field_count == p.invalid_weight_count == 0


@pytest.mark.unit
def test_charge_rejects_unsupported_models_and_trials():
    _, hop, trial, _, w, p = build_test_case()
    h_kin = numpy.array([hop, hop])
    with pytest.raises(ValueError, match="chemical potential"):
        ThermofieldHubbardCharge(h_kin + 0.1 * numpy.eye(2), 8, 0.1, 0.3)
    triangle = numpy.ones((2, 3, 3)) - numpy.eye(3)
    with pytest.raises(ValueError, match="bipartite"):
        ThermofieldHubbardCharge(triangle, 8, 0.1, 0.3)
    trial.eps_T[0] += 0.1
    with pytest.raises(ValueError, match="particle-hole trial"):
        p.init_walker_cache(w, trial)


if __name__ == "__main__":
    test_charge_local_identity(4, 0.01)
    test_charge_all_paths_against_fock_trace(True, "fixed")
    test_charge_population_history_and_endpoint()
    test_charge_fixed_guide_population_endpoint()
    test_charge_rejects_unsupported_models_and_trials()
