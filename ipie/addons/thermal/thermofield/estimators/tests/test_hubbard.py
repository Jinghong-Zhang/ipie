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

from ipie.addons.thermal.thermofield.estimators import cyclic_average
from ipie.addons.thermal.thermofield.estimators.hubbard import HubbardSpinObservables
from ipie.addons.thermal.thermofield.utils.testing import creation_operators


@pytest.mark.unit
def test_hubbard_observables_smoke():
    # System params.
    lx = 5
    ly = 5
    U = 8

    nsites = lx * ly
    obs = HubbardSpinObservables(numpy.zeros((nsites, nsites)), U=U, lx=lx, ly=ly)
    G = numpy.repeat((0.5 * numpy.eye(nsites))[None], 2, axis=0)
    raw = obs.measure(G)
    assert raw.shape == (obs.size,)
    assert numpy.isfinite(raw).all()
    result = obs.finalise(raw)
    assert result["density"] == 1
    assert result["double_occupancy"] == 0.25
    assert result["singles"] == 0.5
    assert result["energy_per_site"] == 2
    assert all(c == 0 for c in result["czz"].values())


@pytest.mark.unit
def test_shells_include_all_equivalent_offsets():
    # System params.
    lx = 20
    ly = 20
    U = 8

    nsites = lx * ly
    obs = HubbardSpinObservables(numpy.zeros((nsites, nsites)), U, lx, ly)
    assert obs.metadata["shell_pair_counts"] == {"1": 1600, "2": 1600, "4": 1600, "5": 3200}
    expected = {
        1: {(1, 0), (-1, 0), (0, 1), (0, -1)},
        2: {(1, 1), (-1, 1), (1, -1), (-1, -1)},
        4: {(2, 0), (-2, 0), (0, 2), (0, -2)},
        5: {(sx * a, sy * b) for a, b in ((1, 2), (2, 1)) for sx in (-1, 1) for sy in (-1, 1)},
    }
    for d, (i, j) in obs.pairs.items():
        actual = set()
        for target in j[i == 0]:
            x, y = target % lx, target // lx
            actual.add((x if x < lx // 2 else x - lx, y if y < ly // 2 else y - ly))
        assert actual == expected[d]
        assert len(set(zip(i, j))) == len(i)


@pytest.mark.unit
def test_wick_observables_against_fock_trace():
    # System params.
    U = 3.2
    hop = numpy.array([[0, -1 + 0.3j], [-1 - 0.3j, 0]])
    seed = 781
    numpy.random.seed(seed)

    # A complex non-Hermitian Gaussian product exercises the transpose conventions.
    cre = creation_operators(4)
    number = [c @ c.T for c in cre]
    generators = []
    for _ in range(2):
        op = numpy.zeros((16, 16), dtype=complex)
        for s in range(2):
            a = numpy.random.normal(size=(2, 2)) + 1j * numpy.random.normal(size=(2, 2))
            h = a + a.conj().T
            for i in range(2):
                for j in range(2):
                    op += h[i, j] * (cre[2 * s + i] @ cre[2 * s + j].T)
        generators.append(op)
    rho = scipy.linalg.expm(-0.35 * generators[0]) @ scipy.linalg.expm(-0.24 * generators[1])
    Z = numpy.trace(rho)

    def expectation(op):
        return numpy.trace(rho @ op) / Z

    G = numpy.empty((2, 2, 2), dtype=complex)
    for s in range(2):
        for i in range(2):
            for j in range(2):
                G[s, j, i] = expectation(cre[2 * s + i] @ cre[2 * s + j].T)
    obs = HubbardSpinObservables(hop, U, 2, 1, (1,))
    raw = obs.measure(G)
    one_body = sum(
        hop[i, j] * (cre[2 * s + i] @ cre[2 * s + j].T)
        for s in range(2)
        for i in range(2)
        for j in range(2)
    )
    doubles = number[0] @ number[2] + number[1] @ number[3]
    magnetization = [number[i] - number[2 + i] for i in range(2)]
    expected = numpy.array(
        [
            expectation(one_body + U * doubles) / 2,
            expectation(sum(number)) / 2,
            expectation(doubles) / 2,
            expectation(sum(number) - 2 * doubles) / 2,
            *(expectation(m) for m in magnetization),
            expectation(magnetization[0] @ magnetization[1]),
        ]
    )
    numpy.testing.assert_allclose(raw, expected, atol=2e-14)
    assert numpy.max(numpy.abs(raw.imag)) > 1e-5


@pytest.mark.unit
def test_connected_subtraction_must_follow_ensemble_average():
    obs = HubbardSpinObservables(numpy.zeros((2, 2)), 4, 2, 1, (1,))
    polarized = numpy.array([numpy.eye(2), numpy.zeros((2, 2))])
    up, down = obs.measure(polarized), obs.measure(polarized[::-1])
    assert obs.finalise(up)["czz"]["1"] == 0
    assert obs.finalise(down)["czz"]["1"] == 0
    assert obs.finalise((up + down) / 2)["czz"]["1"] == 1


@pytest.mark.unit
def test_rom_log_shift_and_nonlinear_jackknife():
    obs = HubbardSpinObservables(numpy.zeros((2, 2)), 4, 2, 1, (1,))
    occupations = numpy.array(
        [
            [[0.8, 0.7], [0.1, 0.2]],
            [[0.2, 0.3], [0.7, 0.6]],
            [[0.6, 0.9], [0.2, 0.4]],
            [[0.4, 0.1], [0.8, 0.9]],
        ]
    )
    raw = numpy.array(
        [obs.measure(numpy.array([numpy.diag(n) for n in row])) for row in occupations]
    )
    weight = numpy.array([1 + 0.2j, 2 - 0.1j, 0.7 + 0.1j, 4 + 0.3j])
    shifts = numpy.array([-2, 3, 1, -1])
    numer = weight[:, None] * raw * numpy.exp(-shifts[:, None])
    denom = weight * numpy.exp(-shifts)
    result = obs.summary(numer, denom, shifts)
    shifted = obs.summary(numer, denom, shifts + 1000)
    physical_mean = numpy.sum(weight[:, None] * raw, axis=0) / weight.sum()
    expected_czz = (physical_mean[-1] - physical_mean[4] * physical_mean[5]).real
    assert result["mean"]["czz"]["1"] == pytest.approx(expected_czz, abs=2e-15)
    assert shifted["mean"]["czz"]["1"] == pytest.approx(expected_czz, abs=2e-15)
    assert abs(expected_czz - numpy.mean(raw[:, -1] - raw[:, 4] * raw[:, 5]).real) > 0.05
    leave_values = []
    for skip in range(4):
        use = numpy.arange(4) != skip
        mean = numpy.sum(weight[use, None] * raw[use], axis=0) / weight[use].sum()
        leave_values.append((mean[-1] - mean[4] * mean[5]).real)
    expected_error = numpy.sqrt(3 / 4 * numpy.sum((leave_values - numpy.mean(leave_values)) ** 2))
    assert result["error"]["czz"]["1"] == pytest.approx(expected_error, abs=2e-15)
    assert shifted["error"]["czz"]["1"] == pytest.approx(expected_error, abs=2e-15)
    grouped = obs.summary(numer, denom, shifts, block_size=2)
    assert grouped["jackknife_groups"] == 2
    with pytest.raises(ValueError, match="divide"):
        obs.summary(numer, denom, shifts, block_size=3)


@pytest.mark.unit
def test_unavailable_shell_and_zero_denominator():
    with pytest.raises(ValueError, match="No periodic"):
        HubbardSpinObservables(numpy.zeros((2, 2)), 4, 2, 1, (4,))
    obs = HubbardSpinObservables(numpy.zeros((2, 2)), 4, 2, 1, (1,))
    with pytest.raises(ValueError, match="vanishes"):
        obs.summary(numpy.zeros((2, obs.size)), [1, -1], [0, 0])


@pytest.mark.unit
def test_optional_cyclic_phase_certificate():
    # A sign error at a later cut is invisible to magnitude-only diagnostics.
    original_kernel = cyclic_average.stabilized_inverse_one_plus_batched

    def run(phase, check_phase=None):
        calls = 0
        measured = []

        def perturbed_kernel(*args, **kwargs):
            nonlocal calls
            logdet, inverse = original_kernel(*args, **kwargs)
            calls += 1
            # Calls 1 and 2 are the two spins at cut zero. Only the phase of the
            # first spin at cut one changes; all magnitudes and G are unchanged.
            if calls == 3:
                logdet = logdet + 1j * phase
            return logdet, inverse

        options = {} if check_phase is None else {"check_phase": check_phase}
        cyclic_average.stabilized_inverse_one_plus_batched = perturbed_kernel
        try:
            result = cyclic_average.cyclic_average_engine(
                numpy.ones(1),
                2,
                1,
                1,
                lambda l, ids: numpy.ones((len(ids), 2, 1, 1), dtype=complex),
                lambda iw, G, w: measured.append(G.copy()),
                **options,
            )
        finally:
            cyclic_average.stabilized_inverse_one_plus_batched = original_kernel
        return result, numpy.asarray(measured)

    for phase, expected in ((numpy.pi, numpy.pi), (2 * numpy.pi, 0.0)):
        (default_dev, default_log0), default_greens = run(phase)
        (disabled_dev, disabled_log0), disabled_greens = run(phase, False)
        (checked_dev, checked_log0), checked_greens = run(phase, True)
        assert default_dev == disabled_dev == 0.0
        assert checked_dev == pytest.approx(expected, abs=1e-14)
        numpy.testing.assert_array_equal(default_log0, disabled_log0)
        numpy.testing.assert_array_equal(default_log0, checked_log0)
        numpy.testing.assert_array_equal(default_greens, disabled_greens)
        numpy.testing.assert_array_equal(default_greens, checked_greens)


@pytest.mark.unit
def test_jackknife_preserves_small_mass_after_deleting_dominant_block():
    obs = HubbardSpinObservables(numpy.zeros((2, 2)), 4, 2, 1, (1,))
    greens = [
        numpy.array([numpy.diag([0.8, 0.7]), numpy.diag([0.1, 0.2])]),
        numpy.array([numpy.diag([0.3, 0.1]), numpy.diag([0.6, 0.8])]),
        numpy.array([numpy.diag([0.4, 0.6]), numpy.diag([0.5, 0.2])]),
    ]
    raw = numpy.array([obs.measure(g) for g in greens])
    result = obs.summary(raw, numpy.ones(3), [70.0, 0.0, -1.0])
    assert numpy.isfinite(result["error"]["czz"]["1"])
    assert result["error"]["czz"]["1"] > 0


if __name__ == "__main__":
    test_hubbard_observables_smoke()
    test_shells_include_all_equivalent_offsets()
    test_wick_observables_against_fock_trace()
    test_connected_subtraction_must_follow_ensemble_average()
    test_rom_log_shift_and_nonlinear_jackknife()
    test_unavailable_shell_and_zero_denominator()
    test_optional_cyclic_phase_certificate()
    test_jackknife_preserves_small_mass_after_deleting_dominant_block()
