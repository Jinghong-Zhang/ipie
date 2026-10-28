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

from types import SimpleNamespace

import numpy
import pytest

from ipie.addons.thermal.thermofield.hamiltonians.hubbard import build_hubbard_hamiltonian
from ipie.addons.thermal.thermofield.propagation.density_density import (
    ThermofieldDensityDensityDiscrete,
)
from ipie.addons.thermal.thermofield.propagation.discrete_fields_gpu import (
    sample_discrete_fields,
)
from ipie.addons.thermal.thermofield.propagation.hubbard_discrete import (
    ThermofieldHubbardCharge,
    ThermofieldHubbardDiscrete,
)

cupy = pytest.importorskip("cupy")


def _reference(prop, green, uniforms, live):
    # The masked per-field loop on the host, with the propagator's own hooks.
    n, nfields = uniforms.shape
    nbasis = green.shape[-1]
    factors = numpy.ones((n, 2, nbasis), dtype=numpy.complex128)
    record = numpy.zeros((n, nfields))
    log_weight = numpy.zeros(n)
    log_s = numpy.zeros(n, dtype=numpy.complex128)
    counts = numpy.zeros(4, dtype=numpy.int64)
    rows = numpy.arange(n)
    for field in range(nfields):
        probabilities = prop._field_probabilities(green, field)
        with numpy.errstate(invalid="ignore"):
            good = numpy.isfinite(probabilities).all(axis=1) & (
                numpy.abs(probabilities.imag)
                <= prop.real_tolerance * numpy.maximum(1.0, numpy.abs(probabilities.real))
            ).all(axis=1)
        counts[0] += (live & ~good).sum()
        live &= good
        prob = numpy.where(live[:, None], probabilities.real, 1.0)
        counts[1] += live.sum()
        counts[3] += (live[:, None] & (prob <= 0)).sum()
        positive = numpy.maximum(prob, 0.0)
        norm = positive.sum(axis=1)
        counts[2] += (live & (norm == 0)).sum()
        live &= norm > 0
        safe_norm = numpy.where(live, norm, 1.0)
        chosen = (uniforms[:, field] >= positive[:, 0] / safe_norm).astype(numpy.int64)
        with numpy.errstate(invalid="ignore", divide="ignore"):
            prop._field_green_update(green, field, chosen)
        valid = numpy.isfinite(green).all(axis=(1, 2, 3))
        counts[0] += (live & ~valid).sum()
        prop._field_factors(factors, live, field, chosen[live])
        log_weight[live] += numpy.log(safe_norm[live])
        picked = prob[rows, chosen]
        log_s[live] += (numpy.log(2.0) + numpy.log(picked[live])) - prop._field_log_scalar(
            field, chosen
        )[live]
        record[live, field] = 1 - 2 * chosen[live]
        live &= valid
    return green, live, log_weight, log_s, factors, record, counts


def _thermal_green(nbasis, nwalkers):
    # Physical Green's functions (I + e^{h})^{-1} of random Hermitian h, slightly perturbed.
    green = numpy.empty((nwalkers, 2, nbasis, nbasis), dtype=numpy.complex128)
    for iw in range(nwalkers):
        for s in range(2):
            h = numpy.random.standard_normal((nbasis, nbasis))
            h = h + h.T + 1j * numpy.random.standard_normal((nbasis, nbasis)) * 0.1
            e, v = numpy.linalg.eig(h)
            green[iw, s] = (v / (1.0 + numpy.exp(e))) @ numpy.linalg.inv(v)
    return green


def _propagators():
    nbasis = 4
    ham, hop, _ = build_hubbard_hamiltonian(2, 2, 4.0, 0.3, pbc_x=False)
    spin = ThermofieldHubbardDiscrete(ham.H1 + 2.0 * numpy.eye(nbasis), 4.0, 0.05, 1.0)
    charge = ThermofieldHubbardCharge(numpy.array([hop, hop]), 4.0, 0.05, 1.0)
    # Opposite-spin, same-spin (with exchange) and attractive pairs.
    pairs = [(0, 0, 1, 0, 4.0), (0, 0, 0, 1, 1.5), (1, 2, 1, 3, -0.8), (0, 3, 1, 2, 2.0)]
    pair = ThermofieldDensityDensityDiscrete(ham.H1, pairs, 0.05, 1.0, grouped=False)
    return {"spin": spin, "charge": charge, "pair": pair}


@pytest.mark.gpu
@pytest.mark.parametrize("kmax", [0, 1, 3])
@pytest.mark.parametrize("kind", ["spin", "charge", "pair"])
def test_kernel_matches_host_loop(kind, kmax):
    numpy.random.seed(17)
    prop = _propagators()[kind]
    # Random Green's functions give complex probabilities; accept them so that
    # most walkers survive.
    prop.real_tolerance = 1.0
    nwalkers, nfields = 64, prop.nfields
    green = _thermal_green(prop.BK2.shape[-1], nwalkers)
    green[3, 1, 2, 0] = numpy.nan  # An invalid walker.
    uniforms = numpy.random.random((nwalkers, nfields))
    live = numpy.ones(nwalkers, dtype=bool)
    live[5] = False  # A walker killed before the fields.
    dtypes = (numpy.int32, numpy.int32, numpy.complex128, numpy.complex128, numpy.complex128)
    legs = tuple(
        cupy.asarray(numpy.ascontiguousarray(a, dtype=t))
        for a, t in zip(prop._field_legs(), dtypes)
    )
    out = dict(
        green=cupy.asarray(green),
        live=cupy.asarray(live),
        log_weight=cupy.zeros(nwalkers),
        log_s=cupy.zeros(nwalkers, dtype=cupy.complex128),
        factors=cupy.ones((nwalkers, 2, green.shape[-1]), dtype=cupy.complex128),
        record=cupy.zeros((nwalkers, nfields)),
    )
    counts = sample_discrete_fields(
        out["green"],
        cupy.asarray(uniforms),
        legs,
        out["live"],
        out["log_weight"],
        out["log_s"],
        out["factors"],
        out["record"],
        prop.real_tolerance,
        kmax=kmax,
    )
    ref = dict(
        zip(
            ("green", "live", "log_weight", "log_s", "factors", "record", "counts"),
            _reference(prop, green.copy(), uniforms, live.copy()),
        )
    )
    numpy.testing.assert_array_equal(cupy.asnumpy(counts), ref["counts"])
    alive = ref["live"]
    numpy.testing.assert_array_equal(cupy.asnumpy(out["live"]), alive)
    assert 0 < alive.sum() < nwalkers
    for key in ("green", "log_weight", "log_s", "factors", "record"):
        numpy.testing.assert_allclose(
            cupy.asnumpy(out[key])[alive], ref[key][alive], rtol=1e-12, atol=1e-12
        )


def _completion_propagators():
    ham, _, _ = build_hubbard_hamiltonian(2, 2, 4.0, 0.3, pbc_x=False)
    fixed = dict(guide_mode="fixed", fixed_guide_time=1.0, importance_guide="completion")
    pairs = [(0, 0, 1, 0, 4.0), (0, 0, 0, 1, 1.5), (1, 2, 1, 3, -0.8), (0, 3, 1, 2, 2.0)]
    return {
        "spin": ThermofieldHubbardDiscrete(ham.H1 + 2.0 * numpy.eye(4), 4.0, 0.05, 1.0, **fixed),
        "pair": ThermofieldDensityDensityDiscrete(ham.H1, pairs, 0.05, 1.0, grouped=False, **fixed),
    }


@pytest.mark.gpu
@pytest.mark.parametrize("kmax", [0, 1, 3])
@pytest.mark.parametrize("kind", ["spin", "pair"])
def test_completion_kernel_matches_slice_fields(kind, kmax):
    # The kernel against the propagator's own host loop, counters included.
    numpy.random.seed(23)
    prop = _completion_propagators()[kind]
    prop.real_tolerance = 1.0
    prop.record_fields = True
    nwalkers, nfields = 64, prop.nfields
    nbasis = prop.BK2.shape[-1]
    green = _thermal_green(nbasis, nwalkers)
    green[3, 1, 2, 0] = numpy.nan  # An invalid walker.
    live = numpy.ones(nwalkers, dtype=bool)
    live[5] = False  # A walker killed before the fields.
    state = SimpleNamespace(
        two_guides=True,
        green=green,
        live=live,
        log_weight=numpy.zeros(nwalkers),
        log_s=numpy.zeros(nwalkers, dtype=numpy.complex128),
        factors=numpy.ones((nwalkers, 2, nbasis), dtype=numpy.complex128),
        green_imp=_thermal_green(nbasis, nwalkers),
        log_s_imp=numpy.zeros(nwalkers, dtype=numpy.complex128),
    )
    uniforms = numpy.random.random((nwalkers, nfields))
    out = {k: cupy.asarray(v) for k, v in vars(state).items() if isinstance(v, numpy.ndarray)}
    record = cupy.zeros((nwalkers, nfields))
    dtypes = (numpy.int32, numpy.int32, numpy.complex128, numpy.complex128, numpy.complex128)
    legs = tuple(
        cupy.asarray(numpy.ascontiguousarray(a, dtype=t))
        for a, t in zip(prop._field_legs(), dtypes)
    )
    counts = sample_discrete_fields(
        out["green"],
        cupy.asarray(uniforms),
        legs,
        out["live"],
        out["log_weight"],
        out["log_s"],
        out["factors"],
        record,
        prop.real_tolerance,
        kmax=kmax,
        completion=(out["green_imp"], out["log_s_imp"]),
    )
    walkers = SimpleNamespace(field_hist=numpy.zeros((nwalkers, 1, nfields)))
    with numpy.errstate(all="ignore"):
        prop._slice_fields(walkers, 0, numpy.arange(nwalkers), uniforms, state)
    expected = [
        prop.invalid_weight_count,
        prop.step_count,
        prop.kill_count,
        prop.forbidden_field_count,
    ]
    numpy.testing.assert_array_equal(cupy.asnumpy(counts), expected)
    alive = state.live
    numpy.testing.assert_array_equal(cupy.asnumpy(out["live"]), alive)
    assert 0 < alive.sum() < nwalkers
    for key, value in out.items():
        if key != "live":
            numpy.testing.assert_allclose(
                cupy.asnumpy(value)[alive], getattr(state, key)[alive], rtol=1e-12, atol=1e-12
            )
    numpy.testing.assert_array_equal(cupy.asnumpy(record)[alive], walkers.field_hist[alive, 0])


if __name__ == "__main__":
    for k in ("spin", "charge", "pair"):
        test_kernel_matches_host_loop(k, 3)
    for k in ("spin", "pair"):
        test_completion_kernel_matches_slice_fields(k, 3)
