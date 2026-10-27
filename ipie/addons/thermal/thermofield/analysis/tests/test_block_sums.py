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

import json
import pathlib
import tempfile

import numpy
import pytest

from ipie.addons.thermal.thermofield.analysis.block_sums import (
    fit_mu_scan,
    load_block_sums,
    summarize_blocks,
)


def write_run(path, energy, denom, shifts=None, nav=None, **metadata):
    denom = numpy.asarray(denom, dtype=complex)
    energy = numpy.broadcast_to(energy, denom.shape)
    nav = numpy.broadcast_to(66.0 if nav is None else nav, denom.shape)
    shifts = numpy.zeros(len(denom)) if shifts is None else numpy.asarray(shifts)
    data = dict(
        rs=3.0,
        theta=0.5,
        mu_run=-0.024,
        guide="thf",
        guide_mode="fixed",
        trace_branching=False,
        nwalkers=512,
        nranks=1,
        blocks_done=len(denom),
    )
    data.update(metadata)
    for name, values in (("numer", energy * denom), ("denom", denom), ("navnum", nav * denom)):
        data[name + "_re"] = values.real.tolist()
        data[name + "_im"] = values.imag.tolist()
    data["logshift"] = shifts.tolist()
    data["neff_blk"] = [0.1] * len(denom)
    path.write_text(json.dumps(data))
    return path


@pytest.mark.unit
def test_pooling_is_independent_of_file_partition():
    # Pooling must keep the denominator mass of every run, whatever the file split.
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        a = write_run(tmp_path / "a.json", [1, 2], [1, 1])
        b = write_run(tmp_path / "b.json", [7, 8], [9, 9])
        whole = write_run(tmp_path / "all.json", [1, 2, 7, 8], [1, 1, 9, 9])
        split = summarize_blocks(load_block_sums([a, b]), nboot=300, seed=4)
        joined = summarize_blocks(load_block_sums([whole]), nboot=300, seed=4)
        legacy = summarize_blocks(
            load_block_sums([a, b], normalization="legacy-per-run"), nboot=300
        )
    assert split["energy"] == pytest.approx(6.9)
    assert split["energy"] == joined["energy"]
    assert split["bootstrap_error"] == joined["bootstrap_error"]
    assert legacy["energy"] == pytest.approx(4.5)


@pytest.mark.unit
def test_logshift_and_complex_observables():
    e = numpy.array([1 + 2j, 3 - 1j, 2 + 4j])
    d = numpy.array([1 + 0.5j, 2 - 0.2j, 0.5 + 0.3j])
    with tempfile.TemporaryDirectory() as tmp_dir:
        # The last block is stored relative to a different logarithmic shift.
        p = write_run(
            pathlib.Path(tmp_dir) / "r.json",
            e,
            d / numpy.array([1, 1, 2]),
            shifts=[1000, 1000, 1000 + numpy.log(2)],
        )
        r = summarize_blocks(load_block_sums([p]), nboot=300)
    assert r["energy"] == pytest.approx((numpy.sum(e * d) / numpy.sum(d)).real)
    assert r["energy"] != pytest.approx(numpy.sum((e * d).real) / numpy.sum(d.real))


@pytest.mark.unit
def test_bootstrap_preserves_energy_number_covariance():
    n = numpy.linspace(62, 70, 30)
    with tempfile.TemporaryDirectory() as tmp_dir:
        p = write_run(pathlib.Path(tmp_dir) / "r.json", 0.4 * n - 20, numpy.arange(1, 31), nav=n)
        r = summarize_blocks(load_block_sums([p]), nboot=400)
    draws = numpy.asarray(r["bootstrap_samples"])
    numpy.testing.assert_allclose(draws[:, 0], 0.4 * draws[:, 1] - 20, atol=1e-13)


@pytest.mark.unit
def test_mixed_ensembles_are_rejected():
    changes = [{"trace_branching": True}, {"guide": "fe"}, {"mu_run": -0.02}, {"nwalkers": 640}]
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        a = write_run(tmp_path / "a.json", [1, 2], [1, 2])
        for change in changes:
            b = write_run(tmp_path / "b.json", [3, 4], [1, 2], **change)
            with pytest.raises(ValueError, match="Incompatible"):
                load_block_sums([a, b])


@pytest.mark.unit
def test_missing_population_normalization_is_rejected():
    with tempfile.TemporaryDirectory() as tmp_dir:
        p = write_run(pathlib.Path(tmp_dir) / "r.json", [1, 2], [1, 2])
        with pytest.raises(ValueError, match="pop_log_norm"):
            load_block_sums([p], restore_population=True)


@pytest.mark.unit
def test_recorded_population_normalization():
    with tempfile.TemporaryDirectory() as tmp_dir:
        p = write_run(
            pathlib.Path(tmp_dir) / "r.json", [1, 2], [1, 1], pop_log_norm=[0, numpy.log(3)]
        )
        r = summarize_blocks(load_block_sums([p], restore_population=True), nboot=300)
    assert r["energy"] == pytest.approx(1.75)


@pytest.mark.unit
def test_scan_joint_bootstrap_finds_root():
    seed = 21
    numpy.random.seed(seed)
    rows = []
    for mu in [-0.026, -0.024, -0.022, -0.020]:
        nav = 66 + 500 * (mu + 0.023)
        energy = 5.5 + 0.2 * (nav - 66)
        noise = numpy.random.normal(0, 0.08, 600)
        rows.append(
            dict(
                mu=mu,
                energy=energy,
                nav=nav,
                energy_error=0.016,
                nav_error=0.08,
                bootstrap_samples=numpy.column_stack((energy + 0.2 * noise, nav + noise)).tolist(),
            )
        )
    fit = fit_mu_scan(rows)
    assert fit["mu_star"] == pytest.approx(-0.023)
    assert fit["energy_at_target"] == pytest.approx(5.5)
    assert fit["dE_dN"] == pytest.approx(0.2)
    assert fit["energy_at_target_error"] < 1e-12
    assert fit["bracketed"]
    assert fit["mu_star_error"] > 0
    outside = fit_mu_scan(rows, target=100)
    assert not outside["bracketed"]


if __name__ == "__main__":
    test_pooling_is_independent_of_file_partition()
    test_logshift_and_complex_observables()
    test_bootstrap_preserves_energy_number_covariance()
    test_mixed_ensembles_are_rejected()
    test_missing_population_normalization_is_rejected()
    test_recorded_population_normalization()
    test_scan_joint_bootstrap_finds_root()
