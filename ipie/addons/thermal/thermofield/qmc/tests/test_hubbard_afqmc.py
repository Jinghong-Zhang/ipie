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

import os
import tempfile

import numpy
import pytest
import scipy.linalg

from ipie.addons.thermal.thermofield.estimators.cyclic_average import cyclic_average_hubbard
from ipie.addons.thermal.thermofield.hamiltonians.hubbard import build_hubbard_hamiltonian
from ipie.addons.thermal.thermofield.propagation.hubbard import ThermofieldHubbardCP
from ipie.addons.thermal.thermofield.propagation.hubbard_discrete import ThermofieldHubbardDiscrete
from ipie.addons.thermal.thermofield.qmc.hubbard_afqmc import (
    ThermofieldHubbardAFQMC,
    tune_chemical_potential,
)
from ipie.addons.thermal.thermofield.trial.hubbard import rhf_k_trial, ThermofieldCPTrial
from ipie.addons.thermal.thermofield.walkers.uhf_walkers import ThermofieldWalkers
from ipie.config import MPI
from ipie.utils.mpi import MPIHandler

comm = MPI.COMM_WORLD


def _build_driver(
    U=2.0,
    beta=0.4,
    mu=0.2,
    guide_mode="fixed",
    wrap_stride=2,
    nblocks=3,
    measure_one_rdm=False,
    wrap_stabilize_freq=20,
):
    # Periodic 4x4 lattice with an RHF guide and continuous spin fields.
    lx = ly = 4
    ham, hop, v = build_hubbard_hamiltonian(lx, ly, U, mu)
    k_trial, _, _ = rhf_k_trial(hop, v, beta, 0.9 * lx * ly)
    trial = ThermofieldCPTrial(ham, beta, k_trial=k_trial, guide_time=beta)
    mpi = MPIHandler()
    walkers = ThermofieldWalkers(trial, ham.nbasis, 4, mpi)
    kinetic = ham.H1 + (U / 2) * numpy.eye(ham.nbasis)
    prop = ThermofieldHubbardCP(
        kinetic, U, 0.02, beta, guide_mode=guide_mode, fixed_guide_time=beta
    )
    return ThermofieldHubbardAFQMC(
        ham,
        hop,
        trial,
        walkers,
        prop,
        mpi,
        lx,
        ly,
        nblocks=nblocks,
        wrap_stride=wrap_stride,
        wrap_memory=1e6,
        measure_one_rdm=measure_one_rdm,
        wrap_stabilize_freq=wrap_stabilize_freq,
    )


def _exact_free_green(driver):
    # Grand-canonical Green's function of the noninteracting lattice.
    eigs, vecs = scipy.linalg.eigh(driver.hamiltonian.H1[0])
    occupied = 1 / (1 + numpy.exp(driver.propagator.beta * eigs))
    return (vecs * occupied) @ vecs.T


@pytest.mark.unit
@pytest.mark.parametrize("guide_mode", ["fixed", "completion"])
def test_hubbard_driver_smoke(guide_mode):
    driver = _build_driver(guide_mode=guide_mode)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "blocks.npz")
        driver.run(output=path, verbose=False)
        summary = driver.finalise(verbose=False)
        assert numpy.isfinite(summary["cyclic"]["mean"]["energy_per_site"])
        assert summary["max_recdev"] < 1e-10
        assert summary["max_wrapdev"] < 1e-10
        assert driver.results[0]["numer"].shape == (len(driver.observables.names),)
        if comm.rank == 0:
            with numpy.load(path) as saved:
                assert saved["numer"].shape[0] == 3
                assert saved["numer"].dtype == numpy.complex128


@pytest.mark.unit
@pytest.mark.parametrize("guide_mode", ["fixed", "completion"])
@pytest.mark.parametrize("wrap_stabilize_freq", [1, 5, 20])
def test_noninteracting_trace_and_spin_correlations(guide_mode, wrap_stabilize_freq):
    driver = _build_driver(U=0, guide_mode=guide_mode, wrap_stabilize_freq=wrap_stabilize_freq)
    driver.run(verbose=False)
    green = _exact_free_green(driver)
    exact = driver.observables.measure(numpy.array([green, green]))
    for row in driver.results:
        numpy.testing.assert_allclose(row["numer"] / row["denom"], exact, atol=2e-12)
        numpy.testing.assert_allclose(row["numer_endpoint"] / row["denom"], exact, atol=2e-12)


@pytest.mark.unit
def test_recording_and_cyclic_measurement_do_not_change_endpoint():
    endpoints = []
    for stride in (None, 2):
        driver = _build_driver(wrap_stride=stride)
        driver.run(verbose=False)
        endpoints.append(numpy.array([r["numer_endpoint"] / r["denom"] for r in driver.results]))
    numpy.testing.assert_array_equal(*endpoints)


@pytest.mark.unit
def test_density_tuner_uses_measured_density_and_positive_mu_convention():
    mu, history = tune_chemical_potential(lambda x: (1 - x / 2, 0), 0.875, (0, 1), tolerance=1e-10)
    assert mu == pytest.approx(0.25)
    assert history[-1]["density"] == pytest.approx(0.875)
    with pytest.raises(ValueError, match="bracket"):
        tune_chemical_potential(lambda x: (1.2, 0), 0.875, (0, 1))
    with pytest.raises(RuntimeError, match="noisy"):
        tune_chemical_potential(lambda x: (1 - x / 2, 0.02), 0.75, (0, 1), tolerance=0.001)


@pytest.mark.skipif(comm.size < 2, reason="Requires at least two MPI ranks.")
@pytest.mark.mpi
def test_hubbard_mpi_histories_and_raw_reduction():
    driver = _build_driver(nblocks=2)
    driver.propagator.energy_shift = -20.0
    driver.run(verbose=False)
    assert driver.finalise(verbose=False)["max_recdev"] < 1e-10
    # Every rank has the same globally reduced complex raw arrays.
    all_numer = comm.allgather(driver.results[0]["numer"])
    for numer in all_numer[1:]:
        numpy.testing.assert_array_equal(numer, all_numer[0])


@pytest.mark.skipif(comm.size < 2, reason="Requires at least two MPI ranks.")
@pytest.mark.mpi
def test_measurement_failure_is_synchronized_across_ranks(monkeypatch):
    driver = _build_driver(nblocks=2)

    def failing_wrap(*args, **kwargs):
        if comm.rank == 1:
            raise numpy.linalg.LinAlgError("injected rank-local failure")
        return cyclic_average_hubbard(*args, **kwargs)

    monkeypatch.setattr(
        "ipie.addons.thermal.thermofield.qmc.hubbard_afqmc.cyclic_average_hubbard", failing_wrap
    )
    with pytest.raises(RuntimeError, match="rank 1.*injected"):
        driver.run(verbose=False)


@pytest.mark.unit
def test_target_lattice_noninteracting_smoke():
    # A 12x12 lattice at beta = 4 exercises the 144-orbital path; one walker suffices in the
    # deterministic noninteracting limit.
    # System params.
    lx = ly = 12
    U = 0.0
    mu = 0.2

    # Thermofield AFQMC params.
    beta = 4.0
    timestep = 0.2
    nwalkers = 1
    energy_shift = -200.0

    ham, hop, _ = build_hubbard_hamiltonian(lx, ly, U, mu)
    trial = ThermofieldCPTrial(ham, beta, k_trial=ham.H1, guide_time=beta)
    mpi = MPIHandler()
    walkers = ThermofieldWalkers(trial, ham.nbasis, nwalkers, mpi)
    prop = ThermofieldHubbardCP(
        ham.H1,
        U,
        timestep,
        beta,
        guide_mode="fixed",
        fixed_guide_time=beta,
        energy_shift=energy_shift,
    )
    driver = ThermofieldHubbardAFQMC(
        ham, hop, trial, walkers, prop, mpi, lx, ly, nblocks=2, wrap_stride=10, pop_control_freq=1
    )
    driver.run(verbose=False)
    # The log determinants are of magnitude |energy_shift| beta ~ 800, so 1e-7 absolute is
    # ~1e-10 relative. The deviation is limited by BLAS rounding and depends on the thread
    # count (0.9e-8 with one thread, 1.2e-8 with four).
    assert driver.finalise(verbose=False)["max_recdev"] < 1e-7


@pytest.mark.driver
def test_target_hubbard_interacting_half_filling():
    # Half filling of the 12x12 lattice at U = 8 is exact by particle-hole symmetry. The tiny
    # population checks construction and numerics, not convergence of the interacting spin
    # correlations.
    # System params.
    lx = ly = 12
    U = 8.0
    mu = 0.0

    # Thermofield AFQMC params.
    beta = 4.0
    timestep = 0.01
    nwalkers = 2
    energy_shift = -648.0

    ham, hop, _ = build_hubbard_hamiltonian(lx, ly, U, mu)
    trial = ThermofieldCPTrial(ham, beta, k_trial=numpy.array([hop, hop]), guide_time=beta)
    mpi = MPIHandler()
    walkers = ThermofieldWalkers(trial, ham.nbasis, nwalkers, mpi)
    prop = ThermofieldHubbardCP(
        numpy.array([hop, hop]),
        U,
        timestep,
        beta,
        guide_mode="fixed",
        fixed_guide_time=beta,
        energy_shift=energy_shift,
    )
    driver = ThermofieldHubbardAFQMC(
        ham, hop, trial, walkers, prop, mpi, lx, ly, nblocks=2, wrap_stride=40
    )
    driver.run(verbose=False)
    summary = driver.finalise(verbose=False)
    assert abs(summary["cyclic"]["mean"]["density"] - 1) < 1e-9
    assert summary["mean_sign"] == pytest.approx(1, abs=1e-9)
    assert summary["max_recdev"] < 1e-7


@pytest.mark.unit
def test_paired_magnitude_only_measurement_retains_observable_signs():
    driver = _build_driver(wrap_stride=None)
    driver.measure_magnitude_only = True
    walkers = driver.walkers
    trace, greens = driver.trial.calc_log_overlap_and_greens_function_scaled_batched(
        walkers.Qmat, walkers.log_d, walkers.Tmat, 0.0
    )
    greens = greens.copy()
    for i, occupancy in enumerate([0.2, 0.4, 0.6, 0.8]):
        greens[i] = occupancy * numpy.eye(walkers.nbasis)
    magnitudes = numpy.array([1.0, 2.0, 3.0, 4.0])
    walkers.weight[:] = [1.0, 2.0, 1.0, 1.0]
    walkers.log_ovlp = trace - numpy.log(magnitudes)
    trace = trace.copy()
    trace[0] += 1j * numpy.pi

    def contraction(*args, **kwargs):
        return trace, greens

    driver.trial.calc_log_overlap_and_greens_function_scaled_batched = contraction
    row = driver._measure_block()
    values = numpy.array([driver.observables.measure(g) for g in greens])
    absolute = walkers.weight * magnitudes / magnitudes.max()
    signed = absolute * numpy.array([-1.0, 1.0, 1.0, 1.0])
    nranks = comm.size
    numpy.testing.assert_allclose(row["numer_endpoint"], nranks * (signed @ values), atol=1e-12)
    numpy.testing.assert_allclose(
        row["numer_endpoint_magnitude"], nranks * (absolute @ values), atol=1e-12
    )
    assert row["absolute_weight"] == pytest.approx(nranks * absolute.sum())
    assert row["denom"].real == pytest.approx(nranks * signed.sum())
    assert not numpy.isclose(
        row["numer_endpoint"][0].real / row["denom"].real,
        row["numer_endpoint_magnitude"][0].real / row["absolute_weight"],
    )


@pytest.mark.unit
def test_driver_refuses_numerically_limited_block():
    driver = _build_driver(nblocks=1)
    # A recorded limiter event must prevent publication of any block.
    driver.propagator.population_weight_bound_count = 1
    with pytest.raises(FloatingPointError, match="refusing block"):
        driver.run(verbose=False)
    assert driver.results == []


@pytest.mark.unit
@pytest.mark.parametrize("pop_frequency", [1, 3, 10, 100])
def test_retained_population_normalization_matches_exact_free_trace(pop_frequency):
    driver = _build_driver(U=0.0, nblocks=2, wrap_stride=None)
    driver.propagator.energy_shift = -20.0
    driver.preserve_population_normalization = True
    driver.params.pop_control_freq = pop_frequency
    beta = driver.propagator.beta
    initial_log = driver.trial.calc_log_overlap_and_greens_function_scaled_batched(
        driver.walkers.Qmat, driver.walkers.log_d, driver.walkers.Tmat, beta
    )[0][0].real
    energies = numpy.linalg.eigvalsh(driver.hamiltonian.H1[0])
    expected = 2 * numpy.logaddexp(0.0, -beta * energies).sum() - initial_log + beta * (-20.0)
    driver.run(verbose=False)
    population = driver.walkers.nwalkers * comm.size
    for row in driver.results:
        observed = numpy.log(abs(row["denom"])) + row["log_shift"] - numpy.log(population)
        assert observed == pytest.approx(expected, abs=3e-12)


@pytest.mark.unit
def test_one_rdm_smoke_and_checkpoint():
    driver = _build_driver(measure_one_rdm=True, nblocks=2)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "one_rdm.npz")
        driver.run(output=path, verbose=False)
        summary = driver.one_rdm_summary()
        assert summary["mean"].shape == (2, 16, 16)
        assert numpy.isfinite(summary["mean"]).all()
        assert numpy.isfinite(summary["error_real"]).all()
        # The density is an independent component of the spin observables.
        for row in driver.results:
            rho = row["one_rdm_numer"] / row["denom"]
            numpy.testing.assert_allclose(
                numpy.trace(rho, axis1=-2, axis2=-1).sum() / 16,
                row["numer"][1] / row["denom"],
                atol=2e-13,
            )
            copies = comm.allgather(row["one_rdm_numer"])
            for other in copies[1:]:
                numpy.testing.assert_array_equal(other, copies[0])
        if comm.rank == 0:
            with numpy.load(path, allow_pickle=False) as saved:
                assert saved["one_rdm_numer"].shape == (2, 2, 16, 16)
                assert saved["one_rdm_numer_endpoint"].shape == (2, 2, 16, 16)


@pytest.mark.unit
@pytest.mark.parametrize("wrap_stride", [None, 2])
def test_one_rdm_exact_thermal_density_and_passive_measurement(wrap_stride):
    driver = _build_driver(U=0, wrap_stride=wrap_stride, measure_one_rdm=True)
    driver.run(verbose=False)
    eps, c = numpy.linalg.eigh(driver.hamiltonian.H1)
    exact = numpy.einsum("sik,sk,sjk->sij", c, 1 / (1 + numpy.exp(driver.trial.beta * eps)), c)
    for estimator in ("cyclic", "endpoint"):
        summary = driver.one_rdm_summary(estimator=estimator)
        numpy.testing.assert_allclose(summary["mean"], exact, atol=2e-12)
        numpy.testing.assert_allclose(summary["error_real"], 0, atol=2e-13)
    control = _build_driver(U=0, wrap_stride=wrap_stride)
    control.run(verbose=False)
    for measured, original in zip(driver.results, control.results):
        for key in original:
            numpy.testing.assert_array_equal(measured[key], original[key])


@pytest.mark.unit
def test_one_rdm_signed_ratio_keeps_raw_nonhermitian_density():
    driver = _build_driver(measure_one_rdm=True, nblocks=2)
    a = numpy.zeros((2, 16, 16), dtype=complex)
    b = a.copy()
    a[:, 0, 1] = 2 + 3j
    b[:, 0, 1] = 5 + 1j
    driver.results = [
        dict(one_rdm_numer=2 * a, denom=2, log_shift=0),
        dict(one_rdm_numer=-b, denom=-1, log_shift=0),
    ]
    summary = driver.one_rdm_summary()
    numpy.testing.assert_allclose(summary["mean"], 2 * a - b)
    numpy.testing.assert_allclose(summary["error_real"], 0.5 * numpy.abs(a.real - b.real))
    numpy.testing.assert_allclose(summary["error_imag"], 0.5 * numpy.abs(a.imag - b.imag))
    assert summary["mean"][0, 1, 0] == 0
    with pytest.raises(ValueError, match="block_size"):
        driver.one_rdm_summary(block_size=3)


@pytest.mark.unit
def test_snapshots_reweight_to_mixed_estimator_and_leave_walk_unchanged():
    # Pinned 2x2 lattice with discrete fields and a fixed guide at tau_c.
    U, beta, dt, tau_c = 4.0, 0.4, 0.02, 0.2
    nslices = round(beta / dt)
    results = []
    for snapshots in ((), ((nslices, tau_c), (nslices // 2, 0.1))):
        numpy.random.seed(3)
        ham, hop, pin = build_hubbard_hamiltonian(
            2, 2, U, 0.5, pbc_x=False, pin_h=0.1, pin_edges=("left",)
        )
        k_trial, _, _ = rhf_k_trial(hop, pin, beta, 3.5)
        trial = ThermofieldCPTrial(ham, beta, k_trial=k_trial, guide_time=tau_c)
        mpi = MPIHandler()
        walkers = ThermofieldWalkers(trial, ham.nbasis, 8, mpi)
        prop = ThermofieldHubbardDiscrete(
            ham.H1 + U / 2 * numpy.eye(ham.nbasis),
            U,
            dt,
            beta,
            guide_mode="fixed",
            fixed_guide_time=tau_c,
            energy_shift=-4.0,
        )
        driver = ThermofieldHubbardAFQMC(
            ham,
            hop,
            trial,
            walkers,
            prop,
            mpi,
            2,
            2,
            nblocks=2,
            wrap_stride=None,
            seed=5,
            shell_distances_squared=(),
            snapshots=snapshots,
        )
        results.append(driver.run(verbose=False))
    # The measurements consume no random numbers and do not touch the walkers.
    for plain, snap in zip(*results):
        for key in ("numer", "denom", "numer_mixed", "denom_mixed"):
            numpy.testing.assert_array_equal(plain[key], snap[key])
        # With eta = tau_c at the last slice F = 1: the mixed estimator.
        assert snap["snap0_eta"] == tau_c
        numpy.testing.assert_allclose(
            snap["snap0_numer"] / snap["snap0_denom"],
            snap["numer_mixed"] / snap["denom_mixed"],
            rtol=1e-12,
        )
        assert numpy.isfinite(snap["snap1_numer"]).all() and snap["snap1_denom"] != 0


if __name__ == "__main__":
    test_hubbard_driver_smoke("fixed")
    test_noninteracting_trace_and_spin_correlations("fixed", 5)
    test_recording_and_cyclic_measurement_do_not_change_endpoint()
    test_density_tuner_uses_measured_density_and_positive_mu_convention()
    test_target_lattice_noninteracting_smoke()
    test_target_hubbard_interacting_half_filling()
    test_paired_magnitude_only_measurement_retains_observable_signs()
    test_driver_refuses_numerically_limited_block()
    test_retained_population_normalization_matches_exact_free_trace(3)
    test_one_rdm_smoke_and_checkpoint()
    test_one_rdm_exact_thermal_density_and_passive_measurement(2)
    test_one_rdm_signed_ratio_keeps_raw_nonhermitian_density()
    test_snapshots_reweight_to_mixed_estimator_and_leave_walk_unchanged()
    if comm.size > 1:
        test_hubbard_mpi_histories_and_raw_reduction(True, "trace")
        with pytest.MonkeyPatch.context() as patcher:
            test_measurement_failure_is_synchronized_across_ranks(patcher)
