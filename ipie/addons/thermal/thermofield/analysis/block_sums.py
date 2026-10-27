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
from dataclasses import dataclass
from pathlib import Path

import numpy


@dataclass
class BlockSums:
    """Complex block sums of several runs on a common scale.

    Attributes
    ----------
    numer, denom, navnum : :class:`numpy.ndarray`
        Energy numerator, denominator and particle-number numerator per block.
    neff : :class:`numpy.ndarray`
        Effective walker fraction per block (NaN when not recorded).
    run_index : :class:`numpy.ndarray`
        Run of every block.
    metadata : list
        Run metadata.
    paths : list
        Run files.
    normalization : str
        "global" or "legacy-per-run".
    """

    numer: numpy.ndarray
    denom: numpy.ndarray
    navnum: numpy.ndarray
    neff: numpy.ndarray
    run_index: numpy.ndarray
    metadata: list
    paths: list
    normalization: str


def _signature(d):
    """Settings that must agree between runs that are combined."""
    return dict(
        rs=d.get("rs"),
        theta=d.get("theta"),
        mu=d.get("mu_run", d.get("mu")),
        guide=d.get("guide", "fe"),
        guide_mode=d.get("guide_mode", "fixed"),
        trace_branching=d.get("trace_branching", False),
        walkers=d["nwalkers"] * d.get("nranks", 1),
        **{
            k: d.get(k)
            for k in (
                "M",
                "nel",
                "beta",
                "dt",
                "nslices",
                "mu_T",
                "stabilize_freq",
                "pop_control_freq",
                "wbound_frac",
                "free_projection_until",
                "free_projection_from",
                "free_projection_window",
                "dt_scale",
            )
        },
    )


def load_block_sums(paths, normalization="global", restore_population=False):
    """Read the block sums of compatible runs from their JSON files.

    Runs that record two endpoint estimators contribute their endpoint "A"
    sums. Seeds must be distinct, so that no sample is counted twice.

    Parameters
    ----------
    paths : list
        JSON files.
    normalization : str
        "global": one common log scale for all blocks (ratio of means);
        "legacy-per-run": each run normalized separately, only to reproduce
        old reports. Default "global".
    restore_population : bool
        Add the recorded population-control normalization of every block to
        its log scale; files without it are refused. Default False.

    Returns
    -------
    blocks : :class:`BlockSums`
        Block sums on a common scale.
    """
    if normalization not in ("global", "legacy-per-run"):
        raise ValueError("Unknown normalization.")
    paths = [Path(p).resolve() for p in paths]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("Provide at least one file, without duplicate paths.")
    metadata, arrays, shifts, neff, indices = [], [], [], [], []
    seeds = set()
    for irun, path in enumerate(paths):
        d = json.loads(path.read_text())
        if metadata and _signature(d) != _signature(metadata[0]):
            diff = [k for k, v in _signature(d).items() if v != _signature(metadata[0])[k]]
            raise ValueError(f"Incompatible runs ({', '.join(diff)}): {path}")
        if d.get("seed") is not None:
            if d["seed"] in seeds:
                raise ValueError(
                    f"Duplicate seed {d['seed']} in {path}; possible duplicated samples"
                )
            seeds.add(d["seed"])
        shift = numpy.asarray(d["logshift"], dtype=float)
        nb = len(shift)
        if nb < 1 or d.get("blocks_done", nb) != nb:
            raise ValueError(f"Invalid block count: {path}")
        if restore_population:
            if "pop_log_norm" not in d or len(d["pop_log_norm"]) != nb:
                raise ValueError(f"Missing pop_log_norm: {path}")
            shift = shift + numpy.asarray(d["pop_log_norm"])
        suffix = "A" if "numerA_re" in d else ""
        values = []
        for name in ("numer" + suffix, "denom", "navnum" + suffix):
            value = numpy.asarray(d[name + "_re"]) + 1j * numpy.asarray(d[name + "_im"])
            if value.shape != (nb,) or not numpy.all(numpy.isfinite(value)):
                raise ValueError(f"Invalid {name}: {path}")
            values.append(value)
        if not numpy.all(numpy.isfinite(shift)):
            raise ValueError(f"Nonfinite logshift: {path}")
        eff = numpy.asarray(d.get("neff_blk", [numpy.nan] * nb))
        if eff.shape != (nb,):
            raise ValueError(f"Invalid neff_blk: {path}")
        metadata.append(d)
        arrays.append(numpy.asarray(values))
        shifts.append(shift)
        neff.append(eff)
        indices.append(numpy.full(nb, irun))
    common = max(s.max() for s in shifts)
    for i, (a, s) in enumerate(zip(arrays, shifts)):
        if normalization == "legacy-per-run":
            a = a * numpy.exp(s - s.max())
            scale = numpy.abs(a[1]).mean()
            if scale == 0:
                raise ValueError("Zero denominator mass in a run.")
            arrays[i] = a / scale
        else:
            arrays[i] = a * numpy.exp(s - common)
    numer, denom, navnum = numpy.concatenate(arrays, axis=1)
    return BlockSums(
        numer,
        denom,
        navnum,
        numpy.concatenate(neff),
        numpy.concatenate(indices),
        metadata,
        [str(p) for p in paths],
        normalization,
    )


def _ratio(numer, denom):
    """Real part of a complex ratio; refuses zero denominators and non-finite results."""
    if numpy.any(numpy.abs(denom) == 0):
        raise ValueError("Zero complex denominator; ratio uncertainty is unresolved.")
    value = numer / denom
    if not numpy.all(numpy.isfinite(value)):
        raise ValueError("Nonfinite complex ratio.")
    return value.real


def summarize_blocks(blocks, nboot=4000, seed=1729, winsor_factor=5.0):
    """Ratio of means with paired block bootstrap and jackknife errors.

    Blocks are independent walks over [0, beta], so they are the resampling
    units; complex sums are kept throughout, and the bootstrap samples keep
    the energy-number covariance for later fits. The winsorized estimate is a
    biased diagnostic only.

    Parameters
    ----------
    blocks : :class:`BlockSums`
        Block sums.
    nboot : int
        Bootstrap replicates. Default 4000.
    seed : int
        Seed of the bootstrap. Default 1729.
    winsor_factor : float
        Blocks heavier than this multiple of the median mass are clipped in
        the diagnostic. Default 5.0.

    Returns
    -------
    summary : dict
        Estimates, errors, bootstrap samples and diagnostics.
    """
    d = blocks.denom
    x = numpy.column_stack((blocks.numer, blocks.navnum))
    nb = len(d)
    if nb < 2 or nboot < 2 or winsor_factor <= 0:
        raise ValueError("Need at least two blocks and bootstrap draws and a positive cap.")
    total_x, total_d = x.sum(axis=0), d.sum()
    estimate = _ratio(total_x, total_d)
    individual = _ratio(x, d[:, None])
    mean = individual.mean(axis=0)
    jk = _ratio(total_x - x, (total_d - d)[:, None])
    centered = jk - jk.mean(axis=0)
    jk_cov = (nb - 1) / nb * centered.T @ centered
    # A private generator: the bootstrap must not move the QMC random stream.
    rng = numpy.random.default_rng(seed)
    draws, differences = [], []
    for start in range(0, nboot, 128):
        ids = rng.integers(nb, size=(min(128, nboot - start), nb))
        draw = _ratio(x[ids].sum(axis=1), d[ids].sum(axis=1)[:, None])
        draws.append(draw)
        differences.append(draw - individual[ids].mean(axis=1))
    draws = numpy.concatenate(draws)
    differences = numpy.concatenate(differences)
    boot_cov = numpy.cov(draws.T, ddof=1)
    boot_err = numpy.sqrt(numpy.diag(boot_cov))
    jk_err = numpy.sqrt(numpy.diag(jk_cov))
    mass = numpy.abs(d)
    mass_fraction = mass / mass.sum()
    neff = mass.sum() ** 2 / numpy.sum(mass**2)
    coherence = abs(total_d) / mass.sum()
    factors = numpy.minimum(1.0, winsor_factor * numpy.median(mass) / mass)
    clipped = _ratio((x * factors[:, None]).sum(axis=0), numpy.sum(d * factors))
    endpoints = numpy.unique(numpy.linspace(2, nb, min(200, nb - 1), dtype=int))
    cumulative = _ratio(
        numpy.cumsum(x, axis=0)[endpoints - 1], numpy.cumsum(d)[endpoints - 1, None]
    )
    run_results = []
    for i, path in enumerate(blocks.paths):
        selected = blocks.run_index == i
        run_mean = _ratio(x[selected].sum(axis=0), d[selected].sum())
        row = dict(
            path=path,
            seed=blocks.metadata[i].get("seed"),
            blocks=int(selected.sum()),
            energy=float(run_mean[0]),
            nav=float(run_mean[1]),
        )
        if len(blocks.paths) > 1:
            left = _ratio(x[~selected].sum(axis=0), d[~selected].sum())
            row.update(leave_out_energy=float(left[0]), leave_out_nav=float(left[1]))
        run_results.append(row)
    finite_eff = blocks.neff[numpy.isfinite(blocks.neff)]
    walkers = _signature(blocks.metadata[0])["walkers"]
    return dict(
        energy=float(estimate[0]),
        nav=float(estimate[1]),
        blocks=nb,
        runs=len(blocks.paths),
        walkers=walkers,
        normalization=blocks.normalization,
        jackknife_error=float(jk_err[0]),
        nav_jackknife_error=float(jk_err[1]),
        bootstrap_error=float(boot_err[0]),
        nav_bootstrap_error=float(boot_err[1]),
        energy_error=float(max(jk_err[0], boot_err[0])),
        nav_error=float(max(jk_err[1], boot_err[1])),
        bootstrap_ci95=numpy.percentile(draws, [2.5, 97.5], axis=0).T.tolist(),
        bootstrap_covariance=boot_cov.tolist(),
        bootstrap_samples=draws.tolist(),
        bootstrap_seed=seed,
        bootstrap_replicates=nboot,
        block_mean_energy=float(mean[0]),
        block_mean_nav=float(mean[1]),
        block_mean_error=(individual.std(axis=0, ddof=1) / numpy.sqrt(nb)).tolist(),
        block_std=individual.std(axis=0, ddof=1).tolist(),
        pooled_minus_block_mean=(estimate - mean).tolist(),
        pooled_minus_block_mean_error=differences.std(axis=0, ddof=1).tolist(),
        effective_blocks=float(neff),
        denominator_phase_coherence=float(coherence),
        largest_block_fraction=float(mass_fraction.max()),
        top_five_block_fraction=float(numpy.sort(mass_fraction)[-5:].sum()),
        mean_neff=float(finite_eff.mean()) if len(finite_eff) else None,
        median_effective_walkers=(
            float(numpy.median(finite_eff) * walkers) if len(finite_eff) else None
        ),
        winsor_diagnostic=dict(
            factor=winsor_factor,
            biased=True,
            energy=float(clipped[0]),
            nav=float(clipped[1]),
            shift=(clipped - estimate).tolist(),
            clipped_blocks=int(numpy.count_nonzero(factors < 1)),
        ),
        convergence=dict(
            blocks=endpoints.tolist(),
            energy=cumulative[:, 0].tolist(),
            nav=cumulative[:, 1].tolist(),
            order="sorted files, original block order",
        ),
        denominator_magnitude_over_median=(mass / numpy.median(mass)).tolist(),
        per_run=run_results,
    )


def fit_mu_scan(rows, target=66.0):
    """Energy at a target particle number from a scan of the chemical potential.

    Straight lines N(mu) and E(mu) with fixed inverse-variance weights; every
    paired bootstrap draw refits both lines and evaluates E at its own root of
    N(mu) = target, so the measured N is never treated as a noiseless regressor.

    Parameters
    ----------
    rows : list
        Results of :func:`summarize_blocks` with the chemical potential "mu".
    target : float
        Target particle number. Default 66.0.

    Returns
    -------
    fit : dict
        Root mu_star, slopes, the energy at the target and bootstrap samples.
    """
    if len(rows) < 2:
        raise ValueError("Need at least two chemical potentials.")
    mu = numpy.array([r["mu"] for r in rows])
    center = mu.mean()
    design = numpy.column_stack((numpy.ones(len(mu)), mu - center))
    if numpy.linalg.matrix_rank(design) != 2:
        raise ValueError("Chemical potentials must span a nonzero interval.")
    estimates = numpy.array([[r["energy"], r["nav"]] for r in rows])
    errors = numpy.array([[r["energy_error"], r["nav_error"]] for r in rows])
    if not numpy.all(numpy.isfinite(errors)) or numpy.any(errors <= 0):
        raise ValueError("Fit needs finite positive uncertainties.")
    samples = numpy.array([r["bootstrap_samples"] for r in rows])
    params, replicas = [], []
    for col in range(2):
        w = 1 / errors[:, col] ** 2
        transform = numpy.linalg.solve(design.T @ (w[:, None] * design), design.T * w)
        params.append(transform @ estimates[:, col])
        replicas.append(transform @ samples[:, :, col])
    e, n = params
    eb, nb = replicas
    if n[1] <= 0 or numpy.any(nb[1] <= 0):
        raise ValueError("N(mu) slope is not resolved positive; extend the scan or the statistics.")
    offset = (target - n[0]) / n[1]
    offsets = (target - nb[0]) / nb[1]
    root = center + offset
    root_samples = center + offsets
    energy = e[0] + e[1] * offset
    energy_samples = eb[0] + eb[1] * offsets
    slope_samples = eb[1] / nb[1]
    chi2 = numpy.sum(((estimates[:, 1] - design @ n) / errors[:, 1]) ** 2)
    return dict(
        target=target,
        mu_star=float(root),
        mu_star_error=float(root_samples.std(ddof=1)),
        mu_star_ci95=numpy.percentile(root_samples, [2.5, 97.5]).tolist(),
        dN_dmu=float(n[1]),
        dN_dmu_error=float(nb[1].std(ddof=1)),
        energy_at_target=float(energy),
        energy_at_target_error=float(energy_samples.std(ddof=1)),
        dE_dN=float(e[1] / n[1]),
        dE_dN_error=float(slope_samples.std(ddof=1)),
        bracketed=bool(mu.min() <= root <= mu.max()),
        bootstrap_bracket_fraction=float(
            numpy.mean((root_samples >= mu.min()) & (root_samples <= mu.max()))
        ),
        chi2=float(chi2),
        dof=len(rows) - 2,
        mu_star_samples=root_samples.tolist(),
        energy_at_target_samples=energy_samples.tolist(),
        dE_dN_samples=slope_samples.tolist(),
    )
