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

from ipie.addons.thermal.thermofield.estimators.hubbard import HubbardSpinObservables
from ipie.addons.thermal.thermofield.trial.thermofield import ThermofieldThermalTrial
from ipie.addons.thermal.thermofield.utils.linalg import (
    principal_log_phase,
    stabilized_inverse_one_plus_batched,
)
from ipie.utils.backend import arraylib as xp


class ThermofieldCPTrial(ThermofieldThermalTrial):
    r"""Real thermofield guide for constrained-path Hubbard propagation.

    The guide time is chosen per call (completion or fixed guides), and the
    batched contraction can reuse cached inverses of the walker Q factors.

    Parameters
    ----------
    hamiltonian :
        Real Hamiltonian; sets the default :math:`k_T`.
    beta : float
        Inverse temperature.
    mu : float
        Chemical potential. Default 0.0.
    k_trial : :class:`numpy.ndarray`
        Real one-body matrix :math:`k_T`, shape (2, nbasis, nbasis). Default
        :math:`H_1 - \mu`.
    guide_time : float
        Default guide time. Default beta / 2.
    """

    def __init__(self, hamiltonian, beta, mu=0.0, k_trial=None, guide_time=None):
        if numpy.any(numpy.iscomplex(hamiltonian.H1)):
            raise ValueError("The spin-channel constrained-path Hamiltonian must be real.")
        if k_trial is not None and numpy.any(numpy.iscomplex(k_trial)):
            raise ValueError("The spin-channel constrained-path trial must be real.")
        if not numpy.isfinite(beta) or beta <= 0:
            raise ValueError("beta must be finite and positive.")
        super().__init__(hamiltonian, beta, mu, k_trial=k_trial, guide_time=guide_time)

    def calc_log_overlap_and_greens_function_scaled(self, Qmat, log_d=None, Tmat=None, scale=None):
        """Guide log overlap and Green's function of one walker at guide time `scale`.

        Parameters
        ----------
        Qmat : :class:`numpy.ndarray`
            Walker Q factors, shape (2, nbasis, nbasis).
        log_d : :class:`numpy.ndarray`
            Log scales, shape (2, nbasis). Default zero.
        Tmat : :class:`numpy.ndarray`
            Walker T factors, shape (2, nbasis, nbasis). Default identity.
        scale : float
            Guide time. Default the fixed guide time.

        Returns
        -------
        log_ovlp : complex
            Log overlap.
        G : :class:`numpy.ndarray`
            Green's function in the particle convention, shape (2, nbasis, nbasis).
        """
        if scale is None:
            scale = self.guide_time
        return self.calc_log_overlap_and_greens_function(Qmat, log_d, Tmat, guide_time=scale)

    def calc_log_overlap_and_greens_function_scaled_batched(
        self, Qmat, log_d, Tmat, scale, Qinv=None, log_det_Q=None
    ):
        r"""Guide log overlaps and Green's functions of all walkers at guide time `scale`.

        :math:`\Delta\Lambda = Q e^{D} (T C) e^{-\tau_g \epsilon} C^\dagger` through the
        two-sided kernel. A singular walker gives non-finite entries; callers
        must kill it.

        Parameters
        ----------
        Qmat : :class:`numpy.ndarray`
            Walker Q factors, shape (nwalkers, 2, nbasis, nbasis).
        log_d : :class:`numpy.ndarray`
            Log scales, shape (nwalkers, 2, nbasis).
        Tmat : :class:`numpy.ndarray`
            Walker T factors, shape (nwalkers, 2, nbasis, nbasis).
        scale : float
            Guide time.
        Qinv : :class:`numpy.ndarray`
            Cached inverses of the Q factors, shape (nwalkers, 2, nbasis, nbasis).
            Saves one stacked inversion per spin. Default None.
        log_det_Q : :class:`numpy.ndarray`
            Cached log determinants of the Q factors, shape (nwalkers, 2).
            Default None.

        Returns
        -------
        log_ovlp : :class:`numpy.ndarray`
            Log overlaps, shape (nwalkers,).
        G : :class:`numpy.ndarray`
            Green's functions in the particle convention, shape
            (nwalkers, 2, nbasis, nbasis).
        """
        Qmat, log_d, Tmat = xp.asarray(Qmat), xp.asarray(log_d), xp.asarray(Tmat)
        nwalkers, nbasis = Qmat.shape[0], self.nbasis
        log_ovlp = xp.zeros(nwalkers, dtype=xp.complex128)
        G = xp.empty((nwalkers, 2, nbasis, nbasis), dtype=xp.complex128)
        I = xp.eye(nbasis)
        C_T, _ = self._guide_arrays(xp)
        for s in range(2):
            a = xp.asarray(scale * self.eps_T[s])
            C = C_T[s]
            X = Tmat[:, s] @ C
            log_det, inv = stabilized_inverse_one_plus_batched(
                Qmat[:, s],
                log_d[:, s],
                X,
                -a[None, :],
                Vinv=C,
                log_det_V=self.log_det_C_T_dagger[s],
                Qinv=None if Qinv is None else Qinv[:, s],
                log_det_Q=None if log_det_Q is None else log_det_Q[:, s],
            )
            log_ovlp += log_det
            G[:, s] = I - inv
        return principal_log_phase(log_ovlp), G


def _tune_level_shift(eps, beta, target_n):
    """Shift of the levels `eps` that gives `target_n` particles, by bisection.

    Returns
    -------
    mu_eff : float
        Shift, entering as eps + mu_eff.
    occ : :class:`numpy.ndarray`
        Fermi occupations at that shift.
    """
    margin = max(8.0, 40.0 / beta)
    lo, hi = -eps.max() - margin, -eps.min() + margin
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        occ = 0.5 * (1 - numpy.tanh(0.5 * beta * (eps + mid)))
        if occ.sum() > target_n:
            lo = mid
        else:
            hi = mid
    mu_eff = 0.5 * (lo + hi)
    occ = 0.5 * (1 - numpy.tanh(0.5 * beta * (eps + mu_eff)))
    return mu_eff, occ


def thermal_ip_scf(hop, v, beta, target_n, u_eff, dens0=None, mix=0.5, tol=1e-10, max_iter=500):
    r"""Collinear thermal Hartree-Fock trial at a given total particle number.

    Self-consistent independent-particle Hamiltonian
    :math:`k_\sigma = h + v_\sigma + U_{eff} (n_{\bar\sigma} - 1/2) + \mu_T`, with
    :math:`\mu_T` tuned to `target_n` at every iteration; the chemical potential
    enters as :math:`+\mu_T N`, so a larger :math:`\mu_T` lowers the filling.

    Parameters
    ----------
    hop : :class:`numpy.ndarray`
        Real symmetric hopping matrix, shape (nsites, nsites).
    v : :class:`numpy.ndarray`
        Spin-dependent on-site potential, shape (2, nsites).
    beta : float
        Inverse temperature.
    target_n : float
        Total particle number (not the filling per site).
    u_eff : float
        Effective interaction of the mean field.
    dens0 : :class:`numpy.ndarray`
        Initial site densities, shape (2, nsites). Default a checkerboard seed
        on a square lattice; required on other geometries.
    mix : float
        Linear mixing of the densities. Default 0.5.
    tol : float
        Convergence threshold on the densities. Default 1e-10.
    max_iter : int
        Maximum number of iterations. Default 500.

    Returns
    -------
    k_trial : :class:`numpy.ndarray`
        One-body matrix of the trial with :math:`\mu_T` included, shape
        (2, nsites, nsites).
    density : :class:`numpy.ndarray`
        Site densities, shape (2, nsites).
    mu_T : float
        Chemical potential of the trial.
    """
    hop = numpy.asarray(hop)
    v = numpy.asarray(v)
    if (
        hop.ndim != 2
        or hop.shape[0] != hop.shape[1]
        or hop.shape[0] == 0
        or numpy.any(numpy.iscomplex(hop))
        or not numpy.isfinite(hop).all()
        or not numpy.allclose(hop, hop.T, atol=1e-12)
    ):
        raise ValueError("hop must be a finite real symmetric matrix.")
    n = hop.shape[0]
    if v.shape != (2, n) or not numpy.isfinite(v).all() or numpy.any(numpy.iscomplex(v)):
        raise ValueError("v must be a finite real array with shape (2, nsites).")
    if not numpy.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive.")
    if not numpy.isfinite(target_n) or not 0 < target_n < 2 * n:
        raise ValueError(
            "target_n is the total particle number and must be between 0 and 2*nsites."
        )
    if not numpy.isfinite(u_eff) or u_eff < 0:
        raise ValueError("u_eff must be finite and non-negative.")
    if not numpy.isfinite(mix) or not 0 < mix <= 1 or not numpy.isfinite(tol) or tol <= 0:
        raise ValueError("mix must be in (0, 1] and tol positive.")
    if not isinstance(max_iter, (int, numpy.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer.")
    dens = (
        numpy.full((2, n), target_n / (2.0 * n))
        if dens0 is None
        else numpy.array(dens0, dtype=float, copy=True)
    )
    if dens.shape != (2, n) or not numpy.isfinite(dens).all():
        raise ValueError("dens0 must be finite with shape (2, nsites).")
    if dens0 is None and u_eff > 0:
        lx = int(round(numpy.sqrt(n)))
        if lx * lx != n:
            raise ValueError("A non-square lattice requires an explicit thermal-UHF dens0 seed.")
        sites = numpy.arange(n)
        staggered = min(0.1, target_n / (2 * n), 1 - target_n / (2 * n)) * (-1.0) ** (
            sites % lx + sites // lx
        )
        dens[0] += staggered
        dens[1] -= staggered

    def k_of(d):
        return numpy.array(
            [
                hop + numpy.diag(u_eff * (d[1] - 0.5) + v[0]),
                hop + numpy.diag(u_eff * (d[0] - 0.5) + v[1]),
            ]
        )

    def solve_density(k):
        # A scalar chemical potential only shifts the eigenvalues, so
        # diagonalize once for the whole bisection.
        eps, coeff = numpy.linalg.eigh(k)
        mu_eff, occ = _tune_level_shift(eps, beta, target_n)
        density = numpy.einsum("sik,sk,sik->si", coeff, occ, coeff.conj()).real
        return density, mu_eff

    for _ in range(max_iter):
        k = k_of(dens)
        new, mu_eff = solve_density(k)
        delta = numpy.max(numpy.abs(new - dens))
        if delta < tol:
            return k + mu_eff * numpy.eye(n), new, mu_eff
        dens = (1 - mix) * dens + mix * new
    raise RuntimeError(f"Thermal IP SCF not converged (delta={delta:g}).")


def thermal_uhf_scf(
    hop, v, beta, target_n, u_eff, lx, ly, seed=17, random_starts=1, scf_options=None
):
    """Lowest free energy collinear thermal UHF solution from several starts.

    The starts are Neel, vertical and horizontal antiphase textures and
    reproducible random spin textures; they change the initial trial density
    only, never the physical pinning field.

    Parameters
    ----------
    hop, v, beta, target_n :
        As in :func:`thermal_ip_scf`.
    u_eff : float
        Effective interaction, positive.
    lx, ly : int
        Lattice dimensions.
    seed : int
        Seed of the random textures. Default 17.
    random_starts : int
        Number of random textures. Default 1.
    scf_options : dict
        `mix`, `tol` and `max_iter` of :func:`thermal_ip_scf`. Default None.

    Returns
    -------
    k_trial : :class:`numpy.ndarray`
        One-body matrix of the trial, shape (2, nsites, nsites).
    density : :class:`numpy.ndarray`
        Site densities, shape (2, nsites).
    mu_T : float
        Chemical potential of the trial.
    info : dict
        Selected start, free energies and convergence of all starts.
    """
    if not numpy.isfinite(u_eff) or u_eff <= 0:
        raise ValueError("A Hubbard UHF trial requires positive U_eff.")
    if int(lx) != lx or int(ly) != ly or lx < 1 or ly < 1 or numpy.shape(hop) != (lx * ly, lx * ly):
        raise ValueError("Trial lattice dimensions must match hop.")
    if int(random_starts) != random_starts or random_starts < 0:
        raise ValueError("random_starts must be a non-negative integer.")
    options = dict(scf_options or {})
    if not set(options) <= {"mix", "tol", "max_iter"}:
        raise ValueError("scf_options may specify mix, tol, and max_iter only.")
    n = lx * ly
    filling = target_n / n
    amplitude = min(0.15, filling / 2, 1 - filling / 2)
    x, y = numpy.arange(n) % lx, numpy.arange(n) // lx
    neel = (-1.0) ** (x + y)
    textures = [
        ("neel", neel),
        ("vertical_antiphase", neel * numpy.cos(2 * numpy.pi * x / lx)),
        ("horizontal_antiphase", neel * numpy.cos(2 * numpy.pi * y / ly)),
    ]
    # A private generator: trial construction must not move the QMC random stream.
    rng = numpy.random.default_rng(seed)
    textures += [(f"random_{i}", rng.uniform(-1, 1, n)) for i in range(random_starts)]
    candidates, best = [], None
    for name, texture in textures:
        initial = numpy.array(
            [filling / 2 + amplitude * texture, filling / 2 - amplitude * texture]
        )
        entry = {"start": name}
        try:
            k, density, mu = thermal_ip_scf(hop, v, beta, target_n, u_eff, dens0=initial, **options)
            eps, coeff = numpy.linalg.eigh(k)
            occupation = 0.5 * (1 - numpy.tanh(0.5 * beta * eps))
            green = numpy.einsum("sik,sk,sjk->sij", coeff, occupation, coeff.conj())
            energy = (
                numpy.einsum("ij,sji->", hop, green).real
                + numpy.sum(v * density)
                + u_eff * numpy.sum(density[0] * density[1] - 0.5 * density.sum(axis=0))
            )
            f = occupation[(occupation > 0) & (occupation < 1)]
            entropy = -numpy.sum(f * numpy.log(f) + (1 - f) * numpy.log1p(-f))
            free_energy = float(energy - entropy / beta)
            if not numpy.isfinite(free_energy):
                raise FloatingPointError("Nonfinite UHF free energy.")
        except (RuntimeError, numpy.linalg.LinAlgError, FloatingPointError) as exc:
            entry.update(status="failed", reason=str(exc))
            candidates.append(entry)
            continue
        entry.update(
            status="ok",
            free_energy=free_energy,
            max_local_magnetization=float(numpy.max(numpy.abs(density[0] - density[1]))),
        )
        candidates.append(entry)
        if best is None or free_energy < best[3]["free_energy"]:
            best = (k, density, mu, entry)
    if best is None:
        raise RuntimeError(
            "All thermal-UHF starts failed: " + "; ".join(c["reason"] for c in candidates)
        )
    k, density, mu, chosen = best
    info = dict(
        trial_type="thermal_uhf",
        u_eff=float(u_eff),
        beta_eff=float(beta),
        seed=int(seed),
        selected_start=chosen["start"],
        free_energy=chosen["free_energy"],
        max_local_magnetization=chosen["max_local_magnetization"],
        candidates=candidates,
    )
    return k, density, mu, info


def rhf_k_trial(hop, v, beta, target_n):
    """Noninteracting trial tuned to the total particle number `target_n`.

    :func:`thermal_ip_scf` with U_eff = 0; same arguments and returns.
    """
    return thermal_ip_scf(hop, v, beta, target_n, u_eff=0.0)


def fit_hubbard_density_trial(hop, v, beta, target_n, qmc_densities, u_eff_grid):
    r"""Trial from QMC densities at the physical beta (He et al., PRB 99, Eqs. 19-20).

    With the QMC site densities fixed, every :math:`U_{eff}` on the grid gives
    :math:`k_\sigma = h + v_\sigma + U_{eff} (n^{QMC}_{\bar\sigma} - 1/2) + \mu_{eff}`
    with :math:`\mu_{eff}` tuned to `target_n`; the one whose Fermi-Dirac
    densities are closest to the input (root mean square) is selected. No
    inner SCF and no fit of beta or spin correlations. A uniform
    spin-symmetric input without pinning cannot identify :math:`U_{eff}`; the
    spread of the objective is returned so that the caller can refuse such a fit.

    Parameters
    ----------
    hop : :class:`numpy.ndarray`
        Real symmetric hopping matrix, shape (nsites, nsites).
    v : :class:`numpy.ndarray`
        Pinning potential, shape (2, nsites).
    beta : float
        Physical inverse temperature.
    target_n : float
        Total particle number.
    qmc_densities : :class:`numpy.ndarray`
        QMC site densities, shape (2, nsites).
    u_eff_grid :
        Positive values of :math:`U_{eff}` to scan.

    Returns
    -------
    best : dict
        `k_trial`, `densities`, `u_eff`, `mu_eff`, `chi`, all `candidates`
        and `metadata`.
    """
    hop, v = numpy.asarray(hop), numpy.asarray(v)
    density = numpy.asarray(qmc_densities)
    if (
        hop.ndim != 2
        or hop.shape[0] == 0
        or hop.shape[0] != hop.shape[1]
        or not numpy.isfinite(hop).all()
        or numpy.any(numpy.iscomplex(hop))
        or not numpy.allclose(hop, hop.T, rtol=0, atol=1e-12)
    ):
        raise ValueError("hop must be a finite real symmetric matrix")
    n = hop.shape[0]
    if (
        v.shape != (2, n)
        or density.shape != (2, n)
        or not numpy.isfinite(v).all()
        or not numpy.isfinite(density).all()
        or numpy.any(numpy.iscomplex(v))
        or numpy.any(numpy.iscomplex(density))
    ):
        raise ValueError("Pinning and QMC densities must be finite real (2,M) arrays")
    if not numpy.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    if not numpy.isfinite(target_n) or not 0 < target_n < 2 * n:
        raise ValueError("target_n must be between zero and 2*M")
    grid = numpy.asarray(tuple(u_eff_grid), dtype=float)
    if grid.ndim != 1 or not len(grid) or not numpy.isfinite(grid).all() or numpy.any(grid <= 0):
        raise ValueError("The U_eff grid must be finite, nonempty and strictly positive")
    candidates, best = [], None
    for u_eff in grid:
        potential = v + u_eff * (density[::-1] - 0.5)
        k = numpy.array([hop + numpy.diag(potential[spin]) for spin in range(2)])
        eps, coeff = numpy.linalg.eigh(k)
        mu_eff, occupations = _tune_level_shift(eps, beta, target_n)
        predicted = numpy.einsum("sik,sk,sik->si", coeff, occupations, coeff).real
        chi = float(numpy.sqrt(numpy.mean((predicted - density) ** 2)))
        if not numpy.isfinite(chi):
            raise FloatingPointError("Nonfinite IP-density objective")
        entry = dict(u_eff=float(u_eff), mu_eff=float(mu_eff), chi=chi)
        candidates.append(entry)
        if best is None or chi < best["chi"]:
            best = dict(
                k_trial=k + mu_eff * numpy.eye(n),
                densities=predicted,
                input_densities=density.copy(),
                **entry,
            )
    chis = numpy.array([c["chi"] for c in candidates])
    best["candidates"] = candidates
    best["metadata"] = dict(
        method="PRB99.045108 Eqs19-20; fixed QMC densities; physical beta",
        physical_beta=float(beta),
        target_particles=float(target_n),
        objective="sqrt(mean((n_IP-n_QMC)**2)) over spin and sites",
        objective_spread=float(numpy.ptp(chis)),
        grid_boundary=bool(best["u_eff"] in (grid.min(), grid.max())),
        density_input_range=[float(density.min()), float(density.max())],
        beta_eff_fitted=False,
        inner_uhf_scf=False,
    )
    return best


def fit_hubbard_trial(
    hop,
    v,
    beta,
    target_n,
    lx,
    ly,
    target_czz,
    u_eff_grid,
    beta_eff_grid,
    errors=None,
    dens0=None,
    spin_flip_average=False,
    scf_options=None,
    uhf_options=None,
):
    r"""Thermal Hartree-Fock trial fitted to measured connected spin correlations.

    Every (:math:`U_{eff}`, :math:`\beta_{eff}`) of the grid gives a thermal HF
    trial whose connected :math:`C_{zz}` per distance shell, with the same
    ensemble subtraction as the QMC estimator, is compared with the target;
    the objective is the mean squared residual over the shells, divided by
    the errors when given. The returned trial is
    :math:`k_T = (\beta_{eff}/\beta) k_{HF}`, so a guide built at the physical
    beta keeps the fitted density operator. Call again with new QMC data for
    an outer self-consistency loop.

    A collinear Gaussian has non-positive connected off-site :math:`C_{zz}` and
    cannot match positive antiferromagnetic even shells; with
    `spin_flip_average` the objective uses the equal mixture of the trial and
    its global spin flip instead, which restores spin-flip (not SU(2))
    symmetry of the fitting objective only; the returned guide stays collinear.

    Parameters
    ----------
    hop, v, beta, target_n :
        As in :func:`thermal_ip_scf`; beta is the physical inverse temperature.
    lx, ly : int
        Lattice dimensions.
    target_czz : dict
        Connected :math:`C_{zz}` keyed by the squared shell distance.
    u_eff_grid, beta_eff_grid :
        Values of :math:`U_{eff} \geq 0` and :math:`\beta_{eff} > 0` to scan.
    errors : dict
        Positive errors of `target_czz`, same keys. Default None.
    dens0 : :class:`numpy.ndarray`
        Initial densities of :func:`thermal_ip_scf`. Default None.
    spin_flip_average : bool
        Fit the spin-flip averaged correlations. Default False.
    scf_options : dict
        `mix`, `tol` and `max_iter` of :func:`thermal_ip_scf`. Default None.
    uhf_options : dict
        `seed` and `random_starts`: use :func:`thermal_uhf_scf` (requires
        positive :math:`U_{eff}`). Default None.

    Returns
    -------
    best : dict
        `k_trial`, `densities`, `mu_trial`, `u_eff`, `beta_eff`, `objective`,
        `czz`, all `candidates` and `metadata`.
    """
    if not numpy.isfinite(beta) or beta <= 0:
        raise ValueError("The physical beta must be finite and positive.")
    if not target_czz:
        raise ValueError("target_czz must contain at least one distance shell.")
    if any(int(d) != float(d) for d in target_czz):
        raise ValueError("Shell distances squared must be integer values.")
    target = {int(d): float(value) for d, value in target_czz.items()}
    if len(target) != len(target_czz) or not numpy.isfinite(list(target.values())).all():
        raise ValueError("target_czz must have distinct distance shells and finite values.")
    shells = tuple(target)
    sigma = numpy.ones(len(shells))
    if errors is not None:
        error_map = {int(d): float(value) for d, value in errors.items()}
        if set(error_map) != set(target):
            raise ValueError("errors must contain the same shells as target_czz.")
        sigma = numpy.asarray([error_map[d] for d in shells])
        if not numpy.isfinite(sigma).all() or numpy.any(sigma <= 0):
            raise ValueError("Correlation fitting errors must be finite and positive.")
    ug = numpy.asarray(tuple(u_eff_grid), dtype=float)
    bg = numpy.asarray(tuple(beta_eff_grid), dtype=float)
    if (
        ug.ndim != 1
        or bg.ndim != 1
        or not len(ug)
        or not len(bg)
        or not numpy.isfinite(ug).all()
        or numpy.any(ug < 0)
        or not numpy.isfinite(bg).all()
        or numpy.any(bg <= 0)
    ):
        raise ValueError("Trial grids must be nonempty: finite U_eff >= 0 and beta_eff > 0.")
    if uhf_options is not None:
        if dens0 is not None:
            raise ValueError("dens0 and multi-start uhf_options are mutually exclusive.")
        if numpy.any(ug <= 0):
            raise ValueError("A UHF-only fitting grid requires strictly positive U_eff.")
        if not set(uhf_options) <= {"seed", "random_starts"}:
            raise ValueError("uhf_options may specify seed and random_starts only.")
    options = dict(scf_options or {})
    if not set(options) <= {"mix", "tol", "max_iter"}:
        raise ValueError("scf_options may specify mix, tol, and max_iter only.")
    obs = HubbardSpinObservables(hop, 0.0, lx, ly, shells)
    candidates = []
    best = None
    target_values = numpy.asarray([target[d] for d in shells])
    for u_eff in ug:
        for beta_eff in bg:
            entry = {"u_eff": float(u_eff), "beta_eff": float(beta_eff)}
            try:
                if uhf_options is None:
                    k, density, mu_eff = thermal_ip_scf(
                        hop, v, beta_eff, target_n, u_eff, dens0=dens0, **options
                    )
                else:
                    k, density, mu_eff, uhf_info = thermal_uhf_scf(
                        hop,
                        v,
                        beta_eff,
                        target_n,
                        u_eff,
                        lx,
                        ly,
                        scf_options=options,
                        **uhf_options,
                    )
                    entry["uhf"] = uhf_info
                eps, coeff = numpy.linalg.eigh(k)
                occupations = 0.5 * (1 - numpy.tanh(0.5 * beta_eff * eps))
                G = numpy.einsum("sik,sk,sjk->sij", coeff, occupations, coeff.conj())
                raw = obs.measure(G)
                if spin_flip_average:
                    raw = 0.5 * (raw + obs.measure(G[::-1]))
                prediction = obs.finalise(raw)["czz"]
                values = numpy.asarray([prediction[str(d)] for d in shells])
                objective = float(numpy.mean(((values - target_values) / sigma) ** 2))
                if not numpy.isfinite(objective):
                    raise FloatingPointError("The fitting objective is non-finite.")
            except (RuntimeError, numpy.linalg.LinAlgError, FloatingPointError) as exc:
                entry.update(status="failed", reason=str(exc))
                candidates.append(entry)
                continue
            entry.update(status="ok", objective=objective, czz=prediction, mu_eff=float(mu_eff))
            candidates.append(entry)
            if best is None or objective < best["objective"]:
                best = {
                    "k_trial": k * beta_eff / beta,
                    "densities": density,
                    "mu_trial": float(mu_eff * beta_eff / beta),
                    "u_eff": float(u_eff),
                    "beta_eff": float(beta_eff),
                    "objective": objective,
                    "czz": prediction,
                }
    if best is None:
        reasons = "; ".join(c["reason"] for c in candidates)
        raise RuntimeError("All thermal-HF fitting candidates failed: " + reasons)
    best["candidates"] = candidates
    best["metadata"] = {
        "physical_beta": float(beta),
        "target_particles": float(target_n),
        "target_czz": {str(d): target[d] for d in shells},
        "shell_distances_squared": list(shells),
        "objective": "mean squared (predicted connected Czz - target)/error",
        "spin_flip_average": bool(spin_flip_average),
        "guide_scaling": "k_trial = (beta_eff/physical_beta) * k_HF",
        "method": "user-defined finite candidate grid; not an exact author-procedure reproduction",
    }
    return best
