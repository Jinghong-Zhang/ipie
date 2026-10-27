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

from ipie.addons.thermal.thermofield.hamiltonians.density_density import pair_table
from ipie.addons.thermal.thermofield.propagation.hubbard_discrete import ThermofieldHubbardDiscrete
from ipie.utils.backend import arraylib as xp


class ThermofieldDensityDensityDiscrete(ThermofieldHubbardDiscrete):
    r"""Constrained-path propagation with discrete Hirsch pair fields.

    For a general density-density interaction every pair term
    :math:`V (n_a n_b - (n_a + n_b)/2)` gets a real Hirsch field, and the
    one-body remainder :math:`V (n_a + n_b)/2` is added to the kinetic
    propagator built from `h1`. With :math:`\cosh\gamma = e^{\Delta\tau |V|/2}`,

    .. math::

        V > 0: \; e^{-\Delta\tau V (n_a n_b - (n_a + n_b)/2)}
            = \frac{1}{2} \sum_x e^{\gamma x (n_a - n_b)}, \\
        V < 0: \; e^{-\Delta\tau V (n_a n_b - (n_a + n_b)/2)}
            = \frac{1}{2} \sum_x e^{-\gamma x - \Delta\tau |V|/2} e^{\gamma x (n_a + n_b)}.

    Every factor is real, so sampling, kills and weights are those of
    :class:`ThermofieldHubbardDiscrete` with the field loop over pairs. A pair
    in two spin blocks scales one diagonal entry of each; a same-spin pair
    (i, j) scales two entries of one block, with ratio

    .. math::

        r(x) = (1 + d_a G_{ii})(1 + d_b G_{jj}) - d_a d_b G_{ij} G_{ji}

    at the interaction cut and two sequential rank-one updates. The single
    pair (up i, down i, U) per site reproduces the spin Hirsch propagator with
    :math:`h + U/2`.

    With `grouped`, consecutive fields whose spin-orbitals A fit in
    `group_size` are sampled on the local block :math:`G_{AA}` only, and the
    full Green's function is updated once per group,

    .. math::

        G \leftarrow G + (I - G)_{:, A} E (I + G_{AA} E)^{-1} G_{A, :},

    with E the accumulated diagonal change on A: a rank-|A| product instead of
    two rank-one updates of every entry per field. Both orders agree to
    rounding. Grouping requires importance_guide "same".

    Parameters
    ----------
    h1 : :class:`numpy.ndarray`
        Physical one-body Hamiltonian, shape (2, nbasis, nbasis).
    pairs :
        Pair terms (spin_a, orb_a, spin_b, orb_b, V), see
        :func:`ipie.addons.thermal.thermofield.hamiltonians.density_density.pair_table`.
    timestep : float
        Imaginary timestep.
    beta : float
        Inverse temperature of the walk.
    rng :
        Ignored, as in :class:`ThermofieldHubbardCP`.
    grouped : bool
        Grouped sampling. Default True.
    group_size : int
        Largest number of spin-orbitals per group. Default 10.
    kwargs :
        Options of :class:`ThermofieldHubbardDiscrete`.
    """

    def __init__(self, h1, pairs, timestep, beta, rng=None, grouped=True, group_size=10, **kwargs):
        if grouped and kwargs.get("importance_guide", "same") != "same":
            raise ValueError(
                "Grouped pair updates need importance_guide='same'; pass grouped=False."
            )
        h1 = numpy.asarray(h1)
        if h1.ndim != 3 or h1.shape[0] != 2:
            raise ValueError("h1 must have shape (2, M, M).")
        spin, orb, V = pair_table(pairs, h1.shape[-1])
        h_kin = numpy.array(h1, dtype=float, copy=True)
        for leg in range(2):
            numpy.add.at(h_kin, (spin[:, leg], orb[:, leg], orb[:, leg]), 0.5 * V)
        super().__init__(h_kin, 0.0, timestep, beta, rng=rng, **kwargs)
        with numpy.errstate(over="ignore", invalid="ignore"):
            gamma = numpy.arccosh(numpy.exp(0.5 * timestep * numpy.abs(V)))
        x = numpy.array([1.0, -1.0])
        # Second leg couples to -x (spin-like) for V > 0 and +x (charge-like) for V < 0.
        self.pair_coupling = numpy.stack([gamma, numpy.where(V > 0, -gamma, gamma)], axis=1)
        self.auxf = numpy.exp(x[None, :, None] * self.pair_coupling[:, None, :])
        if not numpy.isfinite(self.auxf).all():
            raise ValueError("Hirsch factors overflow; reduce timestep*|V|.")
        self.delta = self.auxf - 1.0
        self.field_log_scalar = numpy.where(
            V[:, None] < 0,
            -gamma[:, None] * x[None, :] - 0.5 * timestep * numpy.abs(V)[:, None],
            0.0,
        )
        self.pair_spin, self.pair_orb, self.pair_V = spin, orb, V
        self.gamma = gamma
        self.nfields = len(V)
        # No single on-site U describes this interaction.
        self.U = None
        self.hs_type = "discrete_pair"
        self.field_proposal = self.field_proposal.replace("site", "pair")
        self.grouped = bool(grouped)
        self.field_groups = self._build_groups(int(group_size))

    def _build_groups(self, group_size):
        """Consecutive fields whose spin-orbitals fit in group_size, with local indices per spin."""
        groups, members, orbitals = [], [], set()
        for f in range(self.nfields):
            legs = {(int(self.pair_spin[f, k]), int(self.pair_orb[f, k])) for k in range(2)}
            if members and len(orbitals | legs) > group_size:
                groups.append(self._make_group(members, orbitals))
                members, orbitals = [], set()
            members.append(f)
            orbitals |= legs
        groups.append(self._make_group(members, orbitals))
        return groups

    def _make_group(self, members, orbitals):
        per_spin = [sorted(i for s, i in orbitals if s == spin) for spin in range(2)]
        local = {(s, i): k for s in range(2) for k, i in enumerate(per_spin[s])}
        local_orb = numpy.array(
            [
                [
                    local[(int(self.pair_spin[f, leg]), int(self.pair_orb[f, leg]))]
                    for leg in range(2)
                ]
                for f in members
            ]
        )
        return SimpleNamespace(
            fields=numpy.array(members),
            local_orb=local_orb,
            orbitals=[numpy.array(o, dtype=int) for o in per_spin],
            size=max(len(o) for o in per_spin),
        )

    def _field_probabilities(self, green, field, ids=None, orb=None):
        sa, sb = self.pair_spin[field]
        i, j = self.pair_orb[field] if orb is None else orb
        gii, gjj = green[:, sa, i, i], green[:, sb, j, j]
        if ids is not None:
            gii, gjj = gii[ids], gjj[ids]
        da, db = self.delta[field, :, 0], self.delta[field, :, 1]
        ratio = (1.0 + da[None, :] * gii[:, None]) * (1.0 + db[None, :] * gjj[:, None])
        if sa == sb:
            exchange = green[:, sa, i, j] * green[:, sa, j, i]
            if ids is not None:
                exchange = exchange[ids]
            ratio = ratio - (da * db)[None, :] * exchange[:, None]
        return 0.5 * xp.exp(self.field_log_scalar[field])[None, :] * ratio

    def _field_green_update(self, green, field, chosen, orb=None):
        for leg in range(2):
            s = self.pair_spin[field, leg]
            i = self.pair_orb[field, leg] if orb is None else orb[leg]
            delta = self.delta[field, chosen, leg]
            column = -green[:, s, :, i].copy()
            column[:, i] += 1.0
            row = green[:, s, i, :].copy()
            ratio = 1.0 + delta * green[:, s, i, i]
            green[:, s] += (delta / ratio)[:, None, None] * column[:, :, None] * row[:, None, :]

    def _field_factors(self, factors, active, field, chosen):
        for leg in range(2):
            s, i = self.pair_spin[field, leg], self.pair_orb[field, leg]
            factors[active, s, i] *= self.auxf[field, chosen, leg]

    def _field_log_scalar(self, field, chosen):
        return self.field_log_scalar[field, chosen]

    def _field_legs(self):
        return self.pair_spin, self.pair_orb, self.delta, self.auxf, self.field_log_scalar

    def _slice_start(self, walkers, trial, step, ids):
        state = super()._slice_start(walkers, trial, step, ids)
        if self.grouped and not numpy.any(state.green.imag):
            # Real one-body matrices, fields and guide: the contraction is exactly
            # real, so the grouped sampling and flush GEMMs run in real arithmetic.
            state.green = numpy.ascontiguousarray(state.green.real)
        return state

    def _slice_fields(self, walkers, step, ids, uniforms, state):
        if not self.grouped:
            return super()._slice_fields(walkers, step, ids, uniforms, state)
        green, live, factors = state.green, state.live, state.factors
        log_weight, log_s = state.log_weight, state.log_s
        for group in self.field_groups:
            idx = numpy.flatnonzero(live)
            if len(idx) == 0:
                break
            local = numpy.zeros((len(idx), 2, group.size, group.size), dtype=green.dtype)
            f0 = numpy.ones((len(idx), 2, group.size))
            for s, A in enumerate(group.orbitals):
                if len(A):
                    local[:, s, : len(A), : len(A)] = green[numpy.ix_(idx, [s], A, A)][:, 0]
                    f0[:, s, : len(A)] = factors[numpy.ix_(idx, [s], A)][:, 0]
            alive = numpy.ones(len(idx), dtype=bool)
            for row, field in enumerate(group.fields):
                active = numpy.flatnonzero(alive)
                if len(active) == 0:
                    break
                orb = group.local_orb[row]
                with numpy.errstate(over="ignore", invalid="ignore"):
                    probabilities = self._field_probabilities(local, field, active, orb)
                good_p = numpy.isfinite(probabilities).all(axis=1) & (
                    numpy.abs(probabilities.imag)
                    <= self.real_tolerance * numpy.maximum(1.0, numpy.abs(probabilities.real))
                ).all(axis=1)
                self.invalid_weight_count += int((~good_p).sum())
                alive[active[~good_p]] = False
                probabilities = probabilities[good_p].real
                active = active[good_p]
                self.step_count += len(active)
                self.forbidden_field_count += int((probabilities <= 0).sum())
                positive = numpy.maximum(probabilities, 0.0)
                with numpy.errstate(over="ignore", invalid="ignore"):
                    norm = positive.sum(axis=1)
                valid_norm = numpy.isfinite(norm)
                self.invalid_weight_count += int((~valid_norm).sum())
                self.kill_count += int((valid_norm & (norm == 0)).sum())
                accepted = valid_norm & (norm > 0)
                alive[active[~accepted]] = False
                active, norm, positive = active[accepted], norm[accepted], positive[accepted]
                walker = idx[active]
                chosen = (uniforms[walker, field] >= positive[:, 0] / norm).astype(int)
                all_active = len(active) == len(idx)
                update = local if all_active else local[active]
                with numpy.errstate(divide="ignore", over="ignore", invalid="ignore"):
                    self._field_green_update(update, field, chosen, orb)
                valid_g = numpy.isfinite(update).all(axis=(1, 2, 3))
                self.invalid_weight_count += int((~valid_g).sum())
                alive[active[~valid_g]] = False
                if not all_active:
                    local[active] = update
                self._field_factors(factors, walker, field, chosen)
                log_weight[walker] += numpy.log(norm)
                sel = numpy.arange(len(chosen))
                log_s[walker] += (
                    numpy.log(2.0)
                    + numpy.log(probabilities[accepted][sel, chosen])
                    - self._field_log_scalar(field, chosen)
                )
                if self.record_fields:
                    walkers.field_hist[ids[walker], step, field] = 1 - 2 * chosen
            live[idx[~alive]] = False
            keep = numpy.flatnonzero(alive)
            if len(keep) == 0:
                continue
            with numpy.errstate(divide="ignore", over="ignore", invalid="ignore"):
                self._flush_group(green, factors, idx[keep], f0[keep], group)
            valid = numpy.isfinite(green[idx[keep]]).all(axis=(1, 2, 3))
            self.invalid_weight_count += int((~valid).sum())
            live[idx[keep][~valid]] = False

    @staticmethod
    def _flush_group(green, factors, walker, f0, group):
        """G <- G + (I - G)[:, A] E (I + G_AA E)^{-1} G[A, :] per spin, E = factor change on A."""
        whole = len(walker) == green.shape[0]
        for s, A in enumerate(group.orbitals):
            k = len(A)
            if k == 0:
                continue
            e = factors[numpy.ix_(walker, [s], A)][:, 0] / f0[:, s, :k] - 1.0
            G = green[:, s] if whole else green[walker, s]
            column = -G[:, :, A]
            column[:, A, numpy.arange(k)] += 1.0
            g_aa = G[:, A[:, None], A[None, :]]
            core = e[:, :, None] * numpy.linalg.inv(numpy.eye(k) + g_aa * e[:, None, :])
            change = column @ (core @ G[:, A, :])
            if whole:
                green[:, s] += change
            else:
                green[walker, s] = G + change

    def construct_slice_propagators(self, fields):
        r"""Slice propagators :math:`B(x) = K D(x) K` for recorded pair fields.

        Parameters
        ----------
        fields : :class:`numpy.ndarray`
            Fields of one slice, shape (nwalkers, nfields).

        Returns
        -------
        B : :class:`numpy.ndarray`
            Shape (nwalkers, 2, nbasis, nbasis).
        """
        nbasis = self.BK2.shape[-1]
        dv = self.slice_diagonal(fields)
        B = numpy.empty((len(fields), 2, nbasis, nbasis), dtype=numpy.complex128)
        for s in range(2):
            B[:, s] = self.BK2[s] @ (dv[:, s, :, None] * self.BK2[s])
        return B

    def slice_diagonal(self, fields):
        """Diagonal of D(x), shape (nwalkers, 2, nbasis), for recorded fields (nwalkers, nfields)."""
        log_diag = numpy.zeros((len(fields), 2, self.BK2.shape[-1]))
        for leg in range(2):
            numpy.add.at(
                log_diag,
                (slice(None), self.pair_spin[:, leg], self.pair_orb[:, leg]),
                fields * self.pair_coupling[:, leg],
            )
        return numpy.exp(log_diag)
