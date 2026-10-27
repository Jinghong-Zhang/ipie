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
from dataclasses import dataclass

import numpy

LAYOUTS = ("rhf", "uhf", "ghf", "uhf_ph")


def fermi(eps, beta):
    r"""Fermi occupations :math:`1 / (1 + e^{\beta\epsilon})`, accurate also when tiny.

    Parameters
    ----------
    eps : :class:`numpy.ndarray`
        Levels measured from the chemical potential.
    beta : float
        Inverse temperature.

    Returns
    -------
    occ : :class:`numpy.ndarray`
        Occupations.
    """
    return numpy.exp(-numpy.logaddexp(0.0, beta * numpy.asarray(eps)))


def fermi_entropy(eps, beta):
    r"""Entropy :math:`-\sum_p [f_p \ln f_p + (1 - f_p) \ln(1 - f_p)]` of the levels `eps`."""
    x = numpy.abs(beta * numpy.asarray(eps))
    ex = numpy.exp(-x)
    return float(numpy.sum(numpy.log1p(ex) + x * ex / (1.0 + ex)))


def _log_cosh(a):
    a = numpy.abs(a)
    return a - numpy.log(2.0) + numpy.log1p(numpy.exp(-2.0 * a))


def _fixed_n_weights(eps_list, shift, beta):
    """-f'(eps - shift) of every level, scaled so that the largest is 1 (no underflow)."""
    log_w = [-2.0 * _log_cosh(0.5 * beta * (e - shift)) for e in eps_list]
    top = max(float(lw.max()) for lw in log_w)
    return [numpy.exp(lw - top) for lw in log_w]


def fermi_divided_difference(eps, beta):
    r"""Divided differences of the Fermi function, stable for any :math:`\beta\epsilon`.

    .. math::

        D_{pq} = \frac{f_p - f_q}{\epsilon_p - \epsilon_q}
               = -\frac{\beta}{4} \frac{\sinh(a_p - a_q)}{(a_p - a_q) \cosh a_p \cosh a_q},
        \qquad a = \beta\epsilon / 2,

    with :math:`D_{pp} = f'(\epsilon_p)`, evaluated in logarithms.

    Parameters
    ----------
    eps : :class:`numpy.ndarray`
        Levels, shape (n,).
    beta : float
        Inverse temperature.

    Returns
    -------
    D : :class:`numpy.ndarray`
        Shape (n, n).
    """
    a = 0.5 * beta * numpy.asarray(eps)
    d = numpy.abs(a[:, None] - a[None, :])
    small = d < 1e-4
    dd = numpy.where(small, 1.0, d)
    log_sinhc = numpy.where(
        small, d * d / 6.0, dd - numpy.log(2.0 * dd) + numpy.log(-numpy.expm1(-2.0 * dd))
    )
    log_cosh = _log_cosh(a)
    return -0.25 * beta * numpy.exp(log_sinhc - log_cosh[:, None] - log_cosh[None, :])


def spin_rotation(angle, axis):
    r"""SU(2) matrix :math:`e^{-i \theta \hat{n} \cdot \sigma / 2}` with :math:`\hat{n}` = axis / |axis|."""
    n = numpy.asarray(axis, dtype=float)
    n = n / numpy.linalg.norm(n)
    n_sigma = numpy.array([[n[2], n[0] - 1j * n[1]], [n[0] + 1j * n[1], -n[2]]])
    return numpy.cos(0.5 * angle) * numpy.eye(2) - 1j * numpy.sin(0.5 * angle) * n_sigma


def rotate_spin(k, rotation):
    r""":math:`(R \otimes I) k (R \otimes I)^\dagger` for a spin-grouped matrix k of shape (2M, 2M)."""
    R = numpy.kron(rotation, numpy.eye(k.shape[0] // 2))
    return R @ k @ R.conj().T


def spin_block_diag(blocks):
    """Spin-grouped matrix of shape (2M, 2M) from the (up, down) blocks of shape (M, M)."""
    M = blocks[0].shape[0]
    out = numpy.zeros((2 * M, 2 * M), dtype=numpy.result_type(blocks[0], blocks[1]))
    out[:M, :M] = blocks[0]
    out[M:, M:] = blocks[1]
    return out


@dataclass
class ThermalHFState:
    """Free energy, density, Fock matrices and gradient at one point of the minimization.

    Lists hold one matrix per functional block (up and down, or one for ghf).
    """

    variables: list
    blocks: list
    eps: list
    vecs: list
    shift: float
    rho: list
    fock: list
    energy: float
    entropy: float
    nelec: float
    free_energy: float
    gradient: list
    direction: list


@dataclass
class ThermalHFSolution:
    """Converged (or last) thermal Hartree-Fock point.

    Attributes
    ----------
    layout : str
        Layout of the problem.
    beta : float
        Inverse temperature.
    k : :class:`numpy.ndarray`
        Shifted Fock matrix F - mu of the last iterate, shape (2, M, M) for
        "rhf", "uhf" and "uhf_ph", (2M, 2M) for "ghf".
    rho : :class:`numpy.ndarray`
        Density matrix f(k), same shape as k.
    mu : float
        Chemical potential.
    free_energy : float
        E - T S - mu N (grand canonical) or E - T S (fixed N).
    energy : float
        Energy, with the -U n / 2 shift of the interaction.
    entropy : float
        Entropy.
    nelec : float
        Mean particle number.
    residual : float
        Largest change of rho in the last Roothaan map (the convergence criterion).
    gradient_norm : float
        Norm of the gradient.
    iterations : int
        Number of iterations.
    converged : bool
        Residual below the tolerance.
    variables : list
        Variables of the last iterate.
    """

    layout: str
    beta: float
    k: numpy.ndarray
    rho: numpy.ndarray
    mu: float
    free_energy: float
    energy: float
    entropy: float
    nelec: float
    residual: float
    gradient_norm: float
    iterations: int
    converged: bool
    variables: list


class ThermalHubbardHF:
    r"""Mermin free energy functional of the Hubbard model in one mean-field layout.

    :math:`\Omega[\rho] = E_{HF}[\rho] - T S[\rho] - \mu N[\rho]` is minimized over
    the one-body operators k, :math:`\rho = f(k)`, by preconditioned nonlinear
    conjugate gradients (:meth:`solve`), with :meth:`stability` for the
    Hessian at a stationary point.

    Parameters
    ----------
    h : :class:`numpy.ndarray`
        Hermitian one-body blocks (up, down) of shape (2, M, M) for "rhf",
        "uhf" and "uhf_ph"; a spin-grouped Hermitian matrix of shape (2M, 2M)
        for "ghf".
    U : float or :class:`numpy.ndarray`
        Hubbard interaction, scalar or per site.
    beta : float
        Inverse temperature.
    layout : str
        "rhf", "uhf", "ghf", or "uhf_ph" (particle-hole symmetric UHF at half
        filling).
    nelec : float
        Fixed mean particle number. Give exactly one of nelec and mu.
    mu : float
        Chemical potential (grand canonical); "uhf_ph" requires mu = 0.
    eta : :class:`numpy.ndarray`
        Sublattice signs, shape (M,); required for "uhf_ph".
    """

    def __init__(self, h, U, beta, layout, nelec=None, mu=None, eta=None):
        if layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}.")
        if not numpy.isfinite(beta) or beta <= 0:
            raise ValueError("beta must be finite and positive.")
        if (nelec is None) == (mu is None):
            raise ValueError("Give exactly one of nelec or mu.")
        h = numpy.asarray(h)
        if layout == "ghf":
            if h.ndim != 2 or h.shape[0] != h.shape[1] or h.shape[0] % 2:
                raise ValueError("A ghf one-body matrix must be 2M x 2M.")
            self.M = h.shape[0] // 2
            self.h = [h]
        else:
            if h.ndim != 3 or h.shape[0] != 2 or h.shape[1] != h.shape[2]:
                raise ValueError("Spin one-body blocks must have shape (2, M, M).")
            self.M = h.shape[1]
            self.h = [h[0], h[1]]
        if not all(numpy.allclose(b, b.conj().T, atol=1e-12, rtol=0) for b in self.h):
            raise ValueError("One-body matrices must be Hermitian.")
        self.U = numpy.broadcast_to(numpy.asarray(U, dtype=float), (self.M,)).copy()
        if not numpy.isfinite(self.U).all():
            raise ValueError("U must be finite.")
        self.beta = float(beta)
        self.layout = layout
        self.nelec = None if nelec is None else float(nelec)
        self.mu = None if mu is None else float(mu)
        nmodes = 2 * self.M
        if self.nelec is not None and not 0 < self.nelec < nmodes:
            raise ValueError("nelec must lie strictly between 0 and 2M.")
        if layout == "uhf_ph":
            if self.mu != 0.0:
                raise ValueError("The uhf_ph class is half filled: use mu=0.")
            self.eta = numpy.asarray(eta, dtype=float)
            if self.eta.shape != (self.M,) or not numpy.all(numpy.abs(self.eta) == 1):
                raise ValueError("eta must be M sublattice signs +-1.")
            if not numpy.allclose(self.h[1], self._ph(self.h[0]), atol=1e-12, rtol=0):
                raise ValueError("uhf_ph needs h_dn = -eta h_up^T eta.")
        self.ghf = layout == "ghf"
        self.max_line_search = 60
        self.max_step = 4.0

    # Maps between the variables and the functional blocks.
    def _ph(self, k):
        return -self.eta[:, None] * k.T * self.eta[None, :]

    def blocks(self, variables):
        """Functional blocks (up, dn), or the single ghf block, from variables."""
        if self.layout == "rhf":
            return [variables[0], variables[0]]
        if self.layout == "uhf_ph":
            return [variables[0], self._ph(variables[0])]
        return list(variables)

    def _chain(self, blocks):
        """Gradient with respect to the variables from block gradients."""
        if self.layout == "rhf":
            return [blocks[0] + blocks[1]]
        if self.layout == "uhf_ph":
            return [blocks[0] + self._ph(blocks[1])]
        return blocks

    def _project(self, blocks):
        """Variables closest to block matrices that obey the layout's constraint."""
        if self.layout == "rhf":
            return [0.5 * (blocks[0] + blocks[1])]
        if self.layout == "uhf_ph":
            return [0.5 * (blocks[0] + self._ph(blocks[1]))]
        return blocks

    @staticmethod
    def dot(a, b):
        """Real inner product of two lists of matrices."""
        return float(sum(numpy.real(numpy.vdot(x, y)) for x, y in zip(a, b)))

    # The functional.
    def interaction(self, rho):
        """Hubbard HF interaction energy and Fock matrices, including the -U n / 2 shift."""
        U, M = self.U, self.M
        idx = numpy.arange(M)
        if self.ghf:
            r = rho[0]
            n_up, n_dn = r[idx, idx].real, r[idx + M, idx + M].real
            x = r[idx, idx + M]
            energy = float(numpy.sum(U * (n_up * n_dn - (x * r[idx + M, idx]).real)))
            fock = numpy.zeros_like(r)
            fock[idx, idx] = U * n_dn
            fock[idx + M, idx + M] = U * n_up
            fock[idx, idx + M] = -U * x
            fock[idx + M, idx] = -U * r[idx + M, idx]
            focks = [fock]
        else:
            n_up, n_dn = numpy.diagonal(rho[0]).real, numpy.diagonal(rho[1]).real
            energy = float(numpy.sum(U * n_up * n_dn))
            focks = [numpy.zeros_like(rho[0]), numpy.zeros_like(rho[1])]
            focks[0][idx, idx] = U * n_dn
            focks[1][idx, idx] = U * n_up
        energy -= 0.5 * float(numpy.sum(U * (n_up + n_dn)))
        for f in focks:
            f[numpy.arange(f.shape[0]), numpy.arange(f.shape[0])] -= 0.5 * numpy.tile(
                U, f.shape[0] // M
            )
        return energy, focks

    def energy_fock(self, rho):
        """E_HF[rho] and the Fock blocks F = dE / d rho^T."""
        e_int, f_int = self.interaction(rho)
        e_one = float(sum(numpy.real(numpy.vdot(h, r)) for h, r in zip(self.h, rho)))
        return e_one + e_int, [h + f for h, f in zip(self.h, f_int)]

    def free_energy_of_rho(self, rho):
        """Omega evaluated directly from density blocks (entropy from eigenvalues of rho)."""
        energy, _ = self.energy_fock(rho)
        lam = numpy.concatenate([numpy.linalg.eigvalsh(r) for r in rho])
        if not numpy.all((lam > 0) & (lam < 1)):
            raise ValueError("rho eigenvalues must lie in (0, 1).")
        entropy = -float(numpy.sum(lam * numpy.log(lam) + (1 - lam) * numpy.log1p(-lam)))
        nelec = float(lam.sum())
        mu = 0.0 if self.mu is None else self.mu
        return energy - entropy / self.beta - mu * nelec

    def _shift(self, eps):
        """nu with sum_p f(eps_p - nu) = nelec, by bisection to machine precision.

        N(nu) - nelec is evaluated as particles above minus holes below the
        n0 = round(nelec) lowest levels, each with relative precision (in
        logarithms when nelec = n0), so the root stays at the true, mid-gap
        position when nelec fills a gap much wider than T.
        """
        e = numpy.sort(numpy.concatenate(eps))
        n0 = int(numpy.clip(numpy.rint(self.nelec), 0, len(e)))
        lo = e.min() - 50.0 / self.beta - 1.0
        hi = e.max() + 50.0 / self.beta + 1.0
        for _ in range(400):
            mid = 0.5 * (lo + hi)
            if mid in (lo, hi):
                break
            if n0 == self.nelec:
                log_p = numpy.logaddexp.reduce(
                    -numpy.logaddexp(0.0, self.beta * (e[n0:] - mid)), initial=-numpy.inf
                )
                log_h = numpy.logaddexp.reduce(
                    -numpy.logaddexp(0.0, self.beta * (mid - e[:n0])), initial=-numpy.inf
                )
                excess = log_p - log_h
            else:
                excess = (
                    fermi(e[n0:] - mid, self.beta).sum()
                    - fermi(mid - e[:n0], self.beta).sum()
                    + (n0 - self.nelec)
                )
            if excess < 0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def evaluate(self, variables):
        """Free energy, density, Fock matrices, gradient and preconditioned direction."""
        blocks = self.blocks(variables)
        eigs = [numpy.linalg.eigh(k) for k in blocks]
        eps = [e for e, _ in eigs]
        vecs = [v for _, v in eigs]
        shift = 0.0 if self.nelec is None else self._shift(eps)
        occ = [fermi(e - shift, self.beta) for e in eps]
        rho = [(v * o) @ v.conj().T for v, o in zip(vecs, occ)]
        energy, fock = self.energy_fock(rho)
        entropy = sum(fermi_entropy(e - shift, self.beta) for e in eps)
        nelec = float(sum(o.sum() for o in occ))
        mu = 0.0 if self.mu is None else self.mu
        free_energy = energy - entropy / self.beta - mu * nelec
        dd = [fermi_divided_difference(e - shift, self.beta) for e in eps]
        rt = [
            v.conj().T @ (f - mu * numpy.eye(len(e)) - k) @ v
            for v, f, k, e in zip(vecs, fock, blocks, eps)
        ]
        if self.nelec is None:
            c = 0.0
        else:
            w = _fixed_n_weights(eps, shift, self.beta)
            c = sum(numpy.sum(wb * numpy.diagonal(r).real) for wb, r in zip(w, rt)) / sum(
                wb.sum() for wb in w
            )
        eye = [numpy.eye(len(e)) for e in eps]
        grad = [v @ (d * (r - c * i)) @ v.conj().T for v, d, r, i in zip(vecs, dd, rt, eye)]
        direction = [v @ (r - c * i) @ v.conj().T for v, r, i in zip(vecs, rt, eye)]
        return ThermalHFState(
            variables=[numpy.array(x) for x in variables],
            blocks=blocks,
            eps=eps,
            vecs=vecs,
            shift=shift,
            rho=rho,
            fock=fock,
            energy=energy,
            entropy=entropy,
            nelec=nelec,
            free_energy=free_energy,
            gradient=self._chain(grad),
            direction=self._project(direction),
        )

    def roothaan(self, state):
        """One Roothaan map: (variables F - mu of rho, chemical potential, max |rho' - rho|)."""
        mu = 0.0 if self.mu is None else self.mu
        target = self._project([f - mu * numpy.eye(f.shape[0]) for f in state.fock])
        blocks = self.blocks(target)
        eigs = [numpy.linalg.eigh(k) for k in blocks]
        shift = 0.0 if self.nelec is None else self._shift([e for e, _ in eigs])
        rho = [(v * fermi(e - shift, self.beta)) @ v.conj().T for e, v in eigs]
        residual = max(float(numpy.max(numpy.abs(a - b))) for a, b in zip(rho, state.rho))
        target = [t - shift * numpy.eye(t.shape[0]) for t in target]
        return target, mu + shift, residual

    # Minimization.
    def _line_search(self, state, d, alpha):
        """Step a <= max_step with Omega not above Omega(0) and |dOmega/da| <= 0.1 |dOmega/da(0)|.

        A unit step along the bare residual is one Roothaan iteration; the cap
        keeps steps from running away along directions in which Omega is flat
        (levels saturated far from the chemical potential).  The derivative,
        unlike Omega itself, stays accurate near convergence and drives the
        bracketing.  Returns (0, None) when no step meets the conditions,
        i.e. when Omega varies along d only at the level of rounding.
        """
        f0 = state.free_energy
        g0 = self.dot(state.gradient, d)
        slack = 1e-12 * (1.0 + abs(f0))
        lo, g_lo, hi, g_hi = 0.0, g0, None, None
        a = min(alpha, self.max_step)
        for _ in range(self.max_line_search):
            trial = self.evaluate([x + a * y for x, y in zip(state.variables, d)])
            g = self.dot(trial.gradient, d)
            down = trial.free_energy <= f0 + slack
            if down and abs(g) <= 0.1 * abs(g0):
                return a, trial
            if down and g < 0:
                if a >= self.max_step:
                    return a, trial
                lo, g_lo = a, g
            else:
                hi, g_hi = a, g
            if hi is None:
                a = min(2.0 * a, self.max_step)
                continue
            if g_hi > 0 > g_lo:
                a = lo - g_lo * (hi - lo) / (g_hi - g_lo)
            else:
                a = 0.5 * (lo + hi)
            a = min(max(a, lo + 0.1 * (hi - lo)), hi - 0.1 * (hi - lo))
        return 0.0, None

    def solve(self, variables, tol=1e-10, max_iter=1000):
        """Minimize the free energy by preconditioned nonlinear conjugate gradients.

        Polak-Ribiere+ with Powell restarts. Converged when one Roothaan map
        changes rho by less than `tol`. When the gradient has no usable descent
        information (every level far from the chemical potential, or the
        rounding floor near convergence) the step is a plain Roothaan
        iteration k <- F - mu.

        Parameters
        ----------
        variables : list
            Initial variables.
        tol : float
            Convergence threshold on rho. Default 1e-10.
        max_iter : int
            Maximum number of iterations. Default 1000.

        Returns
        -------
        solution : :class:`ThermalHFSolution`
            The Roothaan polish k = F - mu of the final density, with rho = f(k).
        """
        variables = [0.5 * (x + x.conj().T) for x in variables]
        state = self.evaluate(variables)
        d = state.direction
        alpha = 1.0
        for it in range(max_iter + 1):
            target, mu, residual = self.roothaan(state)
            if residual < tol or it == max_iter:
                break
            if not self.dot(state.gradient, d) < 0:
                d = state.direction
            new = None
            if self.dot(state.gradient, d) < 0:
                alpha, new = self._line_search(state, d, alpha)
            if new is None:
                state = self.evaluate(target)
                d, alpha = state.direction, 1.0
                continue
            rs_old = -self.dot(state.gradient, state.direction)
            rs_new = -self.dot(new.gradient, new.direction)
            cross = -self.dot(new.gradient, state.direction)
            # Powell restart when successive residuals are far from conjugate.
            if abs(cross) >= 0.2 * rs_new:
                beta_pr = 0.0
            else:
                beta_pr = max(0.0, (rs_new - cross) / rs_old)
            d = [z + beta_pr * x for z, x in zip(new.direction, d)]
            state = new
        final = self.evaluate(target)
        if self.ghf:
            k, rho = final.variables[0], final.rho[0]
        else:
            k, rho = numpy.array(self.blocks(final.variables)), numpy.array(final.rho)
        return ThermalHFSolution(
            layout=self.layout,
            beta=self.beta,
            k=k,
            rho=rho,
            mu=mu,
            free_energy=final.free_energy,
            energy=final.energy,
            entropy=final.entropy,
            nelec=final.nelec,
            residual=residual,
            gradient_norm=numpy.sqrt(self.dot(final.gradient, final.gradient)),
            iterations=it,
            converged=bool(residual < tol),
            variables=final.variables,
        )

    # Stability.
    def local_operators(self):
        r"""Hermitian one-body operators whose expectations carry the interaction.

        With :math:`w_a = \mathrm{Tr}(O_a \delta\rho)`,
        :math:`\sum_i U_i (\delta n_{i\uparrow} \delta n_{i\downarrow}
        - |\delta\rho_{i\uparrow, i\downarrow}|^2) = w^T C w / 2`. The spin layouts use
        :math:`n_\uparrow, n_\downarrow`; "ghf" adds the real and imaginary parts of
        :math:`\rho_{i\uparrow, i\downarrow}`.

        Returns
        -------
        operators : list
            (block, matrix) pairs.
        C : :class:`numpy.ndarray`
            Interaction kernel.
        """
        M, U = self.M, self.U
        ops = []
        if self.ghf:
            for i in range(M):
                for p, q, c in ((i, i, 1.0), (i + M, i + M, 1.0)):
                    o = numpy.zeros((2 * M, 2 * M), dtype=complex)
                    o[p, q] = c
                    ops.append((0, o))
                o = numpy.zeros((2 * M, 2 * M), dtype=complex)
                o[i + M, i], o[i, i + M] = 0.5, 0.5
                ops.append((0, o))
                o = numpy.zeros((2 * M, 2 * M), dtype=complex)
                o[i + M, i], o[i, i + M] = -0.5j, 0.5j
                ops.append((0, o))
            C = numpy.zeros((4 * M, 4 * M))
            for i in range(M):
                C[4 * i, 4 * i + 1] = C[4 * i + 1, 4 * i] = U[i]
                C[4 * i + 2, 4 * i + 2] = C[4 * i + 3, 4 * i + 3] = -2.0 * U[i]
        else:
            for b in range(2):
                for i in range(M):
                    o = numpy.zeros((M, M), dtype=complex)
                    o[i, i] = 1.0
                    ops.append((b, o))
            C = numpy.zeros((2 * M, 2 * M))
            C[numpy.arange(M), numpy.arange(M) + M] = U
            C[numpy.arange(M) + M, numpy.arange(M)] = U
        return ops, C

    def stability(self, variables, complex_perturbations=None):
        r"""Free energy Hessian at a stationary point, in coordinates with unit entropy Hessian.

        With L the Frechet derivative of f, the Hessian of :math:`\Omega` in
        :math:`\rho` is :math:`|L|^{-1} + K`. In :math:`x = |L|^{-1/2} \delta\rho` it is
        :math:`S = 1 + B^T C B`, where B maps x to the local expectations w (see
        :meth:`local_operators`). Directions outside the span of :math:`B^T` have
        eigenvalue 1; the others are :math:`1 + \mathrm{eig}(\Gamma^{1/2} C \Gamma^{1/2})`
        with :math:`\Gamma = B B^T` the static Lindhard susceptibility of the local
        operators. Perturbations are independent for every functional block:
        collinear for the spin layouts (so "rhf" and "uhf_ph" are tested as
        UHF), general for "ghf". At fixed N the constraint
        :math:`\mathrm{Tr}\,\delta\rho = 0` is imposed.

        Parameters
        ----------
        variables : list
            Stationary point.
        complex_perturbations : bool
            Allow complex perturbations. Default: complex if the blocks are.

        Returns
        -------
        eigenvalues : :class:`numpy.ndarray`
            Nontrivial eigenvalues of S, ascending.
        modes : list
            For every eigenvalue, the variation :math:`\delta k = -\sum_a y_a O_a`
            (one matrix per block) and the curvature norm, such that
            :math:`\Omega(k + t \delta k)` has curvature eigenvalue times
            :math:`|x|^2` at t = 0.
        """
        blocks = self.blocks(variables)
        eigs = [numpy.linalg.eigh(k) for k in blocks]
        shift = 0.0 if self.nelec is None else self._shift([e for e, _ in eigs])
        s2 = [-fermi_divided_difference(e - shift, self.beta) for e, _ in eigs]
        if complex_perturbations is None:
            complex_perturbations = any(numpy.iscomplexobj(k) for k in blocks)
        if not complex_perturbations and any(numpy.iscomplexobj(v) for _, v in eigs):
            raise ValueError("Complex orbitals need complex perturbations.")
        ops, C = self.local_operators()
        tilde = [eigs[b][1].conj().T @ o @ eigs[b][1] for b, o in ops]
        if not complex_perturbations:
            tilde = [t.real for t in tilde]
        n = len(ops)
        gamma = numpy.zeros((n, n))
        for a in range(n):
            for c in range(a, n):
                if ops[a][0] == ops[c][0]:
                    gamma[a, c] = gamma[c, a] = numpy.sum(
                        s2[ops[a][0]] * (tilde[a].conj() * tilde[c]).real
                    )
        if self.nelec is not None:
            # Unit vector of the Tr d rho constraint in x coordinates, proportional to sqrt(-f').
            w = _fixed_n_weights([e for e, _ in eigs], shift, self.beta)
            norm = numpy.sqrt(sum(wb.sum() for wb in w))
            u = numpy.array(
                [
                    numpy.sum(
                        numpy.sqrt(w[b])
                        / norm
                        * numpy.sqrt(numpy.diagonal(s2[b]))
                        * numpy.diagonal(t).real
                    )
                    for (b, _), t in zip(ops, tilde)
                ]
            )
            gamma -= numpy.outer(u, u)
        w, z = numpy.linalg.eigh(gamma)
        root = (z * numpy.sqrt(numpy.clip(w, 0.0, None))) @ z.T
        lam, vec = numpy.linalg.eigh(root @ C @ root)
        modes = []
        for j in range(n):
            y = C @ root @ vec[:, j]
            dk = [
                numpy.zeros_like(k, dtype=complex if complex_perturbations else float)
                for k in blocks
            ]
            for ya, (b, o) in zip(y, ops):
                dk[b] -= ya * (o if complex_perturbations else o.real)
            modes.append((dk, float(y @ gamma @ y)))
        return 1.0 + lam, modes


def solve_multistart(problem, starts, tol=1e-10, max_iter=1000, max_escapes=5):
    """Solve from every start and leave saddle points along their unstable mode.

    For the unconstrained "uhf" and "ghf" layouts, a converged point whose
    lowest stability eigenvalue is negative is displaced along that mode (to
    max |dk| = 0.1) and solved again, at most `max_escapes` times. For "rhf"
    and "uhf_ph" the eigenvalue is only reported: their instabilities leave
    the constrained class.

    Parameters
    ----------
    problem : :class:`ThermalHubbardHF`
        Functional.
    starts : dict
        Initial variables by label.
    tol, max_iter :
        As in :meth:`ThermalHubbardHF.solve`.
    max_escapes : int
        Maximum number of saddle-point escapes per start. Default 5.

    Returns
    -------
    results : list
        (label, solution, lowest stability eigenvalue), sorted by free energy
        with unconverged solutions last.
    """
    results = []
    for label, variables in starts.items():
        sol = problem.solve(variables, tol=tol, max_iter=max_iter)
        lam = problem.stability(sol.variables)[0][0] if sol.converged else numpy.nan
        escapes = 0
        while (
            sol.converged
            and lam < -1e-8
            and problem.layout in ("uhf", "ghf")
            and escapes < max_escapes
        ):
            dk = problem.stability(sol.variables)[1][0][0]
            t = 0.1 / max(float(numpy.max(numpy.abs(d))) for d in dk)
            moved = [
                v + t * (d if numpy.iscomplexobj(v) else d.real) for v, d in zip(sol.variables, dk)
            ]
            sol = problem.solve(moved, tol=tol, max_iter=max_iter)
            lam = problem.stability(sol.variables)[0][0] if sol.converged else numpy.nan
            escapes += 1
        results.append((label, sol, lam))
    return sorted(results, key=lambda r: (not r[1].converged, r[1].free_energy))


def start_from_site_density(problem, n, m):
    r"""Roothaan start :math:`F[\rho_0] - \mu` from site densities and moments.

    :math:`\rho_0` is site diagonal with the spin blocks
    :math:`(n_i + m_i \cdot \sigma) / 2`; the spin layouts use :math:`m_z` only and
    "rhf" none of m.

    Parameters
    ----------
    problem : :class:`ThermalHubbardHF`
        Functional.
    n : :class:`numpy.ndarray`
        Site densities, shape (M,).
    m : :class:`numpy.ndarray`
        Site moments, shape (M, 3).

    Returns
    -------
    variables : list
        Initial variables.
    """
    M = problem.M
    n = numpy.asarray(n, dtype=float)
    m = numpy.asarray(m, dtype=float)
    if n.shape != (M,) or m.shape != (M, 3):
        raise ValueError("n must have shape (M,) and m shape (M, 3).")
    if problem.ghf:
        rho = numpy.zeros((2 * M, 2 * M), dtype=complex)
        idx = numpy.arange(M)
        rho[idx, idx] = 0.5 * (n + m[:, 2])
        rho[idx + M, idx + M] = 0.5 * (n - m[:, 2])
        rho[idx, idx + M] = 0.5 * (m[:, 0] - 1j * m[:, 1])
        rho[idx + M, idx] = 0.5 * (m[:, 0] + 1j * m[:, 1])
        if not m[:, 1].any() and numpy.isrealobj(problem.h[0]):
            rho = rho.real
        rho = [rho]
    else:
        mz = numpy.zeros(M) if problem.layout == "rhf" else m[:, 2]
        rho = [numpy.diag(0.5 * (n + mz)), numpy.diag(0.5 * (n - mz))]
        rho = [r.astype(numpy.result_type(r, problem.h[0])) for r in rho]
    _, fock = problem.energy_fock(rho)
    mu = 0.0 if problem.mu is None else problem.mu
    return problem._project([f - mu * numpy.eye(f.shape[0]) for f in fock])


def neel_gap_roots(h0, eta, U, beta):
    r"""Staggered moments of the half-filled two-sublattice thermal HF solutions.

    With :math:`k_{\uparrow, \downarrow} = h_0 \mp \Delta \eta`, :math:`\Delta = U m_s / 2`
    and :math:`\eta h_0 \eta = -h_0`, self-consistency reads

    .. math::

        m_s = \frac{\Delta}{M} \sum_k \frac{\tanh(\beta E_k / 2)}{E_k}, \qquad
        E_k^2 = \epsilon_k^2 + \Delta^2,

    with :math:`\epsilon_k` the eigenvalues of :math:`h_0`. Requires equal diagonal
    elements of :math:`\tanh(\beta E / 2) / E` (a uniform moment).

    Parameters
    ----------
    h0 : :class:`numpy.ndarray`
        Real symmetric bipartite one-body matrix, shape (M, M).
    eta : :class:`numpy.ndarray`
        Sublattice signs, shape (M,).
    U : float
        Hubbard interaction.
    beta : float
        Inverse temperature.

    Returns
    -------
    roots : :class:`numpy.ndarray`
        0 and every positive root :math:`m_s`.
    """
    h0 = numpy.asarray(h0)
    eta = numpy.asarray(eta, dtype=float)
    if numpy.iscomplexobj(h0) or not numpy.allclose(h0, h0.T, atol=1e-12, rtol=0):
        raise ValueError("h0 must be real symmetric.")
    if not numpy.allclose(eta[:, None] * h0 * eta[None, :], -h0, atol=1e-12, rtol=0):
        raise ValueError("h0 must be bipartite with respect to eta.")
    eps, vecs = numpy.linalg.eigh(h0)

    def moment(m):
        delta = 0.5 * U * m
        energy = numpy.sqrt(eps**2 + delta**2)
        diag = (vecs**2) @ (numpy.tanh(0.5 * beta * energy) / energy)
        if numpy.ptp(diag) > 1e-12 * numpy.abs(diag).max():
            raise ValueError("The uniform Neel ansatz needs equal diagonal elements.")
        return delta * diag[0]

    grid = numpy.linspace(1e-6, 1.0, 4001)
    vals = numpy.array([moment(m) - m for m in grid])
    roots = [0.0]
    for i in numpy.nonzero(numpy.sign(vals[:-1]) != numpy.sign(vals[1:]))[0]:
        lo, hi = grid[i], grid[i + 1]
        f_lo = vals[i]
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if mid in (lo, hi):
                break
            f_mid = moment(mid) - mid
            if numpy.sign(f_mid) == numpy.sign(f_lo):
                lo, f_lo = mid, f_mid
            else:
                hi = mid
        roots.append(0.5 * (lo + hi))
    return numpy.array(roots)


def neel_guides(h0, eta, U, m_s):
    r"""Longitudinal and transverse Neel guides at mu = 0.

    Parameters
    ----------
    h0, eta, U :
        As in :func:`neel_gap_roots`.
    m_s : float
        Staggered moment.

    Returns
    -------
    k_c : :class:`numpy.ndarray`
        Longitudinal guide, shape (2, M, M).
    k_d : :class:`numpy.ndarray`
        Transverse guide :math:`R_y(\pi/2) k_C R_y^\dagger`, shape (2M, 2M).
    """
    delta = 0.5 * U * m_s
    e = numpy.diag(numpy.asarray(eta, dtype=float))
    k_c = numpy.array([h0 - delta * e, h0 + delta * e])
    k_d = rotate_spin(spin_block_diag(k_c), spin_rotation(0.5 * numpy.pi, [0, 1, 0])).real
    return k_c, k_d
