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

from ipie.addons.thermal.estimators.particle_number import particle_number
from ipie.addons.thermal.thermofield.utils.linalg import (
    inv_stack,
    one_plus_qdy,
    one_plus_qdy_batched,
    principal_log_phase,
    stabilized_inverse_one_plus,
    stabilized_inverse_one_plus_batched,
)
from ipie.utils.backend import arraylib as xp


class ThermofieldThermalTrial:
    r"""Fixed one-body guide for thermofield walkers.

    The guide closes an open walker :math:`\Delta` with the one-body operator
    :math:`\Lambda = e^{-\tau_g k_T}`,

    .. math::

        S_T(\Delta) = \prod_s \det(I + \Lambda_s \Delta_s), \qquad
        G_T(\Delta) = \Delta (I + \Lambda \Delta)^{-1} \Lambda,

    with :math:`\langle c_i^\dagger c_j \rangle = G_{ji}`. Guide times
    :math:`\tau_g = \beta / 2` (the default) and :math:`\tau_g = \beta` (matched to
    the end of a path over :math:`[0, \beta]`) are both evaluated in the guide
    eigenbasis without forming exponentially large factors, so the projector
    limit is numerically stable.

    Parameters
    ----------
    hamiltonian :
        Hamiltonian; sets the default :math:`k_T = H_1 - \mu`.
    beta : float
        Inverse temperature. Sets the thermal occupations of the static
        mean-field subtraction.
    mu : float
        Chemical potential. Ignored when `k_trial` is given. Default 0.0.
    k_trial : :class:`numpy.ndarray`
        Hermitian one-body matrix :math:`k_T` with the chemical potential
        included, shape (2, nbasis, nbasis). Default :math:`H_1 - \mu`.
    verbose : bool
        Print information. Default False.
    guide_time : float
        Guide time :math:`\tau_g`. Default beta / 2.
    force_bias_split : str
        Where the cached Green's function for the force bias is evaluated:
        "frontier" (default), at the walker frontier,
        :math:`(I + \Delta\Lambda)^{-1}`; or "zero", at the trace split point,
        :math:`(I + \Lambda\Delta)^{-1}`, the convention of the thermal addon.
        Overlaps do not depend on the split, so only the force bias changes.

    Attributes
    ----------
    eps_T, C_T : :class:`numpy.ndarray`
        Eigenvalues and eigenvectors of :math:`k_T`.
    diagonal_guide : bool
        True when :math:`k_T` is diagonal, so the guide eigenbasis is the basis
        itself and the kernels skip the eigenvector products.
    dmat : :class:`numpy.ndarray`
        :math:`e^{-\tau_g k_T}` formed directly; may overflow for large
        arguments and is not used by the stabilized kernels.
    P : :class:`numpy.ndarray`
        One-body density matrix at beta, :math:`P_{ij} = \langle c_i^\dagger c_j\rangle`.
    G : :class:`numpy.ndarray`
        Thermal hole Green's function :math:`I - P^T`, as consumed by the
        thermal mean-field shift.
    nav : float
        Average particle number at beta.
    """

    def __init__(
        self,
        hamiltonian,
        beta,
        mu=0.0,
        k_trial=None,
        verbose=False,
        guide_time=None,
        force_bias_split="frontier",
    ):
        self.name = "thermofield"
        self.compute_trial_energy = False
        self.verbose = verbose
        self.beta = beta
        self.mu = mu
        if guide_time is None:
            guide_time = 0.5 * beta
        self.guide_time = float(self._check_guide_time(guide_time))
        if force_bias_split not in ("frontier", "zero"):
            raise ValueError(f"Unknown force bias split {force_bias_split}.")
        self.force_bias_split = force_bias_split
        self._lambda_cache = {}
        self._guide_device = None

        nbasis = hamiltonian.nbasis
        self.nbasis = nbasis

        if k_trial is None:
            muN = mu * numpy.eye(nbasis, dtype=hamiltonian.H1.dtype)
            k_trial = numpy.array([hamiltonian.H1[0] - muN, hamiltonian.H1[1] - muN])
        k_trial = numpy.asarray(k_trial)
        assert k_trial.shape == (2, nbasis, nbasis)
        for s in range(2):
            if not numpy.allclose(k_trial[s], k_trial[s].conj().T, atol=1e-10):
                raise ValueError("k_trial must be Hermitian for the thermofield trial.")
        self.k_trial = k_trial

        # k_T = C diag(eps) C^dagger with real eps.
        eps = numpy.zeros((2, nbasis))
        C = numpy.zeros((2, nbasis, nbasis), dtype=numpy.complex128)
        for s in range(2):
            eps[s], C[s] = numpy.linalg.eigh(k_trial[s])
        # eigh may permute or rotate the columns of a diagonal matrix; use the
        # identity basis for a diagonal k_T, which is exact.
        for s in range(2):
            if numpy.allclose(
                k_trial[s], numpy.diag(numpy.diagonal(k_trial[s])), atol=0.0, rtol=0.0
            ):
                eps[s] = numpy.diagonal(k_trial[s]).real
                C[s] = numpy.eye(nbasis)
        self.diagonal_guide = all(numpy.array_equal(C[s], numpy.eye(nbasis)) for s in range(2))
        self.eps_T = eps
        self.C_T = C
        self.C_T_dagger = C.conj().transpose(0, 2, 1)
        self.log_det_C_T_dagger = numpy.zeros(2, dtype=numpy.complex128)
        for s in range(2):
            sign, logabs = numpy.linalg.slogdet(self.C_T_dagger[s])
            self.log_det_C_T_dagger[s] = logabs + numpy.log(sign)

        self.dmat = numpy.array(
            [
                C[s] @ numpy.diag(numpy.exp(-self.guide_time * eps[s])) @ C[s].conj().T
                for s in range(2)
            ]
        )

        # Occupations at beta for the static mean-field subtraction,
        # f = (I + e^{beta k_T})^{-1}; independent of the guide time.
        occ = 0.5 * (1.0 - numpy.tanh(0.5 * beta * eps))
        f = numpy.array([C[s] @ numpy.diag(occ[s]) @ C[s].conj().T for s in range(2)])
        # P_ij = <c_i^dagger c_j> = f_ji; the transpose matters for complex k_T.
        self.P = numpy.array([f[0].T, f[1].T])
        I = numpy.eye(nbasis)
        self.G = numpy.array([I - f[0], I - f[1]])
        self.nav = particle_number(self.P).real

        if verbose:
            print("# Building ThermofieldThermalTrial.")
            print(f"# beta = {beta}")
            print(f"# Guide time = {self.guide_time}")
            print(f"# mu = {mu}")
            print(f"# Trial average particle number: {self.nav}")

    @property
    def eps_T(self):
        return self._eps_T

    @eps_T.setter
    def eps_T(self, value):
        self._eps_T = value
        self._lambda_cache = {}

    @property
    def C_T(self):
        return self._C_T

    @C_T.setter
    def C_T(self, value):
        self._C_T = value
        self._lambda_cache = {}
        self._guide_device = None

    @staticmethod
    def _check_guide_time(guide_time):
        if guide_time < 0.0:
            raise ValueError("guide_time must be non-negative.")
        return guide_time

    def cast_to_cupy(self, verbose=False):
        """Copy the guide eigenvectors to the GPU; the rest stays on the host."""
        self._guide_arrays(xp)

    def _guide_lambda(self, s, guide_time, xp=numpy):
        r""":math:`\Lambda_s = C e^{-\tau_g \epsilon} C^\dagger` as an array of module `xp`.

        Returns the diagonal of length nbasis when `diagonal_guide` is set, the
        full matrix otherwise. Cached per spin, guide time and backend.
        """
        # String keys keep the trial serializable by `to_json`.
        key = f"{s} {float(guide_time)!r} {'host' if xp is numpy else 'device'}"
        lam = self._lambda_cache.get(key)
        if lam is None:
            lam = numpy.exp(-guide_time * self.eps_T[s])
            if not self.diagonal_guide:
                lam = (self.C_T[s] * lam[None, :]) @ self.C_T_dagger[s]
            if xp is not numpy:
                lam = xp.asarray(lam)
            # Completion guides sweep the guide time; keep the cache small.
            if len(self._lambda_cache) >= 16:
                self._lambda_cache.clear()
            self._lambda_cache[key] = lam
        return lam

    @staticmethod
    def _apply_lambda(Tmat_s, lam, xp=numpy):
        """T Lambda for a diagonal (1-d) or full (2-d) Lambda; T = I when `Tmat_s` is None."""
        if Tmat_s is None:
            return xp.diag(lam) if lam.ndim == 1 else lam
        if lam.ndim == 1:
            return Tmat_s * lam[..., None, :] if Tmat_s.ndim == 3 else Tmat_s * lam[None, :]
        return Tmat_s @ lam

    def _spin_block(self, Qmat, log_d, Tmat, s, guide_time):
        r"""Log determinant and Green's function of :math:`I + \Delta\Lambda` for spin `s`.

        With :math:`\Delta = Q e^{D} T`, :math:`\Delta\Lambda = Q e^{D}(T\Lambda)` is
        evaluated with the one-sided kernel :func:`one_plus_qdy`, since
        :math:`T\Lambda` is bounded. By Sylvester's identity
        :math:`\det(I + \Lambda\Delta) = \det(I + \Delta\Lambda)` and
        :math:`G = I - (I + \Delta\Lambda)^{-1}`.
        """
        d = numpy.zeros(self.nbasis) if log_d is None else log_d[s]
        Y = self._apply_lambda(None if Tmat is None else Tmat[s], self._guide_lambda(s, guide_time))
        return one_plus_qdy(Qmat[s], d, Y)

    def _spin_block_split_zero(self, Qmat, log_d, Tmat, s, guide_time):
        r"""Log determinant and inverse of :math:`I + \Lambda\Delta` for spin `s`.

        The trace-split counterpart of :meth:`_spin_block`:
        :math:`\Lambda\Delta = C e^{-A} (C^\dagger Q) e^{D} T` through the
        two-sided kernel. The log determinant equals the frontier one; the
        inverse is the hole Green's function at the trace split point.
        """
        a = guide_time * self.eps_T[s]
        C = self.C_T[s]
        d = numpy.zeros(self.nbasis) if log_d is None else log_d[s]
        X = self.C_T_dagger[s] @ Qmat[s]
        if Tmat is None:
            V = numpy.eye(self.nbasis)
            Vinv = V
            log_det_V = 0.0 + 0.0j
        else:
            V = Tmat[s]
            Vinv = numpy.linalg.inv(V)
            sign, logabs = numpy.linalg.slogdet(V)
            log_det_V = logabs + numpy.log(sign + 0.0j)
        return stabilized_inverse_one_plus(C, -a, X, d, V, Vinv=Vinv, log_det_V=log_det_V)

    def calc_log_overlap(self, Qmat, log_d=None, Tmat=None, guide_time=None):
        r"""Complex log of the guide overlap :math:`S_T(\Delta)` of one walker.

        Parameters
        ----------
        Qmat : :class:`numpy.ndarray`
            Walker Q factors, shape (2, nbasis, nbasis).
        log_d : :class:`numpy.ndarray`
            Log scales, shape (2, nbasis). Default zero.
        Tmat : :class:`numpy.ndarray`
            Walker T factors, shape (2, nbasis, nbasis). Default identity.
        guide_time : float
            Guide time. Default the fixed guide time; 0 gives the closed trace
            :math:`\det(I + \Delta)`.

        Returns
        -------
        log_ovlp : complex
            :math:`\sum_s \log\det(I + \Lambda_s \Delta_s)`.
        """
        log_ovlp, _ = self.calc_log_overlap_and_greens_function(
            Qmat, log_d, Tmat, guide_time=guide_time
        )
        return log_ovlp

    def calc_log_overlap_and_greens_function(
        self, Qmat, log_d=None, Tmat=None, guide_time=None, split=None
    ):
        r"""Guide log overlap and mixed Green's function of one walker.

        Both come from the same factorization per spin.

        Parameters
        ----------
        Qmat : :class:`numpy.ndarray`
            Walker Q factors, shape (2, nbasis, nbasis).
        log_d : :class:`numpy.ndarray`
            Log scales, shape (2, nbasis). Default zero.
        Tmat : :class:`numpy.ndarray`
            Walker T factors, shape (2, nbasis, nbasis). Default identity.
        guide_time : float
            Guide time. Default the fixed guide time.
        split : str
            "frontier" or "zero". Default `force_bias_split`.

        Returns
        -------
        log_ovlp : complex
            Log overlap, imaginary part in (-pi, pi].
        G : :class:`numpy.ndarray`
            Green's function in the particle convention, shape (2, nbasis, nbasis).
        """
        guide_time = self._check_guide_time(self.guide_time if guide_time is None else guide_time)
        split = self.force_bias_split if split is None else split
        log_ovlp = 0.0 + 0.0j
        G = numpy.zeros((2, self.nbasis, self.nbasis), dtype=numpy.complex128)
        I = numpy.eye(self.nbasis)
        for s in range(2):
            if split == "zero":
                log_det, inv = self._spin_block_split_zero(Qmat, log_d, Tmat, s, guide_time)
                G[s] = I - inv
            else:
                log_det, G[s] = self._spin_block(Qmat, log_d, Tmat, s, guide_time)
            log_ovlp += log_det
        return principal_log_phase(log_ovlp), G

    def _guide_arrays(self, xp):
        """C_T and C_T^dagger as arrays of module `xp` (device copies cached)."""
        if xp is numpy:
            return self.C_T, self.C_T_dagger
        if self._guide_device is None:
            self._guide_device = (xp.asarray(self.C_T), xp.asarray(self.C_T_dagger))
        return self._guide_device

    def calc_log_overlap_and_greens_function_batched(
        self, Qmat, log_d=None, Tmat=None, guide_time=None, split=None, reuse_spin_blocks=False
    ):
        r"""Walker-batched :meth:`calc_log_overlap_and_greens_function`.

        Same arithmetic as the single-walker method, so the results agree bit for
        bit. A singular walker gives non-finite entries instead of raising;
        callers must kill it.

        Parameters
        ----------
        Qmat : :class:`numpy.ndarray`
            Walker Q factors, shape (nwalkers, 2, nbasis, nbasis).
        log_d : :class:`numpy.ndarray`
            Log scales, shape (nwalkers, 2, nbasis). Default zero.
        Tmat : :class:`numpy.ndarray`
            Walker T factors, shape (nwalkers, 2, nbasis, nbasis). Default identity.
        guide_time : float
            Guide time. Default the fixed guide time.
        split : str
            "frontier" or "zero". Default `force_bias_split`.
        reuse_spin_blocks : bool
            The caller has checked that both spin blocks (walker factors and
            guide) are bitwise identical; spin 1 is then copied from spin 0,
            which reproduces the two-spin result exactly. Default False.

        Returns
        -------
        log_ovlp : :class:`numpy.ndarray`
            Log overlaps, shape (nwalkers,).
        G : :class:`numpy.ndarray`
            Green's functions in the particle convention, shape
            (nwalkers, 2, nbasis, nbasis).
        """
        guide_time = self._check_guide_time(self.guide_time if guide_time is None else guide_time)
        split = self.force_bias_split if split is None else split
        Qmat = xp.asarray(Qmat)
        log_d = None if log_d is None else xp.asarray(log_d)
        Tmat = None if Tmat is None else xp.asarray(Tmat)
        nwalkers = Qmat.shape[0]
        nbasis = self.nbasis
        I = xp.eye(nbasis)
        log_ovlp = xp.zeros(nwalkers, dtype=xp.complex128)
        G = xp.zeros((nwalkers, 2, nbasis, nbasis), dtype=xp.complex128)
        C_T, C_T_dagger = self._guide_arrays(xp)
        for s in range(1 if reuse_spin_blocks else 2):
            a = xp.asarray(guide_time * self.eps_T[s])
            C = C_T[s]
            d = xp.zeros((nwalkers, nbasis)) if log_d is None else log_d[:, s]
            if split == "zero":
                # Lambda Delta = C e^{-A} (C^dagger Q) e^{D} T; C does not depend
                # on the walker, so its inverse and log det are formed once.
                X = C_T_dagger[s] @ Qmat[:, s]
                if Tmat is None:
                    Vinv = I
                    log_det_V = 0.0 + 0.0j
                else:
                    V = Tmat[:, s]
                    Vinv = inv_stack(V)
                    with numpy.errstate(divide="ignore", invalid="ignore"):
                        sign, logabs = xp.linalg.slogdet(V)
                        log_det_V = logabs + xp.log(sign + 0.0j)
                sign_c, logabs_c = xp.linalg.slogdet(C)
                log_det, inv = stabilized_inverse_one_plus_batched(
                    C,
                    -a,
                    X,
                    d,
                    Vinv,
                    log_det_V,
                    Qinv=xp.linalg.inv(C),
                    log_det_Q=logabs_c + xp.log(sign_c),
                )
                G[:, s] = I - inv
            else:
                # Delta Lambda = Q e^{D} (T Lambda) through the one-sided kernel.
                Y = self._apply_lambda(
                    None if Tmat is None else Tmat[:, s], self._guide_lambda(s, guide_time, xp), xp
                )
                log_det, G[:, s] = one_plus_qdy_batched(Qmat[:, s], d, Y)
            log_ovlp += log_det
        if reuse_spin_blocks:
            # (0 + ld0) + ld0 equals the two-spin sum (0 + ld0) + ld1 when ld1 == ld0.
            log_ovlp += log_det
            G[:, 1] = G[:, 0]
        return principal_log_phase(log_ovlp), G

    def calc_log_trace_batched(self, Qmat, log_d=None, Tmat=None):
        r"""Closed-trace log weight :math:`\sum_s \log\det(I + \Delta_s)` of every walker.

        The log determinant of the one-sided kernel with :math:`Y = T`, used by
        trace-magnitude branching.

        Parameters
        ----------
        Qmat : :class:`numpy.ndarray`
            Walker Q factors, shape (nwalkers, 2, nbasis, nbasis).
        log_d : :class:`numpy.ndarray`
            Log scales, shape (nwalkers, 2, nbasis). Default zero.
        Tmat : :class:`numpy.ndarray`
            Walker T factors, shape (nwalkers, 2, nbasis, nbasis). Default identity.

        Returns
        -------
        log_tr : :class:`numpy.ndarray`
            Log trace weights, shape (nwalkers,).
        """
        Qmat = xp.asarray(Qmat)
        log_d = None if log_d is None else xp.asarray(log_d)
        Tmat = None if Tmat is None else xp.asarray(Tmat)
        nwalkers = Qmat.shape[0]
        nbasis = self.nbasis
        I = xp.eye(nbasis)
        log_tr = xp.zeros(nwalkers, dtype=xp.complex128)
        for s in range(2):
            d = xp.zeros((nwalkers, nbasis)) if log_d is None else log_d[:, s]
            Y = I if Tmat is None else Tmat[:, s]
            log_det, _ = one_plus_qdy_batched(Qmat[:, s], d, Y, need_inverse=False)
            log_tr += log_det
        return principal_log_phase(log_tr)

    def calc_greens_function(self, Qmat, log_d=None, Tmat=None):
        r"""Mixed Green's function :math:`G_T(\Delta)` of one walker.

        Parameters
        ----------
        Qmat : :class:`numpy.ndarray`
            Walker Q factors, shape (2, nbasis, nbasis).
        log_d : :class:`numpy.ndarray`
            Log scales, shape (2, nbasis). Default zero.
        Tmat : :class:`numpy.ndarray`
            Walker T factors, shape (2, nbasis, nbasis). Default identity.

        Returns
        -------
        G : :class:`numpy.ndarray`
            Green's function with :math:`\langle c_i^\dagger c_j\rangle = G_{ji}`,
            shape (2, nbasis, nbasis).
        """
        _, G = self.calc_log_overlap_and_greens_function(Qmat, log_d, Tmat)
        return G

    # Former explicit-time names (default: the closed trace), kept for compatibility.
    def calc_log_overlap_at_time(self, Qmat, log_d=None, Tmat=None, guide_time=0.0):
        return self.calc_log_overlap(Qmat, log_d, Tmat, guide_time=guide_time)

    def calc_log_overlap_and_greens_function_at_time(
        self, Qmat, log_d=None, Tmat=None, guide_time=0.0
    ):
        return self.calc_log_overlap_and_greens_function(Qmat, log_d, Tmat, guide_time=guide_time)
