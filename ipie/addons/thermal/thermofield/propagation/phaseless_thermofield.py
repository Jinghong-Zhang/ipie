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

import cmath
import math
import time

import numpy
import plum
import scipy.linalg

from ipie.addons.thermal.estimators.thermal import one_rdm_from_G
from ipie.addons.thermal.propagation.phaseless_base import construct_mean_field_shift
from ipie.addons.thermal.propagation.phaseless_generic import PhaselessGeneric
from ipie.addons.thermal.thermofield.hamiltonians.ueg import UEGSparseChol
from ipie.config import config
from ipie.hamiltonians.generic import GenericComplexChol, GenericRealChol
from ipie.propagation.operations import apply_exponential
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import synchronize, to_host


class ThermofieldPhaseless(PhaselessGeneric):
    r"""Phaseless propagation of thermofield walkers, :math:`\Delta \leftarrow B(x)\Delta`.

    The force bias and the importance factor come from the fixed guide of the
    trial: the weight update uses the guide overlap ratio
    :math:`S_T(\Delta')/S_T(\Delta)`, with the cosine projection of the
    thermal addon. Propagation acts on the Q factor of the factorized walkers.

    Parameters
    ----------
    time_step : float
        Imaginary time step.
    mu : float
        Chemical potential.
    exp_nmax : int
        Order of the Taylor expansion of the two-body propagator. Default 6.
    phaseless : bool
        Phaseless projection. False keeps complex weights (free projection).
        Default True.
    verbose : bool
        Print information. Default False.

    Attributes
    ----------
    batched : bool
        Propagate all walkers with stacked kernels; False uses the per-walker
        reference loop. Default True.
    reuse_spin_blocks : bool
        When both spin sectors (one-body propagator, guide and walker factors)
        are bitwise identical, compute spin 0 only and copy it to spin 1. The
        check runs every slice. Default True.
    real_chol_gemm : bool
        When every Cholesky block is purely real or purely imaginary, build VHS
        and the force bias with real GEMMs. The elementwise products are those
        of the complex GEMM, so only the accumulation order differs. Default True.
    group_accum : :class:`numpy.ndarray`
        When set, shape (nwalkers, nspin, nbasis, nbasis) with nspin 1 or 2,
        every slice propagator is multiplied into it, so a driver can store
        products of consecutive slices for cyclic averaging. With nspin 1 the
        spin-1 propagator must equal spin 0; a slice where it does not sets
        `group_accum_spin_mismatch`.
    record_B_slice : :class:`numpy.ndarray`
        When set, shape (nwalkers, 2, nbasis, nbasis), receives the slice
        propagators of the next slice.
    record_x_slice : :class:`numpy.ndarray`
        When set, shape (nwalkers, nfields), receives the shifted fields of the
        next slice.
    """

    def __init__(self, time_step, mu, exp_nmax=6, phaseless=True, verbose=False):
        super().__init__(time_step, mu, exp_nmax=exp_nmax, lowrank=False, verbose=verbose)
        self.phaseless = phaseless
        self.batched = True
        self.reuse_spin_blocks = True
        self.real_chol_gemm = True
        self.group_accum = None
        self.group_accum_spin_mismatch = False
        self.record_B_slice = None
        self.record_x_slice = None
        self._kernels_ready = False
        self._rc_L = None
        self._rc_Lt = None
        self._rc_blocks = None
        self._fb_L = None
        self._fb_P = None
        self._fb_I = None
        self._bh1_diag = None
        self._work_shape = None

    def build(self, hamiltonian, trial=None, walkers=None, mpi_handler=None, verbose=False):
        super().build(
            hamiltonian, trial=trial, walkers=walkers, mpi_handler=mpi_handler, verbose=verbose
        )
        self._kernels_ready = False

    def cast_to_cupy(self, verbose=False):
        """Copy the one-body propagator and the mean-field shift to the GPU."""
        if verbose:
            print(f"# {self.__class__.__name__}: copying the propagator to the GPU.")
        self.BH1 = xp.asarray(self.BH1)
        self.mf_shift = xp.asarray(self.mf_shift)
        self._kernels_ready = False
        self._fb_P = None
        self._work_shape = None

    # Compatibility with the per-array GPU interface of earlier versions.
    def to_device(self):
        """Copy the propagator to the GPU; needs IPIE_USE_GPU=1. Use `cast_to_cupy`."""
        if not config.get_option("use_gpu"):
            raise RuntimeError("to_device needs the GPU backend: set IPIE_USE_GPU=1.")
        self.cast_to_cupy()
        return self

    def _setup_kernels(self, hamiltonian):
        """Set up the real-GEMM layout and the diagonal one-body propagator once."""
        if self._kernels_ready:
            return
        self._kernels_ready = True
        self._setup_real_chol(hamiltonian)
        # A diagonal one-body propagator (plane waves with zero mean-field shift)
        # applies the Trotter split BH1 exp(VHS) BH1 as two elementwise
        # scalings, bitwise equal to the GEMMs since every off-diagonal product
        # there is an exact zero.
        BH1 = to_host(self.BH1)
        diag = numpy.array([numpy.diagonal(BH1[s]) for s in range(2)])
        self._bh1_diag = None
        if all(numpy.array_equal(BH1[s], numpy.diag(diag[s])) for s in range(2)):
            self._bh1_diag = xp.asarray(diag)

    @plum.dispatch
    def _setup_real_chol(self, hamiltonian: GenericRealChol):
        self._setup_real_chol_blocks([hamiltonian.chol])
        self._fb_L = numpy.ascontiguousarray(hamiltonian.chol)

    @plum.dispatch
    def _setup_real_chol(self, hamiltonian: GenericComplexChol):
        self._setup_real_chol_blocks([hamiltonian.A, hamiltonian.B])
        # Field-major Cholesky matrix [A | B], fields split as in construct_VHS.
        self._fb_L = numpy.concatenate([hamiltonian.A, hamiltonian.B], axis=1)

    @plum.dispatch
    def _setup_real_chol(self, hamiltonian: UEGSparseChol):
        pass

    def _setup_real_chol_blocks(self, mats):
        """Real-GEMM layout of the Cholesky blocks.

        Sets `_rc_blocks` to (phase, nfields) pairs with phase 1 or 1j such that
        block = phase * real matrix, `_rc_L` to the field-major real matrix of
        shape (nbasis^2, nfields) and `_rc_Lt` to its transpose. `_rc_L` stays
        None when a block is genuinely complex or `real_chol_gemm` is off.
        """
        self._rc_L = None
        if not self.real_chol_gemm:
            return
        blocks, parts = [], []
        for L in mats:
            L = numpy.asarray(L)
            if not numpy.iscomplexobj(L) or not L.imag.any():
                blocks.append((1.0, L.shape[1]))
                parts.append(numpy.ascontiguousarray(L.real))
            elif not L.real.any():
                blocks.append((1j, L.shape[1]))
                parts.append(numpy.ascontiguousarray(L.imag))
            else:
                return
        self._rc_blocks = blocks
        self._rc_L = numpy.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
        self._rc_Lt = numpy.ascontiguousarray(self._rc_L.T)

    @plum.dispatch
    def construct_one_body_propagator(self, hamiltonian: UEGSparseChol):
        r"""Mean-field shifted one-body propagator for the plane-wave Hamiltonian.

        .. math::

            e^{-\Delta\tau (H_1 - v_0 - \mu)/2}, \qquad
            v_0 = i \sum_n L_n \bar{v}_n,

        with the sparse Cholesky blocks.

        Parameters
        ----------
        hamiltonian : :class:`UEGSparseChol`
            Plane-wave Hamiltonian.

        Returns
        -------
        BH1 : :class:`numpy.ndarray`
            One-body propagator, shape (2, nbasis, nbasis).
        """
        nb = hamiltonian.nbasis
        nchol = hamiltonian.nchol
        shift = 1j * (
            hamiltonian.A.dot(self.mf_shift[:nchol]) + hamiltonian.B.dot(self.mf_shift[nchol:])
        ).reshape(nb, nb)
        muN = self.mu * numpy.identity(nb, dtype=hamiltonian.H1.dtype)
        H1 = hamiltonian.h1e_mod - numpy.array([shift + muN, shift + muN])
        return numpy.array(
            [scipy.linalg.expm(-0.5 * self.dt * H1[0]), scipy.linalg.expm(-0.5 * self.dt * H1[1])]
        )

    @plum.dispatch
    def construct_VHS(self, hamiltonian: UEGSparseChol, xshifted: xp.ndarray):
        r"""Two-body propagator exponent :math:`\sqrt{\Delta\tau}\, i \sum_n x_n L_n` (sparse).

        Parameters
        ----------
        hamiltonian : :class:`UEGSparseChol`
            Plane-wave Hamiltonian.
        xshifted : :class:`numpy.ndarray`
            Shifted fields, shape (nfields, nwalkers).

        Returns
        -------
        VHS : :class:`numpy.ndarray`
            Shape (nwalkers, nbasis, nbasis).
        """
        nwalkers = xshifted.shape[1]
        nbasis = hamiltonian.nbasis
        VHS = self.isqrt_dt * hamiltonian.construct_VHS_flat(xshifted).T
        return xp.ascontiguousarray(VHS).reshape(nwalkers, nbasis, nbasis)

    @plum.dispatch
    def construct_VHS(self, hamiltonian: GenericRealChol, xshifted: xp.ndarray):
        r"""Two-body propagator exponent :math:`\sqrt{\Delta\tau}\, i \sum_n x_n L_n`.

        Uses one real GEMM when every Cholesky block is real or imaginary (see
        `real_chol_gemm`), the complex product of the thermal addon otherwise.

        Parameters
        ----------
        hamiltonian : :class:`GenericRealChol`
            Hamiltonian.
        xshifted : :class:`numpy.ndarray`
            Shifted fields, shape (nfields, nwalkers).

        Returns
        -------
        VHS : :class:`numpy.ndarray`
            Shape (nwalkers, nbasis, nbasis).
        """
        return self._construct_VHS_real(hamiltonian, xshifted)

    @plum.dispatch
    def construct_VHS(self, hamiltonian: GenericComplexChol, xshifted: xp.ndarray):
        r"""Two-body propagator exponent :math:`\sqrt{\Delta\tau}\, i \sum_n x_n L_n`.

        Uses one real GEMM when every Cholesky block is real or imaginary (see
        `real_chol_gemm`), the complex product of the thermal addon otherwise.

        Parameters
        ----------
        hamiltonian : :class:`GenericComplexChol`
            Hamiltonian.
        xshifted : :class:`numpy.ndarray`
            Shifted fields, shape (nfields, nwalkers).

        Returns
        -------
        VHS : :class:`numpy.ndarray`
            Shape (nwalkers, nbasis, nbasis).
        """
        return self._construct_VHS_real(hamiltonian, xshifted)

    def _construct_VHS_real(self, hamiltonian, xshifted):
        # For a block L = phase Lr with Lr real and complex fields x,
        # L x = Lr (x.re + i x.im) for phase 1 and Lr (-x.im + i x.re) for
        # phase 1j, elementwise as in the complex product. Both blocks and both
        # components are stacked into one (2 nwalkers, nfields) x
        # (nfields, nbasis^2) GEMM.
        self._setup_kernels(hamiltonian)
        if self._rc_L is None:
            return super().construct_VHS(hamiltonian, xshifted)
        nfields, nwalkers = xshifted.shape
        nbasis = hamiltonian.nbasis
        X = numpy.empty((2 * nwalkers, nfields))  # Rows: real parts, then imaginary parts.
        start = 0
        for phase, nf in self._rc_blocks:
            xb = xshifted[start : start + nf].T
            if phase == 1.0:
                X[:nwalkers, start : start + nf] = xb.real
                X[nwalkers:, start : start + nf] = xb.imag
            else:
                numpy.negative(xb.imag, out=X[:nwalkers, start : start + nf])
                X[nwalkers:, start : start + nf] = xb.real
            start += nf
        Y = X.dot(self._rc_Lt)  # Shape (2 nwalkers, nbasis^2).
        VHS = self.isqrt_dt * (Y[:nwalkers] + 1j * Y[nwalkers:])
        return VHS.reshape(nwalkers, nbasis, nbasis)

    def _force_bias_real(self, P, nrows):
        """Rows of P (nrows, nbasis^2) times the Cholesky blocks with one real GEMM.

        (p.re + i p.im) Lr = p.re Lr + i p.im Lr for a real block and
        -p.im Lr + i p.re Lr for an imaginary one.
        """
        nbasis2 = P.shape[1]
        Pr = numpy.empty((2 * nrows, nbasis2))
        Pr[:nrows] = P.real
        Pr[nrows:] = P.imag
        R = Pr.dot(self._rc_L)  # Shape (2 nrows, nfields).
        vb = numpy.empty((nrows, R.shape[1]), dtype=numpy.complex128)
        start = 0
        for phase, nf in self._rc_blocks:
            sl = slice(start, start + nf)
            if phase == 1.0:
                vb[:, sl] = R[:nrows, sl] + 1j * R[nrows:, sl]
            else:
                vb[:, sl] = -R[nrows:, sl] + 1j * R[:nrows, sl]
            start += nf
        return vb

    def _spin_blocks_identical(self, walkers, trial):
        """True if the spin-0 and spin-1 inputs of the slice are bitwise equal.

        Deterministic BLAS and LAPACK give identical outputs for identical
        inputs, so spin 1 can then be copied from spin 0. The check is on the
        arrays actually used, and array_equal is False for NaN, so a broken
        walker falls back to the two-spin path.
        """
        return (
            bool(numpy.array_equal(self.BH1[0], self.BH1[1]))
            and bool(numpy.array_equal(trial.eps_T[0], trial.eps_T[1]))
            and bool(numpy.array_equal(trial.C_T[0], trial.C_T[1]))
            and bool(numpy.array_equal(trial.C_T_dagger[0], trial.C_T_dagger[1]))
            and bool(trial.log_det_C_T_dagger[0] == trial.log_det_C_T_dagger[1])
            and bool(numpy.array_equal(walkers.log_d[:, 0], walkers.log_d[:, 1]))
            and bool(numpy.array_equal(walkers.Tmat[:, 0], walkers.Tmat[:, 1]))
            and bool(numpy.array_equal(walkers.Qmat[:, 0], walkers.Qmat[:, 1]))
        )

    def _ensure_work_buffers(self, nwalkers, nbasis):
        """Allocate the work arrays of the batched path for this population size."""
        if self._work_shape == (nwalkers, nbasis):
            return
        shape = (nwalkers, nbasis, nbasis)
        self._BV = xp.empty(shape, dtype=xp.complex128)
        self._Temp = xp.empty(shape, dtype=xp.complex128)
        self._Temp2 = xp.empty(shape, dtype=xp.complex128)
        self._Bs = xp.empty(shape, dtype=xp.complex128)
        self._Q_new = xp.empty((nwalkers, 2, nbasis, nbasis), dtype=xp.complex128)
        self._work_shape = (nwalkers, nbasis)

    def _force_bias_buffers(self, nwalkers, nbasis):
        if self._fb_P is None or self._fb_P.shape[1:3] != (nwalkers, nbasis):
            self._fb_P = xp.empty((2, nwalkers, nbasis, nbasis), dtype=xp.complex128)
            self._fb_I = xp.identity(nbasis)
        return self._fb_P

    @plum.dispatch
    def construct_force_bias_batched(self, hamiltonian: UEGSparseChol, walkers):
        r"""Force bias :math:`\bar{v}_n = \mathrm{Tr}(L_n P_0) + \mathrm{Tr}(L_n P_1)` of all walkers.

        :math:`P_s = I - G_s^T` are the mixed one-body density matrices. Sparse
        plane-wave products.

        Parameters
        ----------
        hamiltonian : :class:`UEGSparseChol`
            Plane-wave Hamiltonian.
        walkers : :class:`ThermofieldWalkers`
            Walkers with cached Green's functions.

        Returns
        -------
        vbias : :class:`numpy.ndarray`
            Shape (nwalkers, nfields).
        """
        nwalkers = walkers.nwalkers
        nbasis = walkers.Ga.shape[-1]
        P = self._force_bias_buffers(nwalkers, nbasis)
        xp.subtract(self._fb_I, walkers.Ga.transpose(0, 2, 1), out=P[0])
        if self.reuse_spin_blocks and bool(xp.array_equal(walkers.Ga, walkers.Gb)):
            return 2.0 * hamiltonian.construct_force_bias_rows(
                P[0].reshape(nwalkers, nbasis * nbasis)
            )
        xp.subtract(self._fb_I, walkers.Gb.transpose(0, 2, 1), out=P[1])
        R = hamiltonian.construct_force_bias_rows(P.reshape(2 * nwalkers, nbasis * nbasis))
        return R[:nwalkers] + R[nwalkers:]

    @plum.dispatch
    def construct_force_bias_batched(self, hamiltonian: GenericRealChol, walkers):
        r"""Force bias :math:`\bar{v}_n = \mathrm{Tr}(L_n P_0) + \mathrm{Tr}(L_n P_1)` of all walkers.

        One GEMM over the flattened density matrices of both spins; each output
        element is the same dot product as the per-walker contraction.

        Parameters
        ----------
        hamiltonian : :class:`GenericRealChol`
            Hamiltonian.
        walkers : :class:`ThermofieldWalkers`
            Walkers with cached Green's functions.

        Returns
        -------
        vbias : :class:`numpy.ndarray`
            Shape (nwalkers, nfields).
        """
        return self._construct_force_bias_dense(hamiltonian, walkers)

    @plum.dispatch
    def construct_force_bias_batched(self, hamiltonian: GenericComplexChol, walkers):
        r"""Force bias :math:`\bar{v}_n = \mathrm{Tr}(L_n P_0) + \mathrm{Tr}(L_n P_1)` of all walkers.

        One GEMM over the flattened density matrices of both spins; each output
        element is the same dot product as the per-walker contraction.

        Parameters
        ----------
        hamiltonian : :class:`GenericComplexChol`
            Hamiltonian.
        walkers : :class:`ThermofieldWalkers`
            Walkers with cached Green's functions.

        Returns
        -------
        vbias : :class:`numpy.ndarray`
            Shape (nwalkers, nfields).
        """
        return self._construct_force_bias_dense(hamiltonian, walkers)

    def _construct_force_bias_dense(self, hamiltonian, walkers):
        self._setup_kernels(hamiltonian)
        nwalkers = walkers.nwalkers
        nbasis = walkers.Ga.shape[-1]
        P = self._force_bias_buffers(nwalkers, nbasis)
        real_gemm = self._rc_L is not None
        numpy.subtract(self._fb_I, walkers.Ga.transpose(0, 2, 1), out=P[0])
        if self.reuse_spin_blocks and numpy.array_equal(walkers.Ga, walkers.Gb):
            # Identical spins: Tr(L P_0) + Tr(L P_1) = 2 Tr(L P_0) exactly.
            P0 = P[0].reshape(nwalkers, nbasis * nbasis)
            R = self._force_bias_real(P0, nwalkers) if real_gemm else P0.dot(self._fb_L)
            return 2.0 * R
        numpy.subtract(self._fb_I, walkers.Gb.transpose(0, 2, 1), out=P[1])
        # Rows are [spin 0 walkers; spin 1 walkers]; the spins are added as in
        # the per-walker contraction.
        P01 = P.reshape(2 * nwalkers, nbasis * nbasis)
        R = self._force_bias_real(P01, 2 * nwalkers) if real_gemm else P01.dot(self._fb_L)
        return R[:nwalkers] + R[nwalkers:]

    def construct_two_body_propagator(self, walkers, hamiltonian, trial, debug=False):
        r"""Shifted fields, constant factors and VHS of one slice for all walkers.

        As in the thermal addon, with the batched force bias. One normal draw of
        shape (nwalkers, nfields) per slice.

        Returns
        -------
        cmf, cfb : :class:`numpy.ndarray`
            Mean-field and force-bias constant factors, shape (nwalkers,), on the
            host.
        xshifted : :class:`numpy.ndarray`
            Shifted fields, shape (nfields, nwalkers).
        VHS : :class:`numpy.ndarray`
            Shape (nwalkers, nbasis, nbasis).
        """
        start_time = time.time()
        self.vbias = self.construct_force_bias_batched(hamiltonian, walkers)
        xbar = -self.sqrt_dt * (1j * self.vbias - self.mf_shift)
        synchronize()
        self.timer.tfbias += time.time() - start_time

        # Force bias bounding.
        xbar = self.apply_bound_force_bias(xbar, self.fbbound)

        # Normally distributed auxiliary fields.
        xi = xp.random.normal(0.0, 1.0, hamiltonian.nfields * walkers.nwalkers).reshape(
            walkers.nwalkers, hamiltonian.nfields
        )

        if debug:
            self.xi = xi  # For debugging.
        xshifted = xi - xbar  # Shape (nwalkers, nfields).

        # Constant factor arising from force bias and mean field shift.
        cmf = -self.sqrt_dt * xp.einsum("wx,x->w", xshifted, self.mf_shift)  # Shape (nwalkers,).
        # Constant factor arising from shifting the probability distribution.
        cfb = xp.einsum("wx,wx->w", xi, xbar) - 0.5 * xp.einsum(
            "wx,wx->w", xbar, xbar
        )  # Shape (nwalkers,).

        xshifted = xshifted.T.copy()  # Shape (nfields, nwalkers).
        start_time = time.time()
        VHS = self.construct_VHS(hamiltonian, xshifted)  # Shape (nwalkers, nbasis, nbasis).
        synchronize()
        self.timer.tvhs += time.time() - start_time
        return to_host(cmf), to_host(cfb), xshifted, VHS

    def propagate_walkers(self, walkers, hamiltonian, trial, eshift=0.0, debug=False):
        """Propagate all walkers by one time slice.

        Parameters
        ----------
        walkers : :class:`ThermofieldWalkers`
            Walkers, updated in place.
        hamiltonian :
            Hamiltonian.
        trial : :class:`ThermofieldThermalTrial`
            Guide for the force bias and the importance factor.
        eshift : float
            Unused; the guide ratio carries the normalization.
        debug : bool
            Keep the field draws in `xi`. Default False.
        """
        cmf, cfb, xshifted, VHS = self.construct_two_body_propagator(
            walkers, hamiltonian, trial, debug=debug
        )
        assert walkers.nwalkers == xshifted.shape[-1]
        if self.record_x_slice is not None:
            self.record_x_slice[:] = to_host(xshifted.T)
        if self.batched:
            self._propagate_walkers_batched(walkers, trial, VHS, cfb, cmf)
        else:
            self._propagate_walkers_scalar(walkers, trial, VHS, cfb, cmf)

    def _propagate_walkers_batched(self, walkers, trial, VHS, cfb, cmf):
        """Propagate the whole population with stacked kernels.

        Every stacked matmul is the per-matrix GEMM of the scalar path in the
        same order, so the two differ only by the rounding of the elementwise
        weight arithmetic.
        """
        start_time = time.time()
        nwalkers = walkers.nwalkers
        nbasis = VHS.shape[-1]
        self._ensure_work_buffers(nwalkers, nbasis)
        BV, Temp, Temp2 = self._BV, self._Temp, self._Temp2

        # BV = exp(VHS) by the truncated Taylor series of apply_exponential. The
        # first-order term VHS @ I is VHS exactly, so it is not multiplied out.
        BV[:] = xp.eye(nbasis)
        if self.exp_nmax >= 1:
            Temp[:] = VHS
            BV += Temp
        for n in range(2, self.exp_nmax + 1):
            xp.matmul(VHS, Temp, out=Temp2)
            Temp2 /= n
            BV += Temp2
            Temp, Temp2 = Temp2, Temp

        # B Delta acts on the Q factor of the walkers.
        Q_new = self._Q_new
        Bs = self._Bs
        bh1_diag = self._bh1_diag
        reuse = self.reuse_spin_blocks and self._spin_blocks_identical(walkers, trial)
        acc = self.group_accum
        record = self.record_B_slice
        for s in range(2):
            if reuse and s == 1:
                # Spin-1 inputs equal spin 0 bitwise: copy.
                if record is not None:
                    record[:, 1] = to_host(Bs)
                if acc is not None and acc.shape[1] == 2:
                    xp.matmul(Bs, acc[:, 1], out=Temp)
                    acc[:, 1] = Temp
                Q_new[:, 1] = Q_new[:, 0]
                break
            if bh1_diag is not None:
                # (BH1 BV BH1)_ij = (BH1_ii BV_ij) BH1_jj, the association of the
                # GEMMs, hence bitwise equal.
                dg = bh1_diag[s]
                xp.multiply(dg[:, None], BV, out=Temp)
                xp.multiply(Temp, dg[None, :], out=Bs)
            else:
                xp.matmul(self.BH1[s], BV, out=Temp)
                xp.matmul(Temp, self.BH1[s], out=Bs)  # Symmetric Trotter split.
            if record is not None:
                record[:, s] = to_host(Bs)
            if acc is not None:
                if s < acc.shape[1]:
                    xp.matmul(Bs, acc[:, s], out=Temp)
                    acc[:, s] = Temp
                elif not reuse:
                    self.group_accum_spin_mismatch = True
            xp.matmul(Bs, walkers.Qmat[:, s], out=Q_new[:, s])
        synchronize()
        self.timer.tgemm += time.time() - start_time

        start_time = time.time()
        log_ovlp_new, G_new = trial.calc_log_overlap_and_greens_function_batched(
            Q_new, walkers.log_d, walkers.Tmat, reuse_spin_blocks=reuse
        )
        synchronize()
        self.timer.tovlp += time.time() - start_time

        start_time = time.time()
        self.update_walker_weights_batched(walkers, log_ovlp_new, cfb, cmf)
        # Commit every walker, including those just given zero weight, and the
        # cached overlap and Green's function used by the next force bias.
        walkers.Qmat[:] = Q_new
        walkers.set_walker_cache_batched(log_ovlp_new, G_new)
        synchronize()
        self.timer.tupdate += time.time() - start_time

    def update_walker_weights_batched(self, walkers, log_ovlp_new, cfb, cmf):
        """Update the weights of all walkers, as :meth:`update_walker_weight`.

        A walker whose importance factor is not finite gets zero weight.

        Parameters
        ----------
        walkers : :class:`ThermofieldWalkers`
            Walkers, updated in place.
        log_ovlp_new : :class:`numpy.ndarray`
            Guide log overlaps after the slice, shape (nwalkers,).
        cfb, cmf : :class:`numpy.ndarray`
            Force-bias and mean-field constant factors, shape (nwalkers,).
        """
        log_R = to_host(log_ovlp_new) - walkers.log_ovlp + cfb + cmf
        with numpy.errstate(over="ignore", invalid="ignore", divide="ignore"):
            expQ = self.mf_const_fac * numpy.exp(log_R)
            magn = numpy.abs(expQ)
            bad = numpy.isinf(magn) | numpy.isnan(magn)
            good = ~bad
            if self.phaseless:
                # Cosine projection with the phase of the guide ratio and cmf,
                # without the factor from shifting the distribution.
                dtheta = numpy.angle(numpy.exp(log_R - cfb))
                cosine_fac = numpy.maximum(0.0, numpy.cos(dtheta))
                walkers.weight[good] *= (magn * cosine_fac)[good]
            else:
                walkers.weight[good] *= magn[good]
                nz = good & (magn > 0.0)
                walkers.phase[nz] *= expQ[nz] / magn[nz]
        walkers.weight[bad] = 0.0

    def _propagate_walkers_scalar(self, walkers, trial, VHS, cfb, cmf):
        """Per-walker reference implementation (`batched` False)."""
        record = self.record_B_slice
        for iw in range(walkers.nwalkers):
            phi = xp.identity(VHS[iw].shape[-1], dtype=xp.complex128)
            BV = apply_exponential(phi, VHS[iw], self.exp_nmax)  # exp(VHS[iw]).
            Q_new = numpy.zeros_like(walkers.Qmat[iw])
            for s in range(2):
                B = self.BH1[s] @ BV @ self.BH1[s]  # Symmetric Trotter split.
                if record is not None:
                    record[iw, s] = B
                Q_new[s] = B @ walkers.Qmat[iw, s]

            log_ovlp_new, G_new = trial.calc_log_overlap_and_greens_function(
                Q_new, walkers.log_d[iw], walkers.Tmat[iw]
            )
            self.update_walker_weight(walkers, iw, log_ovlp_new, cfb[iw], cmf[iw])

            walkers.Qmat[iw] = Q_new
            walkers.set_walker_cache(iw, log_ovlp_new, G_new)

    def update_walker_weight(self, walkers, iw, log_ovlp_new, cfb, cmf):
        r"""Update the weight of walker `iw` with the hybrid update of the thermal addon.

        The importance factor is the guide ratio
        :math:`S_T(\Delta')/S_T(\Delta)` in log space times
        :math:`e^{-\Delta\tau E_{mf}}`,
        :math:`E_{mf} = E_{core} + \bar{v} \cdot \bar{v} / 2`, as in
        `PhaselessBase.update_weight_legacy`.

        Parameters
        ----------
        walkers : :class:`ThermofieldWalkers`
            Walkers, updated in place.
        iw : int
            Walker index.
        log_ovlp_new : complex
            Guide log overlap after the slice.
        cfb, cmf : complex
            Force-bias and mean-field constant factors.
        """
        log_oratio = log_ovlp_new - walkers.log_ovlp[iw]
        log_R = log_oratio + cfb + cmf

        try:
            expQ = self.mf_const_fac * cmath.exp(log_R)
            magn = abs(expQ)
        except OverflowError:
            magn = math.inf

        if math.isinf(magn) or math.isnan(magn):
            walkers.weight[iw] = 0.0
            return

        if self.phaseless:
            # Cosine projection with the phase of the guide ratio and cmf.
            dtheta = cmath.phase(cmath.exp(log_R - cfb))
            cosine_fac = max(0.0, math.cos(dtheta))
            walkers.weight[iw] *= magn * cosine_fac
        else:
            walkers.weight[iw] *= magn
            if magn > 0.0:
                walkers.phase[iw] *= expQ / magn


@construct_mean_field_shift.dispatch
def construct_mean_field_shift(hamiltonian: UEGSparseChol, trial):
    r"""Compute mean field shift.

    .. math::

        \bar{v}_n = \sum_{ik\sigma} v_{(ik),n} P_{ik\sigma}

    """
    P = one_rdm_from_G(trial.G)
    P = (P[0] + P[1]).ravel()
    nchol = hamiltonian.nchol
    mf_shift = numpy.zeros(hamiltonian.nfields, dtype=numpy.complex128)
    mf_shift[:nchol] = 1j * hamiltonian.AT.dot(P)
    mf_shift[nchol:] = 1j * hamiltonian.BT.dot(P)
    return mf_shift  # Shape (nfields,).
