import time
from typing import Optional

import numpy
import plum

from ipie.config import CommType, config, MPI
from ipie.estimators.generic import half_rotated_cholesky_jk
from ipie.estimators.greens_function_single_det import (
    greens_function_single_det,
    greens_function_single_det_batch,
)
from ipie.estimators.utils import gab_spin
from ipie.hamiltonians.generic import GenericComplexChol, GenericRealChol
from ipie.hamiltonians.generic_chunked import GenericRealCholChunked
from ipie.hamiltonians.thc import GenericRealTHC, GenericRealTHCUhf, GenericComplexTHC
from ipie.propagation.force_bias import (
    construct_force_bias_batch_single_det,
    construct_force_bias_batch_single_det_chunked,
)
from ipie.propagation.overlap import calc_overlap_single_det_uhf
from ipie.trial_wavefunction.half_rotate import half_rotate_generic, half_rotate_chunked
from ipie.trial_wavefunction.wavefunction_base import TrialWavefunctionBase
from ipie.utils.backend import arraylib as xp
from ipie.utils.mpi import MPIHandler
from ipie.walkers.uhf_walkers import UHFWalkers
from typing import Union


# class for UHF trial
class SingleDet(TrialWavefunctionBase):
    def __init__(self, wavefunction, num_elec, num_basis, handler=MPIHandler(), verbose=False):
        assert isinstance(wavefunction, numpy.ndarray)
        assert len(wavefunction.shape) == 2
        super().__init__(wavefunction, num_elec, num_basis, verbose=verbose)
        if verbose:
            print("# Parsing input options for trial_wavefunction.MultiSlater.")
        self.psi = wavefunction
        self.num_elec = num_elec
        self._num_dets = 1
        self._max_num_dets = 1
        imag_norm = numpy.sum(self.psi.imag.ravel() * self.psi.imag.ravel())
        if imag_norm <= 1e-8:
            # print("# making trial wavefunction MO coefficient real")
            self.psi = numpy.array(self.psi.real, dtype=numpy.float64)

        self.psi0a = self.psi[:, : self.nalpha]
        self.psi0b = self.psi[:, self.nalpha :]
        self.G, self.Ghalf = gab_spin(self.psi, self.psi, self.nalpha, self.nbeta)
        self.handler = handler

        self.psi0a = numpy.ascontiguousarray(self.psi0a)
        self.psi0b = numpy.ascontiguousarray(self.psi0b)

    def build(self) -> None:
        pass

    @property
    def num_dets(self) -> int:
        return 1

    @num_dets.setter
    def num_dets(self, ndets: int) -> None:
        raise RuntimeError("Cannot modify number of determinants in SingleDet trial.")

    def calculate_energy(self, system, hamiltonian) -> numpy.ndarray:
        if self.verbose:
            print("# Computing trial wavefunction energy.")
        start = time.time()
        self.e1b = (
            numpy.sum(self.Ghalf[0] * self._rH1a)
            + numpy.sum(self.Ghalf[1] * self._rH1b)
            + hamiltonian.ecore
        )
        self.ej, self.ek = half_rotated_cholesky_jk(
            system, self.Ghalf[0], self.Ghalf[1], trial=self
        )
        self.e2b = self.ej + self.ek
        self.energy = self.e1b + self.e2b

        if self.verbose:
            print(
                "# (E, E1B, E2B): (%13.8e, %13.8e, %13.8e)"
                % (self.energy.real, self.e1b.real, self.e2b.real)
            )
            print(f"# Time to evaluate local energy: {time.time() - start} s")

    @plum.dispatch
    def half_rotate(
        self: "SingleDet",
        hamiltonian: GenericRealChol,
        comm: Optional[CommType] = MPI.COMM_WORLD,
    ):
        num_dets = 1
        orbsa = self.psi0a.reshape((num_dets, self.nbasis, self.nalpha))
        orbsb = self.psi0b.reshape((num_dets, self.nbasis, self.nbeta))
        rot_1body, rot_chol = half_rotate_generic(
            self,
            hamiltonian,
            comm,
            orbsa,
            orbsb,
            ndets=num_dets,
            verbose=self.verbose,
        )
        # Single determinant functions do not expect determinant index, so just
        # grab zeroth element.
        self._rH1a = rot_1body[0][0]
        self._rH1b = rot_1body[1][0]
        self._rchola = rot_chol[0][0]
        self._rcholb = rot_chol[1][0]
        self.half_rotated = True

    @plum.dispatch
    def half_rotate(
        self: "SingleDet",
        hamiltonian: GenericRealTHC,
        comm: Optional[CommType] = MPI.COMM_WORLD,
    ):
        # THC: half-rotate the collocation (Xocc = psi_occ^T X) and the one-body
        # integrals; no Cholesky factor is formed.
        h1 = numpy.asarray(hamiltonian.H1[0])
        self._rH1a = self.psi0a.T @ h1                 # (nalpha x nbasis)
        self._rH1b = self.psi0b.T @ h1                 # (nbeta  x nbasis)
        self._thc_Xocca = hamiltonian.half_rotate(self.psi0a)   # (nalpha x Nmu)
        self._thc_Xoccb = hamiltonian.half_rotate(self.psi0b)   # (nbeta  x Nmu)
        self._rchola = None
        self._rcholb = None
        self.half_rotated = True

    @plum.dispatch
    def half_rotate(
        self: "SingleDet",
        hamiltonian: GenericComplexTHC,
        comm: Optional[CommType] = MPI.COMM_WORLD,
    ):
        # COMPLEX THC: with complex X the BRA and KET density moments are distinct
        # objects, so BOTH half-rotated collocations are stored:
        #     Xocc  = psi0^H X        (bra)
        #     Xoccc = psi0^H conj(X)  (ket; == conj(Xocc) only for a REAL trial)
        # CONJUGATE transpose throughout: ipie builds G = conj(psi0) @ Ghalf, so
        # every half-rotation of a complex trial carries psi0^H (the real THC
        # overload's psi0^T is valid only because its trial is real).
        h1 = numpy.asarray(hamiltonian.H1[0])
        p0a = self.psi0a.conj()
        p0b = self.psi0b.conj()
        self._rH1a = p0a.T @ h1                        # (nalpha x nbasis)
        self._rH1b = p0b.T @ h1                        # (nbeta  x nbasis)
        self._thc_Xocca = hamiltonian.half_rotate(self.psi0a)   # (nalpha x Nmu)
        self._thc_Xoccb = hamiltonian.half_rotate(self.psi0b)   # (nbeta  x Nmu)
        Xc = hamiltonian.X.conj()
        self._thc_Xoccac = p0a.T @ Xc
        self._thc_Xoccbc = p0b.T @ Xc
        self._rchola = None
        self._rcholb = None
        self.half_rotated = True

    @plum.dispatch
    def half_rotate(
        self: "SingleDet",
        hamiltonian: GenericRealTHCUhf,
        comm: Optional[CommType] = MPI.COMM_WORLD,
    ):
        # THC with PER-SPIN bases (shared Nmu): each spin half-rotates its own
        # one-body block and its own collocation.  More specific than the
        # GenericRealTHC overload above, so plum routes the UHF subclass here.
        self._rH1a = self.psi0a.T @ numpy.asarray(hamiltonian.H1[0])   # (nalpha x n_a_orb)
        self._rH1b = self.psi0b.T @ numpy.asarray(hamiltonian.H1[1])   # (nbeta  x n_b_orb)
        self._thc_Xocca = hamiltonian.half_rotate_spin(self.psi0a, 0)  # (nalpha x Nmu)
        self._thc_Xoccb = hamiltonian.half_rotate_spin(self.psi0b, 1)  # (nbeta  x Nmu)
        self._rchola = None
        self._rcholb = None
        self.half_rotated = True

    @plum.dispatch
    def half_rotate(
        self: "SingleDet",
        hamiltonian: GenericRealCholChunked,
        comm: Optional[CommType] = MPI.COMM_WORLD,
    ):
        num_dets = 1
        orbsa = self.psi0a.reshape((num_dets, self.nbasis, self.nalpha))
        orbsb = self.psi0b.reshape((num_dets, self.nbasis, self.nbeta))
        rot_1body, rot_chol = half_rotate_chunked(
            self,
            hamiltonian,
            comm,
            orbsa,
            orbsb,
            ndets=num_dets,
            verbose=self.verbose,
        )
        # Single determinant functions do not expect determinant index, so just
        # grab zeroth element.
        self._rH1a = rot_1body[0][0]
        self._rH1b = rot_1body[1][0]
        self._rchola_chunk = rot_chol[0][0]
        self._rcholb_chunk = rot_chol[1][0]
        self.half_rotated = True

        # rot_1body_1 = numpy.load('../Test_Disk_nochunk/rot_1body.npy')
        # rot_chol_1 = numpy.load('../Test_Disk_nochunk/rot_chol.npy')

        # print('compare', [numpy.allclose(rot_1body, rot_1body_1), numpy.allclose(rot_chol, rot_chol_1)])

    @plum.dispatch
    def half_rotate(
        self: "SingleDet",
        hamiltonian: GenericComplexChol,
        comm: Optional[CommType] = MPI.COMM_WORLD,
    ):
        num_dets = 1
        orbsa = self.psi0a.reshape((num_dets, self.nbasis, self.nalpha))
        orbsb = self.psi0b.reshape((num_dets, self.nbasis, self.nbeta))
        rot_1body, rot_chol = half_rotate_generic(
            self,
            hamiltonian,
            comm,
            orbsa,
            orbsb,
            ndets=num_dets,
            verbose=self.verbose,
        )
        # Single determinant functions do not expect determinant index, so just
        # grab zeroth element.
        self._rH1a = rot_1body[0][0]
        self._rH1b = rot_1body[1][0]
        self._rchola = rot_chol[0][0][0]
        self._rcholb = rot_chol[1][0][0]
        self._rcholbara = rot_chol[0][1][0]
        self._rcholbarb = rot_chol[1][1][0]
        self._rAa = rot_chol[0][2][0]
        self._rAb = rot_chol[1][2][0]
        self._rBa = rot_chol[0][3][0]
        self._rBb = rot_chol[1][3][0]
        self.half_rotated = True

    def calc_overlap(self, walkers) -> numpy.ndarray:
        return calc_overlap_single_det_uhf(walkers, self)

    def calc_greens_function(self, walkers, build_full: bool = False) -> numpy.ndarray:
        if config.get_option("use_gpu"):
            return greens_function_single_det_batch(walkers, self, build_full=build_full)
        else:
            return greens_function_single_det(walkers, self, build_full=build_full)

    @plum.dispatch
    def calc_force_bias(
        self,
        hamiltonian: Union[GenericRealChol, GenericRealCholChunked],
        walkers: UHFWalkers,
        mpi_handler: MPIHandler,
    ) -> xp.ndarray:
        if hamiltonian.chunked:
            return construct_force_bias_batch_single_det_chunked(
                hamiltonian, walkers, self, mpi_handler
            )
        else:
            return construct_force_bias_batch_single_det(hamiltonian, walkers, self)

    @plum.dispatch
    def calc_force_bias(
        self,
        hamiltonian: GenericRealTHC,
        walkers: UHFWalkers,
        mpi_handler: MPIHandler,
    ) -> xp.ndarray:
        return construct_force_bias_batch_single_det(hamiltonian, walkers, self)

    @plum.dispatch
    def calc_force_bias(
        self,
        hamiltonian: GenericComplexTHC,
        walkers: UHFWalkers,
        mpi_handler: MPIHandler,
    ) -> xp.ndarray:
        # Same entry point as the real THC path; _force_bias_thc routes the
        # complex Hamiltonian to construct_force_bias_thc_cx internally.
        return construct_force_bias_batch_single_det(hamiltonian, walkers, self)

    @plum.dispatch
    def calc_force_bias(
        self,
        hamiltonian: GenericRealTHCUhf,
        walkers: UHFWalkers,
        mpi_handler: MPIHandler,
    ) -> xp.ndarray:
        # Per-spin bases: alpha term contracts Xocca/Ghalfa through X_a, beta
        # term Xoccb/Ghalfb through X_b, summed over the SHARED mu index (no
        # factor 2; no walkers.rhf shortcut -- never valid here).
        from ipie.lno_thc_uhf import construct_force_bias_thc_uhf

        vb = construct_force_bias_thc_uhf(
            hamiltonian,
            self._thc_Xocca,
            self._thc_Xoccb,
            xp.asarray(walkers.Ghalfa),
            xp.asarray(walkers.Ghalfb),
        )
        return xp.ascontiguousarray(vb.T)          # (nwalkers, nfields)

    @plum.dispatch
    def calc_force_bias(
        self,
        hamiltonian: GenericComplexChol,
        walkers: UHFWalkers,
        mpi_handler: MPIHandler,
    ) -> numpy.ndarray:
        # return construct_force_bias_batch_single_det(hamiltonian, walkers, self)
        Ghalfa = walkers.Ghalfa.reshape(walkers.nwalkers, walkers.nup * hamiltonian.nbasis)
        Ghalfb = walkers.Ghalfb.reshape(walkers.nwalkers, walkers.ndown * hamiltonian.nbasis)
        vbias = xp.zeros((hamiltonian.nfields, walkers.nwalkers), dtype=Ghalfa.dtype)
        vbias[: hamiltonian.nchol, :] = self._rAa.dot(Ghalfa.T) + self._rAb.dot(Ghalfb.T)
        vbias[hamiltonian.nchol :, :] = -self._rBa.dot(Ghalfa.T) - self._rBb.dot(Ghalfb.T)
        vbias = vbias.T.copy()
        return vbias
