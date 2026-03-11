from typing import Optional

import numpy as np

from ipie.config import CommType
from ipie.estimators.greens_function_single_det import greens_function_single_det_spin_batch
from ipie.hamiltonians.generic import GenericRealChol
from ipie.propagation.force_bias import construct_force_bias_batch_cisd
from ipie.propagation.overlap import calc_overlap_cisd
from ipie.trial_wavefunction.half_rotate import half_rotate_spin
from ipie.trial_wavefunction.wavefunction_base import TrialWavefunctionBase
from ipie.utils.mpi import MPIHandler
from ipie.walkers.uhf_walkers import UHFWalkers


class CISD(TrialWavefunctionBase):
    def __init__(
        self,
        wfn,
        nelec,
        nbasis,
        c1a,
        c2aa,
        c1b=None,
        c2ab=None,
        c2bb=None,
        c2_antisymm=False,
        mo_coeffb=None,
        rdm1=None,
        verbose=False,
    ) -> None:
        super().__init__(wfn, nelec, nbasis, verbose=verbose)
        self.psi0a = self.psi[:, : self.nalpha]
        self.psi0b = self.psi[:, self.nalpha :]
        self.c1a = c1a
        self.c2aa = c2aa
        self.c1b = c1b
        self.c2ab = c2ab
        self.c2bb = c2bb
        self.c2_antisymm = c2_antisymm
        self.mo_coeffb = mo_coeffb
        self.G = rdm1
        self.ovlp_ratio_cisd = None
        self._num_dets = 1
        self._max_num_dets = 1

    def build(self) -> None:
        if self.G is None:
            self.G = self.compute_rdm1()

    def half_rotate(
        self: "CISD",
        hamiltonian: GenericRealChol,
        comm: Optional[CommType] = MPIHandler().scomm,
    ):
        num_dets = 1
        orbsa = self.psi0a.reshape((num_dets, self.nbasis, self.nalpha))
        orbsb = self.psi0b.reshape((num_dets, self.nbasis, self.nbeta))
        rot_1body, rot_chol = half_rotate_spin(
            self,
            hamiltonian,
            comm,
            orbsa,
            orbsb,
            ndets=num_dets,
            verbose=self.verbose,
        )
        self._rH1a = rot_1body[0][0]
        self._rH1b = rot_1body[1][0]
        self._rchola = rot_chol[0][0]
        self._rcholb = rot_chol[1][0]
        self.rh1 = self._rH1a
        self.rh1a = self._rH1a
        self.rh1b = self._rH1b
        self.rchola = self._rchola.reshape(hamiltonian.nchol, self.nalpha, hamiltonian.nbasis).transpose(1, 2, 0)
        self.rcholb = self._rcholb.reshape(hamiltonian.nchol, self.nbeta, hamiltonian.nbasis).transpose(1, 2, 0)
        self.rchol = self.rchola
        self.half_rotated = True

    def calc_force_bias(
        self,
        hamiltonian: GenericRealChol,
        walkers: UHFWalkers,
        mpi_handler: MPIHandler,
    ) -> np.ndarray:
        return construct_force_bias_batch_cisd(hamiltonian, walkers, self)

    def compute_rdm1(self):
        raise NotImplementedError("please provide rdm1 for CISD")

    def calc_overlap(self, walkers) -> np.ndarray:
        if not hasattr(walkers, "ghalfa"):
            return self.calc_greens_function(walkers)
        return calc_overlap_cisd(walkers, self)

    def calc_greens_function(self, walkers, build_full: bool = False) -> np.ndarray:
        greens_function_single_det_spin_batch(walkers, self, build_full=build_full)
        walkers.ghalfa = walkers.Ghalfa
        walkers.ghalfb = walkers.Ghalfb
        return calc_overlap_cisd(walkers, self)