import math
import time

import numpy

from ipie.utils.pack_numba import unpack_VHS_batch

try:
    from ipie.utils.pack_numba_gpu import unpack_VHS_batch_gpu
except:
    pass

import plum

from ipie.config import config
from ipie.hamiltonians.kpt_hamiltonian import KptComplexChol
from ipie.hamiltonians.generic_base import GenericBase
from ipie.propagation.operations import apply_exponential, apply_exponential_batch
from ipie.propagation.phaseless_kpt_base import PhaselessKptBase
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import synchronize
from ipie.walkers.uhf_walkers import UHFWalkers


class PhaselessKpt(PhaselessKptBase):
    """A class for performing phaseless propagation with k-point Hamiltonian."""

    def __init__(self, time_step, exp_nmax=6, verbose=False):
        super().__init__(time_step, verbose=verbose)
        self.exp_nmax = exp_nmax

    @plum.dispatch
    def apply_VHS(self, walkers: UHFWalkers, hamiltonian: GenericBase, xshifted: xp.ndarray):
        start_time = time.time()
        VHS = self.construct_VHS(hamiltonian, xshifted)
        synchronize()
        self.timer.tvhs += time.time() - start_time
        assert len(VHS.shape) == 3  # shape = nwalkers, nk * nbasis, nk * nbasis

        start_time = time.time()
        if config.get_option("use_gpu"):
            # walkers.phia = apply_exponential_batch(walkers.phia, VHS, self.exp_nmax)
            # if walkers.ndown > 0 and not walkers.rhf:
            #     walkers.phib = apply_exponential_batch(walkers.phib, VHS, self.exp_nmax)
            raise NotImplementedError
        else:
            for iw in range(walkers.nwalkers):
                # 2.b Apply two-body
                walkers.phia[iw] = apply_exponential(walkers.phia[iw], VHS[iw], self.exp_nmax)
                if walkers.ndown > 0 and not walkers.rhf:
                    walkers.phib[iw] = apply_exponential(walkers.phib[iw], VHS[iw], self.exp_nmax)
        synchronize()
        self.timer.tgemm += time.time() - start_time

    @plum.dispatch.abstract
    def construct_VHS(self, hamiltonian: GenericBase, xshifted: xp.ndarray) -> xp.ndarray:
        print("JOONHO here abstract function for construct VHS")
        "abstract function for construct VHS"

    # Any class inherited from PhaselessGeneric should override this method.
    @plum.dispatch
    def construct_VHS(self, hamiltonian: KptComplexChol, xshifted: xp.ndarray) -> xp.ndarray:
        """
        Construct the VHS matrix for phaseless propagation.
        
        xshifted: [2, nwalkers, naux, nk]
        """
        nwalkers = xshifted.shape[1]
        VHS = numpy.zeros((nwalkers, hamiltonian.nk, hamiltonian.nbasis, hamiltonian.nk, hamiltonian.nbasis), dtype=numpy.complex128)
        for iq in range(hamiltonian.nk):
            for ik in range(hamiltonian.nk):
                ikpq = hamiltonian.ikpq_mat[ik, iq]
                xtildepiq = xshifted[0, :, :, iq]
                xtildemiq = xshifted[1, :, :, iq]
                xvhsiq = (1j * xtildepiq + xtildemiq) / 2
                VHS[:, ik, :, ikpq, :] = self.sqrt_dt * numpy.einsum('wx, xpr -> wpr', xvhsiq, hamiltonian.chol[:, ik, :, iq, :])
        VHS = VHS.reshape(nwalkers, hamiltonian.nk * hamiltonian.nbasis, hamiltonian.nk * hamiltonian.nbasis)
        if config.get_option("use_gpu"):
            raise NotImplementedError
        return VHS

Phaseless = {"generic": PhaselessKpt}
