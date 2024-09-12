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
from ipie.hamiltonians.kpt_hamiltonian import KptComplexChol, KptComplexCholSymm, KptISDF
from ipie.hamiltonians.generic_base import GenericBase
from ipie.propagation.operations import apply_exponential, apply_exponential_batch
from ipie.propagation.phaseless_kpt_base import PhaselessKptBase
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import synchronize
from ipie.walkers.uhf_walkers import UHFWalkers
from numba import jit

@jit(nopython=True, fastmath=True)
def construct_VHS_kernel_symm(chol, sqrt_dt, xshifted, nk, nbasis, nwalkers, ikpq_mat, Sset, Qplus):
    VHS = numpy.zeros((nwalkers, nk, nk, nbasis * nbasis), dtype=numpy.complex128)
    for iq in range(len(Sset)):
        iq_real = Sset[iq]
        for ik in range(nk):
            ikpq = ikpq_mat[iq_real, ik]
            x_iq = .5 * (1j * xshifted[0, :, :, iq] + xshifted[1, :, :, iq])
            xconj_iq = .5 * (1j * xshifted[0, :, :, iq] - xshifted[1, :, :, iq])
            cholkq = chol[:, ik, :, iq, :].copy()
            cholkqT = chol[:, ik, :, iq, :].transpose(0, 2, 1).copy()
            cholkq = cholkq.reshape(-1, nbasis*nbasis)
            cholkqT = cholkqT.reshape(-1, nbasis*nbasis)
            for iw in range(nwalkers):
                # VHS[iw, ik, ikpq] += numpy.einsum('wx, xpr -> wpr', x_iq[iw], chol[:, ik, :, iq, :])
                VHS[iw, ik, ikpq] += sqrt_dt * x_iq[iw] @ cholkq
                VHS[iw, ikpq, ik] += sqrt_dt * xconj_iq[iw] @ cholkqT.conj()

    for iq in range(len(Sset), len(Sset) + len(Qplus)):
        iq_real = Qplus[iq - len(Sset)]
        for ik in range(nk):
            ikpq = ikpq_mat[iq_real, ik]
            x_iq = .5 * (1j * xshifted[0, :, :, iq] + xshifted[1, :, :, iq])
            xconj_iq = .5 * (1j * xshifted[0, :, :, iq] - xshifted[1, :, :, iq])
            cholkq = chol[:, ik, :, iq, :].copy()
            cholkqT = chol[:, ik, :, iq, :].transpose(0, 2, 1).copy()
            cholkq = cholkq.reshape(-1, nbasis*nbasis)
            cholkqT = cholkqT.reshape(-1, nbasis*nbasis)
            for iw in range(nwalkers):
                # VHS[iw, ik, ikpq] += numpy.einsum('wx, xpr -> wpr', x_iq[iw], chol[:, ik, :, iq, :])
                VHS[iw, ik, ikpq] += sqrt_dt * math.sqrt(2) * x_iq[iw] @ cholkq
                VHS[iw, ikpq, ik] += sqrt_dt * math.sqrt(2) * xconj_iq[iw] @ cholkqT.conj()
    VHS = VHS.reshape(nwalkers, nk, nk, nbasis, nbasis).transpose(0, 1, 3, 2, 4).copy()
    VHS = VHS.reshape(nwalkers, nk * nbasis, nk * nbasis)
    return VHS


class PhaselessKptChol(PhaselessKptBase):
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
        # print(f"norm of x^+[0, gamma] = {numpy.linalg.norm(xp[:, :, 0])}")

        for iq in range(hamiltonian.nk):
            for ik in range(hamiltonian.nk):
                ikpq = hamiltonian.ikpq_mat[ik, iq]
                imq = hamiltonian.imq_vec[iq]
                xtildepiq = xshifted[0, :, :, iq] + xshifted[0, :, :, imq]
                xtildemiq = xshifted[1, :, :, iq] - xshifted[1, :, :, imq]
                xvhsiq = (1j * xtildepiq + xtildemiq) / 2
                VHS[:, ik, :, ikpq, :] = self.sqrt_dt * numpy.einsum('wx, xpr -> wpr', xvhsiq, hamiltonian.chol[:, ik, :, iq, :])
        # print(f"norm of VHS = {numpy.linalg.norm(VHS.ravel())}")
        VHS = VHS.reshape(nwalkers, hamiltonian.nk * hamiltonian.nbasis, hamiltonian.nk * hamiltonian.nbasis)
        if config.get_option("use_gpu"):
            raise NotImplementedError
        return VHS
    
    @plum.dispatch
    def construct_VHS(self, hamiltonian: KptComplexCholSymm, xshifted: xp.ndarray) -> xp.ndarray:
        """
        Construct the VHS matrix for phaseless propagation.
        
        xshifted: [2, nwalkers, naux, unique_nk]
        """
        nwalkers = xshifted.shape[1]
        VHS = construct_VHS_kernel_symm(hamiltonian.chol, self.sqrt_dt, xshifted, hamiltonian.nk, hamiltonian.nbasis, nwalkers, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus)
        # VHS = numpy.zeros((nwalkers, hamiltonian.nk, hamiltonian.nbasis, hamiltonian.nk, hamiltonian.nbasis), dtype=numpy.complex128)
        # # print(f"norm of x^+[0, gamma] = {numpy.linalg.norm(xp[:, :, 0])}")

        # for iq in range(len(hamiltonian.Sset)):
        #     iq_real = hamiltonian.Sset[iq]
        #     for ik in range(hamiltonian.nk):
        #         ikpq = hamiltonian.ikpq_mat[iq_real, ik]
        #         x_iq = .5 * (1j * xshifted[0, :, :, iq] + xshifted[1, :, :, iq])
        #         xconj_iq = .5 * (1j * xshifted[0, :, :, iq] - xshifted[1, :, :, iq])
        #         VHS[:, ik, :, ikpq, :] += self.sqrt_dt * numpy.einsum('wx, xpr -> wpr', x_iq, hamiltonian.chol[:, ik, :, iq, :])
        #         VHS[:, ikpq, :, ik, :] += self.sqrt_dt * numpy.einsum('wx, xpr -> wrp', xconj_iq, hamiltonian.chol[:, ik, :, iq, :].conj())

        # for iq in range(len(hamiltonian.Sset), len(hamiltonian.Sset) + len(hamiltonian.Qplus)):
        #     iq_real = hamiltonian.Qplus[iq - len(hamiltonian.Sset)]
        #     for ik in range(hamiltonian.nk):
        #         ikpq = hamiltonian.ikpq_mat[iq_real, ik]
        #         x_iq = .5 * (1j * xshifted[0, :, :, iq] + xshifted[1, :, :, iq])
        #         xconj_iq = .5 * (1j * xshifted[0, :, :, iq] - xshifted[1, :, :, iq])
        #         VHS[:, ik, :, ikpq, :] += self.sqrt_dt * math.sqrt(2) * numpy.einsum('wx, xpr -> wpr', x_iq, hamiltonian.chol[:, ik, :, iq, :])
        #         # VHS[:, ik, :, ikpq, :] += self.sqrt_dt * 2. * numpy.einsum('wx, xpr -> wpr', x_iq, hamiltonian.chol[:, ik, :, iq, :])
        #         VHS[:, ikpq, :, ik, :] += self.sqrt_dt * math.sqrt(2) * numpy.einsum('wx, xpr -> wrp', xconj_iq, hamiltonian.chol[:, ik, :, iq, :].conj())
        #         # VHS[:, ikpq, :, ik, :] += self.sqrt_dt * 2. * numpy.einsum('wx, xpr -> wrp', xconj_iq, hamiltonian.chol[:, ik, :, iq, :].conj())
        # # print(f"norm of VHS = {numpy.linalg.norm(VHS.ravel())}")
        # VHS = VHS.reshape(nwalkers, hamiltonian.nk * hamiltonian.nbasis, hamiltonian.nk * hamiltonian.nbasis)
        # print(f"norm of VHS = {numpy.linalg.norm(VHS.ravel())}")
        if config.get_option("use_gpu"):
            raise NotImplementedError
        return VHS

class PhaselessKptISDF(PhaselessKptBase):
    """A class for performing phaseless propagation with k-point Hamiltonian with ERI approximated by ISDF."""

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
    def construct_VHS(self, hamiltonian: KptISDF, xshifted: xp.ndarray) -> xp.ndarray:
        """
        Construct the VHS matrix for phaseless propagation.
        
        xshifted: [2, nwalkers, naux, nk]
        """
        nwalkers = xshifted.shape[1]
        Lmat = numpy.zeros((nwalkers, hamiltonian.nk, hamiltonian.nbasis, hamiltonian.nk, hamiltonian.nbasis), dtype=numpy.complex128)
        Lmatdagger

        # print(f"norm of x^+[0, gamma] = {numpy.linalg.norm(xp[:, :, 0])}")

        for iq in range(hamiltonian.nk):
            Glis = hamiltonian.q2G[iq]
            for iG in range(len(Glis)):
                try:
                    ik_lis = hamiltonian.qG2k[(iq, iG)]
                    for ik in ik_lis:
                        ikpq = hamiltonian.ikpq_mat[ik, iq]
                        Lmat[:, ik, :, ikpq, :] += self.sqrt_dt * numpy.einsum("wX, XP, Pp, Pr -> wpr", xshifted[0, :, :, iq], hamiltonian.cholM[:, iq, iG, :], hamiltonian.weights[:, :, ik].conj(), hamiltonian.weights[:, :, ikpq])
                        Lmatdagger[:, ikpq, :, ik, :] += self.sqrt_dt * numpy.einsum("wX, XP, Pp, Pr -> wpr", xshifted[1, :, :, iq], hamiltonian.cholM[:, iq, iG, :].conj(), hamiltonian.weights[:, :, ikpq].conj(), hamiltonian.weights[:, :, ik])
                except KeyError:
                    continue
        # print(f"norm of VHS = {numpy.linalg.norm(VHS.ravel())}")
        Lmat = Lmat.reshape(nwalkers, hamiltonian.nk * hamiltonian.nbasis, hamiltonian.nk * hamiltonian.nbasis)
        Lmatdagger = Lmatdagger.reshape(nwalkers, hamiltonian.nk * hamiltonian.nbasis, hamiltonian.nk * hamiltonian.nbasis)
        VHS = Lmat + Lmatdagger
        if config.get_option("use_gpu"):
            raise NotImplementedError
        return VHS

Phaseless = {"cholesky": PhaselessKptChol, "isdf": PhaselessKptISDF}
