import math
import sys
import time

import numpy
import scipy.linalg
import scipy.sparse

import ipie.utils
from ipie.utils.io import write_qmcpack_sparse
from ipie.legacy.systems.ueg import UEG as UEGsys
from ipie.legacy.hamiltonians.ueg import UEG

class trscorr_UEG(UEG, object):
    def __init__(self, system, options, verbose=False):
        UEG.__init__(self, system, options, verbose=False)
        if options.get('jastrow') == 'ueg':
            self.jastrow = 'ueg'
            self.ne = system.nup + system.ndown
            self.kc_scaled = numpy.sqrt(2 * system.ecut) * self.kfac
        skip_cholesky = options.get("skip_cholesky", False)
        if skip_cholesky == False:
            if verbose:
                print("# Constructing two-body potentials incore.")

            (self.chol_vecs, self.iA, self.iB) = self.two_body_potentials_incore()
            write_ints = options.get("write_integrals", False)

            if write_ints:
                self.write_integrals(system)

        if verbose:
            print(
                "# Approximate memory required for "
                "two-body potentials: {:13.8e} GB.".format((3 * self.iA.nnz * 16 / (1024**3)))
            )
            print("# Finished constructing two-body potentials.")
            print("# Finished setting up UEG system object.")
        
    
    def uq(self, q):
        '''
        The two body part of the geminal, note that the q here is the scaled q, qscaled = q * kfac
        '''
        if self.jastrow == 'ueg':
            qnorm = numpy.linalg.norm(q, 2)
            if qnorm > self.kc_scaled:
                return - 4 * numpy.pi / qnorm**4 
            else:
                return 0.
        else: 
            raise NotImplementedError

    def vprime_effq(self, q):
        '''
        The effective potential used in the calculation of scaled rho operator, see TC-AFQMC notes Eq.(47). Here our v'_eff = v_eff - |q|^2u(q), note that the q here is the scaled q, qscaled = q * kfac
        '''
        vq = 4 * numpy.pi / numpy.dot(q, q)
        sum_over_qprime = 0
        #TODO: determine the range of q'
        for fqp in self.qvecs:
            qp = fqp * self.kfac
            sum_over_qprime += 1 / self.vol * numpy.dot((q - qp), qp) * self.uq(q - qp) * self.uq(qp)
        return vq - (self.ne - 2)/self.vol * numpy.dot(q,q) * self.uq(q) + sum_over_qprime

    def veffq(self, q):
        '''
        The effective potential defined in Eq.(19) of TC-AFQMC notes. The q here is the scaled q, qscaled = q * kfac
        '''
        return self.vprime_effq(q) + numpy.dot(q,q) * self.uq(q)
        
    def scaled_kappa_operator_incore(self, transpose):
        """scaled kappa operator as defined in Eq.(51~54) of TC-AFQMC notes
        Parameters
        ----------
        transpose: bool
            whether to transpose the resulting operator
        Returns
        -------
        kappa_q: float
            density operator
        """

        nq = len(self.qvecs)
        col_index = []
        row_index = []
        values = []

        if transpose:
            for iq in range(nq):
                qscaled = self.kfac * self.qvecs[iq]
                prefac = numpy.sqrt(-self.uq(qscaled)/ (4 * self.vol))

                for innz, kpq in enumerate(self.rho_ikpq_kpq[iq]):
                    row_index += [self.rho_ikpq_kpq[iq][innz] + self.rho_ikpq_i[iq][innz] * self.nbasis]
                    col_index += [iq]
                    k = self.basis[self.rho_ikpq_i[iq][innz]] 
                    values += [prefac * numpy.dot(k * self.kfac, qscaled)]
        else:
            for iq in range(nq):
                qscaled = self.kfac * self.qvecs[iq]
                prefac = numpy.sqrt(-self.uq(qscaled)/ (4 * self.vol))

                for innz, kpq in enumerate(self.rho_ikpq_kpq[iq]):
                    row_index += [self.rho_ikpq_kpq[iq][innz] * self.nbasis + self.rho_ikpq_i[iq][innz]]
                    col_index += [iq]
                    k = self.basis[self.rho_ikpq_i[iq][innz]]
                    values += [prefac * numpy.dot(k * self.kfac, qscaled)]

        kappa_q = scipy.sparse.csc_matrix(
            (values, (row_index, col_index)),
            shape=(self.nbasis * self.nbasis, nq),
            dtype=numpy.complex128,
        )

        return kappa_q

    def scaled_density_operator_0_incore(self, transpose):
        '''
        Scaled density operator as defined in Eq.(47) in TC-AFQMC notes.
        '''
        nq = len(self.qvecs)
        col_index = []
        row_index = []
        values = []
        #print(self.kc_scaled)

        if transpose:
            for iq in range(nq):
                qscaled = self.kfac * self.qvecs[iq]
                prefac = numpy.sqrt(self.vprime_effq(qscaled)/ (4 * self.vol))

                for innz, kpq in enumerate(self.rho_ikpq_kpq[iq]):
                    row_index += [self.rho_ikpq_kpq[iq][innz] + self.rho_ikpq_i[iq][innz] * self.nbasis]
                    col_index += [iq]
                    k = self.basis[self.rho_ikpq_i[iq][innz]] 
                    values += [prefac]
        else:
            for iq in range(nq):
                qscaled = self.kfac * self.qvecs[iq]
                prefac = numpy.sqrt(self.vprime_effq(qscaled)/ (4 * self.vol))

                for innz, kpq in enumerate(self.rho_ikpq_kpq[iq]):
                    row_index += [self.rho_ikpq_kpq[iq][innz] * self.nbasis + self.rho_ikpq_i[iq][innz]]
                    col_index += [iq]
                    k = self.basis[self.rho_ikpq_i[iq][innz]]
                    values += [prefac]

        rho_q = scipy.sparse.csc_matrix(
            (values, (row_index, col_index)),
            shape=(self.nbasis * self.nbasis, nq),
            dtype=numpy.complex128,
        )

        return rho_q

    def scaled_density_operator_1_incore(self, transpose):
        '''
        Scaled density operator as defined in Eq.(51~54) in TC-AFQMC notes.
        '''
        nq = len(self.qvecs)
        col_index = []
        row_index = []
        values = []

        if transpose:
            for iq in range(nq):
                qscaled = self.kfac * self.qvecs[iq]
                #print("qscaled", qscaled)
                prefac = numpy.sqrt(-self.uq(qscaled)/ (4 * self.vol))

                for innz, kpq in enumerate(self.rho_ikpq_kpq[iq]):
                    row_index += [self.rho_ikpq_kpq[iq][innz] + self.rho_ikpq_i[iq][innz] * self.nbasis]
                    col_index += [iq]
                    k = self.basis[self.rho_ikpq_i[iq][innz]] 
                    values += [prefac]
        else:
            for iq in range(nq):
                qscaled = self.kfac * self.qvecs[iq]
                prefac = numpy.sqrt(-self.uq(qscaled)/ (4 * self.vol))

                for innz, kpq in enumerate(self.rho_ikpq_kpq[iq]):
                    row_index += [self.rho_ikpq_kpq[iq][innz] * self.nbasis + self.rho_ikpq_i[iq][innz]]
                    col_index += [iq]
                    k = self.basis[self.rho_ikpq_i[iq][innz]]
                    values += [prefac]

        rho_q = scipy.sparse.csc_matrix(
            (values, (row_index, col_index)),
            shape=(self.nbasis * self.nbasis, nq),
            dtype=numpy.complex128,
        )

        return rho_q

    def two_body_potentials_incore(self):
        """Calculate As and Bs of Eq.(47-48) and Eq.(51-54) of TC-AFQMC notes for a given plane-wave vector q

        Parameters
        ----------
        system :
            system class
        q : float
            a plane-wave vector
        Returns
        -------
        iA : numpy array
            Eq.(13a)
        iB : numpy array
            Eq.(13b)
        """
        # qscaled = self.kfac * self.qvecs

        # # Due to the HS transformation, we have to do pi / 2*vol as opposed to 2*pi / vol
        rho0_q = self.scaled_density_operator_0_incore(transpose=False)
        rho0_qH = self.scaled_density_operator_0_incore(transpose=True)
        kappa_q = self.scaled_kappa_operator_incore(transpose=False)
        kappa_qH = self.scaled_kappa_operator_incore(transpose=True)
        rho1_q = self.scaled_density_operator_1_incore(transpose=False)
        rho1_qH = self.scaled_density_operator_1_incore(transpose=True)    
        A0 = (rho0_q + rho0_qH)
        B0 = 1j * (rho0_q - rho0_qH)
        A1 = 1j * (kappa_q + rho1_qH)
        B1 = - (kappa_q - rho1_qH)
        A2 = - (kappa_qH + rho1_q)
        B2 = 1j *(kappa_qH - rho1_q)

        assert (A0.shape == A1.shape and A0.shape == A2.shape, "The shapes of the A arrays are not compatible")
        assert (B0.shape == B1.shape and B0.shape == B2.shape, "The shapes of the B arrays are not compatible")
        A = scipy.sparse.hstack([A0, A1, A2])
        B = scipy.sparse.hstack([B0, B1, B2])
        print('shape of the A matrix: ', A.shape)
        return (A, B)
    
    

def test_trscorr_ueg():
    options = {"nup": 2, "ndown": 2, "rs": 1.0, "thermal": False, "ecut": 3, 'jastrow': 'ueg'}
    system = UEGsys(options, True)
    ueg = UEG(system, options, True)
    trsueg = trscorr_UEG(system, options, True)
    print(trsueg.scaled_density_operator_incore(True))
    print(trsueg.scaled_density_operator_1_incore(True))
    print(trsueg.scaled_density_operator_0_incore(True))
    return

if __name__ == '__main__':
    test_trscorr_ueg()
