# Single-determinant UHF trial with PER-SPIN orbital bases (n_a_orb != n_b_orb
# allowed).  Companion to the GenericRealTHCUhf hamiltonian: the alpha and beta
# determinants live in different cluster bases that share only the auxiliary
# (ISDF) index of the two-body factorization.
#
# SingleDet assumes ONE stacked wavefunction array (nbasis, na+nb); this
# subclass takes psia (n_a_orb, na) and psib (n_b_orb, nb) separately and builds
# all per-spin quantities directly.  All inherited plum methods registered on
# SingleDet (half_rotate / calc_force_bias for GenericRealTHCUhf, greens
# function, overlap) are per-spin already and shape-agnostic, so they apply
# unchanged.

import numpy

from ipie.estimators.utils import gab_mod
from ipie.trial_wavefunction.single_det import SingleDet
from ipie.trial_wavefunction.wavefunction_base import TrialWavefunctionBase
from ipie.utils.mpi import MPIHandler


class SingleDetUhfBasis(SingleDet):
    """Single-determinant trial with different alpha/beta orbital bases.

    Parameters
    ----------
    psia : ndarray (n_a_orb, nalpha)
        Alpha determinant in the alpha cluster basis.
    psib : ndarray (n_b_orb, nbeta)
        Beta determinant in the beta cluster basis.
    num_elec : tuple(int, int)
        (nalpha, nbeta).
    """

    # NOTE: deliberately does NOT call SingleDet.__init__ (it slices a single
    # stacked (nbasis, na+nb) array, impossible for per-spin bases).
    def __init__(self, psia, psib, num_elec, handler=MPIHandler(), verbose=False):
        psia = numpy.asarray(psia)
        psib = numpy.asarray(psib)
        assert psia.ndim == 2 and psib.ndim == 2
        TrialWavefunctionBase.__init__(self, psia, num_elec, psia.shape[0], verbose=verbose)
        self.nbasis_a = psia.shape[0]
        self.nbasis_b = psib.shape[0]
        assert psia.shape[1] == self.nalpha, f"psia cols {psia.shape[1]} != nalpha {self.nalpha}"
        assert psib.shape[1] == self.nbeta, f"psib cols {psib.shape[1]} != nbeta {self.nbeta}"
        self._num_dets = 1
        self._max_num_dets = 1
        self.psi = None                    # no stacked single-basis form exists

        def _realify(psi):
            imag_norm = numpy.sum(psi.imag.ravel() * psi.imag.ravel()) if \
                numpy.iscomplexobj(psi) else 0.0
            if imag_norm <= 1e-8:
                psi = numpy.array(psi.real, dtype=numpy.float64)
            return numpy.ascontiguousarray(psi)

        self.psi0a = _realify(psia)
        self.psi0b = _realify(psib)

        # Trial density / half-rotated Green's functions, per spin (same
        # conventions as estimators.utils.gab_spin, but without the stacked
        # array).  G is a LIST: the two spin blocks have different shapes.
        Ga, Gha = gab_mod(self.psi0a, self.psi0a)
        Gb, Ghb = gab_mod(self.psi0b, self.psi0b)
        self.G = [Ga, Gb]
        self.Ghalf = [Gha, Ghb]
        self.handler = handler

    def build(self) -> None:
        pass
