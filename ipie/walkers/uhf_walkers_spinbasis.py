# UHF walkers with PER-SPIN orbital bases (n_a_orb != n_b_orb allowed).
# Companion to SingleDetUhfBasis / GenericRealTHCUhf: walkers.phia lives in the
# alpha cluster basis (n_a_orb x nalpha), walkers.phib in the beta cluster basis
# (n_b_orb x nbeta).
#
# All inherited UHFWalkers machinery (reortho per-spin QR, buff_names-based pop
# control pack/unpack, greens-function/overlap consumers) operates on phia and
# phib independently and is shape-agnostic; only __init__ needs replacing (the
# parent slices ONE initial_walker array and allocates Ghalf with one shared
# nbasis).

import numpy

from ipie.utils.backend import arraylib as xp
from ipie.walkers.base_walkers import BaseWalkers
from ipie.walkers.uhf_walkers import UHFWalkers


class UHFWalkersSpinBasis(UHFWalkers):
    """UHF walkers whose alpha/beta determinants live in different bases.

    Parameters
    ----------
    initial_walker_a : ndarray (n_a_orb, nup)
    initial_walker_b : ndarray (n_b_orb, ndown)
    """

    # NOTE: deliberately does NOT call UHFWalkers.__init__ (it slices a single
    # stacked initial_walker and allocates all Green's functions with one
    # shared nbasis); replicates it per spin instead.
    def __init__(
        self,
        initial_walker_a: numpy.ndarray,
        initial_walker_b: numpy.ndarray,
        nup: int,
        ndown: int,
        nbasis_a: int,
        nbasis_b: int,
        nwalkers: int,
        mpi_handler,
        verbose: bool = False,
    ):
        assert initial_walker_a.shape == (nbasis_a, nup)
        assert initial_walker_b.shape == (nbasis_b, ndown)
        self.nup = nup
        self.ndown = ndown
        self.nbasis_a = nbasis_a
        self.nbasis_b = nbasis_b
        # Generic bookkeeping only; per-spin consumers use array shapes directly.
        self.nbasis = max(nbasis_a, nbasis_b)
        self.mpi_handler = mpi_handler

        BaseWalkers.__init__(self, nwalkers, verbose=verbose)

        self.field_configs = None

        self.phia = xp.array(
            [initial_walker_a.copy() for _ in range(self.nwalkers)],
            dtype=xp.complex128,
        )
        self.phib = xp.array(
            [initial_walker_b.copy() for _ in range(self.nwalkers)],
            dtype=xp.complex128,
        )

        # Built only on request (greens_function_single_det writes in place).
        self.Ga = numpy.zeros(
            shape=(self.nwalkers, self.nbasis_a, self.nbasis_a), dtype=numpy.complex128
        )
        self.Gb = numpy.zeros(
            shape=(self.nwalkers, self.nbasis_b, self.nbasis_b), dtype=numpy.complex128
        )
        self.Ghalfa = numpy.zeros(
            shape=(self.nwalkers, self.nup, self.nbasis_a), dtype=numpy.complex128
        )
        self.Ghalfb = numpy.zeros(
            shape=(self.nwalkers, self.ndown, self.nbasis_b), dtype=numpy.complex128
        )

        self.buff_names += ["phia", "phib"]
        self.buff_size = round(self.set_buff_size_single_walker() / float(self.nwalkers))
        self.walker_buffer = numpy.zeros(self.buff_size, dtype=numpy.complex128)

        self.rhf = False   # the closed-shell fast path never applies here
