# Copyright 2022 The ipie Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Correlated walker container for paired system propagation."""

import numpy

from ipie.utils.backend import arraylib as xp


class CorrelatedWalkers:
    """Treat (walker_A, walker_B) as one paired walker.

    The channel A/B walkers carry the independent orbital state for the two
    systems. The correlated container carries the single weight, overlap, phase
    and hybrid energy that are used for estimators and population control.
    """

    def __init__(self, walkers_a, walkers_b):
        if walkers_a.nwalkers != walkers_b.nwalkers:
            raise ValueError("walkers_a and walkers_b must have the same number of walkers.")

        self.walkers_A = walkers_a
        self.walkers_B = walkers_b
        self.nwalkers = walkers_a.nwalkers
        self.mpi_handler = walkers_a.mpi_handler

        self.weight = xp.ones(self.nwalkers, dtype=xp.float64)
        self.unscaled_weight = self.weight.copy()
        self.ovlp = xp.ones(self.nwalkers, dtype=xp.complex128)
        self.hybrid_energy = xp.zeros(self.nwalkers, dtype=xp.complex128)
        self.phase = xp.ones(self.nwalkers, dtype=xp.complex128)
        self.ovlp_A = walkers_a.ovlp.copy()
        self.ovlp_B = walkers_b.ovlp.copy()

        self.write_restart = bool(getattr(walkers_a, "write_restart", False))
        self.write_freq = getattr(walkers_a, "write_freq", None)
        self.write_time = getattr(walkers_a, "write_time", None)

        # Buffer stores walker A, walker B, then the paired walker state
        # (weight, unscaled_weight, ovlp, phase, hybrid_energy).
        self.buff_size = (
            int(getattr(walkers_a, "buff_size", 0))
            + int(getattr(walkers_b, "buff_size", 0))
            + 5
        )
        self.walker_buffer = numpy.zeros(self.buff_size, dtype=numpy.complex128)
        self.buff_names = []

        self.sync_combined_state()

    @staticmethod
    def _pack_single_walker(walker_set, iw):
        s = 0
        buff = xp.zeros(walker_set.buff_size, dtype=numpy.complex128)
        for d in walker_set.buff_names:
            data = walker_set.__dict__[d]
            if data is None:
                continue
            assert data.size % walker_set.nwalkers == 0
            if isinstance(data[iw], xp.ndarray):
                buff[s : s + data[iw].size] = xp.array(data[iw].ravel())
                s += data[iw].size
            elif isinstance(data[iw], list):
                for l in data[iw]:
                    if isinstance(l, xp.ndarray):
                        buff[s : s + l.size] = xp.array(l.ravel())
                        s += l.size
                    elif isinstance(l, (int, float, complex, numpy.float64, numpy.complex128)):
                        buff[s : s + 1] = l
                        s += 1
            else:
                buff[s : s + 1] = xp.array(data[iw])
                s += 1
        return buff

    @staticmethod
    def _unpack_single_walker(walker_set, iw, buff):
        s = 0
        for d in walker_set.buff_names:
            data = walker_set.__dict__[d]
            if data is None:
                continue
            assert data.size % walker_set.nwalkers == 0
            if isinstance(data[iw], xp.ndarray):
                if walker_set.__dict__[d][iw].dtype == numpy.float64:
                    walker_set.__dict__[d][iw] = xp.array(
                        buff[s : s + data[iw].size].reshape(data[iw].shape).real.copy()
                    )
                elif walker_set.__dict__[d][iw].dtype == numpy.complex128:
                    walker_set.__dict__[d][iw] = xp.array(
                        buff[s : s + data[iw].size].reshape(data[iw].shape).copy()
                    )
                s += data[iw].size
            elif isinstance(data[iw], list):
                for ix, l in enumerate(data[iw]):
                    if isinstance(l, xp.ndarray):
                        walker_set.__dict__[d][iw][ix] = xp.array(
                            buff[s : s + l.size].reshape(l.shape).copy()
                        )
                        s += l.size
                    elif isinstance(l, (int, float, complex)):
                        walker_set.__dict__[d][iw][ix] = buff[s]
                        s += 1
            else:
                if isinstance(walker_set.__dict__[d][iw], (int, numpy.int64)):
                    walker_set.__dict__[d][iw] = int(buff[s].real)
                elif isinstance(walker_set.__dict__[d][iw], (float, numpy.float64)):
                    walker_set.__dict__[d][iw] = buff[s].real
                else:
                    walker_set.__dict__[d][iw] = buff[s]
                s += 1

    def get_buffer(self, iw):
        buff_a = self._pack_single_walker(self.walkers_A, iw)
        buff_b = self._pack_single_walker(self.walkers_B, iw)
        split_a = int(self.walkers_A.buff_size)
        split_b = int(self.walkers_B.buff_size)
        buff = xp.zeros(self.buff_size, dtype=numpy.complex128)
        buff[:split_a] = buff_a
        buff[split_a : split_a + split_b] = buff_b
        s = split_a + split_b
        buff[s + 0] = self.weight[iw]
        buff[s + 1] = self.unscaled_weight[iw]
        buff[s + 2] = self.ovlp[iw]
        buff[s + 3] = self.phase[iw]
        buff[s + 4] = self.hybrid_energy[iw]
        return buff

    def set_buffer(self, iw, buff):
        split_a = int(self.walkers_A.buff_size)
        split_b = int(self.walkers_B.buff_size)
        self._unpack_single_walker(self.walkers_A, iw, buff[:split_a])
        self._unpack_single_walker(self.walkers_B, iw, buff[split_a : split_a + split_b])
        s = split_a + split_b
        self.weight[iw] = buff[s + 0].real
        self.unscaled_weight[iw] = buff[s + 1].real
        self.ovlp[iw] = buff[s + 2]
        self.phase[iw] = buff[s + 3]
        self.hybrid_energy[iw] = buff[s + 4]
        self.sync_combined_state()

    def set_combined_weight(self, iw, value):
        self.weight[iw] = value

    def fill_combined_weight(self, value):
        self.weight.fill(value)

    def sync_combined_state(self):
        """Refresh channel overlap bookkeeping only."""
        self.ovlp_A = self.walkers_A.ovlp
        self.ovlp_B = self.walkers_B.ovlp

    def orthogonalise(self, free_projection=False):
        """Orthogonalise both walker sets and refresh overlap bookkeeping."""
        det_a = self.walkers_A.orthogonalise(free_projection=free_projection)
        det_b = self.walkers_B.orthogonalise(free_projection=free_projection)
        self.sync_combined_state()
        return det_a, det_b

    def cast_to_cupy(self, verbose=False):
        if hasattr(self.walkers_A, "cast_to_cupy"):
            self.walkers_A.cast_to_cupy(verbose=verbose)
        if hasattr(self.walkers_B, "cast_to_cupy"):
            self.walkers_B.cast_to_cupy(verbose=verbose)
        self.sync_combined_state()

    def write_walkers_batch(self, comm):
        """Write both walker sets using their native serialization paths."""
        if hasattr(self.walkers_A, "write_walkers_batch"):
            self.walkers_A.write_walkers_batch(comm)
        if hasattr(self.walkers_B, "write_walkers_batch"):
            self.walkers_B.write_walkers_batch(comm)

    @property
    def total_weight(self):
        """Return the current total weight sum across local walkers."""
        return xp.sum(self.weight)
