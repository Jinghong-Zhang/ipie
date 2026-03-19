# Copyright 2022 The ipie Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Correlated walker container for paired system propagation."""

import numpy

from ipie.utils.backend import arraylib as xp


class CorrelatedWalkers:
    """Wrapper around two walker collections with aggregate state.

    The A and B walker sets are propagated independently, while this wrapper
    maintains product quantities used by correlated analysis:
    - total weight: w_tot = w_a * w_b
    - total overlap: ovlp_tot = ovlp_a * ovlp_b
    """

    def __init__(self, walkers_a, walkers_b):
        if walkers_a.nwalkers != walkers_b.nwalkers:
            raise ValueError("walkers_a and walkers_b must have the same number of nwalkers.")

        self.walkers_A = walkers_a
        self.walkers_B = walkers_b
        self.nwalkers = walkers_a.nwalkers
        self.mpi_handler = walkers_a.mpi_handler

        self.weight = None
        self.unscaled_weight = None
        self.ovlp = None
        self.hybrid_energy = None
        self.phase = None
        self.weight_A = None
        self.weight_B = None
        self.ovlp_A = None
        self.ovlp_B = None

        self.write_restart = bool(getattr(walkers_a, "write_restart", False))
        self.write_freq = getattr(walkers_a, "write_freq", None)
        self.write_time = getattr(walkers_a, "write_time", None)

        self.buff_size = int(getattr(walkers_a, "buff_size", 0)) + int(
            getattr(walkers_b, "buff_size", 0)
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
        return xp.concatenate((buff_a, buff_b))

    def set_buffer(self, iw, buff):
        split = int(self.walkers_A.buff_size)
        self._unpack_single_walker(self.walkers_A, iw, buff[:split])
        self._unpack_single_walker(self.walkers_B, iw, buff[split:])
        self.sync_combined_state()

    def set_combined_weight(self, iw, value):
        current = self.walkers_A.weight[iw] * self.walkers_B.weight[iw]
        if abs(current) > 1e-14:
            self.walkers_A.weight[iw] *= value / current
        else:
            self.walkers_A.weight[iw] = value
            self.walkers_B.weight[iw] = 1.0
        self.sync_combined_state()

    def fill_combined_weight(self, value):
        self.walkers_A.weight.fill(value)
        self.walkers_B.weight.fill(1.0)
        self.sync_combined_state()

    def sync_combined_state(self):
        """Refresh aggregate arrays from the two wrapped walkers."""
        self.weight_A = self.walkers_A.weight
        self.weight_B = self.walkers_B.weight
        self.ovlp_A = self.walkers_A.ovlp
        self.ovlp_B = self.walkers_B.ovlp
        self.weight = self.walkers_A.weight * self.walkers_B.weight
        self.unscaled_weight = self.walkers_A.unscaled_weight * self.walkers_B.unscaled_weight
        self.ovlp = self.walkers_A.ovlp * self.walkers_B.ovlp
        self.hybrid_energy = self.walkers_A.hybrid_energy + self.walkers_B.hybrid_energy
        self.phase = self.walkers_A.phase * self.walkers_B.phase

    def orthogonalise(self, free_projection=False):
        """Orthogonalise both walker sets and refresh aggregate state."""
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

    def reset_weights(self, value=1.0):
        """Reset channel and aggregate weights to a constant."""
        self.walkers_A.weight.fill(value)
        self.walkers_B.weight.fill(value)
        self.sync_combined_state()

    @property
    def total_weight(self):
        """Return the current total weight sum across local walkers."""
        return xp.sum(self.weight)
