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
from abc import ABCMeta, abstractmethod

import numpy

from ipie.estimators.estimator_base import EstimatorBase
from ipie.utils.backend import to_host


def delete_group_sums(group_numer, group_denom):
    """Numerator and denominator sums with each group left out.

    Built from prefix and suffix sums, so that a dominant group is never
    subtracted from an almost equal total, which would lose the remaining mass.

    Parameters
    ----------
    group_numer : :class:`numpy.ndarray`
        Numerator sums per group, shape (ngroups, ...).
    group_denom : :class:`numpy.ndarray`
        Denominator sums per group, shape (ngroups,).

    Returns
    -------
    leave_numer, leave_denom : :class:`numpy.ndarray`
        Sums over all groups but one, same shapes as the inputs.
    """
    zero = numpy.zeros_like(group_numer[:1])
    leave_numer = numpy.concatenate(
        (zero, numpy.cumsum(group_numer[:-1], axis=0))
    ) + numpy.concatenate((numpy.cumsum(group_numer[:0:-1], axis=0)[::-1], zero))
    leave_denom = numpy.concatenate(([0.0], numpy.cumsum(group_denom[:-1]))) + numpy.concatenate(
        (numpy.cumsum(group_denom[:0:-1])[::-1], [0.0])
    )
    return leave_numer, leave_denom


class TraceRatioObservables(metaclass=ABCMeta):
    """Base class for observables estimated as a ratio of means of trace sums.

    A subclass defines `names` and `size` of the raw vector, :meth:`measure`
    for one contraction, and :meth:`_derived` and :meth:`_as_dict`, which form
    the final quantities from the ensemble ratio of the raw vector.
    """

    names = ()
    size = 0

    @abstractmethod
    def measure(self, G):
        """Raw observable vector of one Green's function, shape (size,)."""

    @abstractmethod
    def _derived(self, mean_vector):
        """Final complex quantities from the ensemble ratio of the raw vector."""

    @abstractmethod
    def _as_dict(self, values):
        """Final quantities as a dictionary."""

    def finalise(self, mean_vector):
        """Form the nonlinear final quantities from the complex ensemble ratio.

        Parameters
        ----------
        mean_vector : :class:`numpy.ndarray`
            Ensemble ratio of the raw vector, shape (size,).

        Returns
        -------
        result : dict
            Real parts of the final quantities and the largest imaginary part
            `imaginary_max`.
        """
        derived = self._derived(mean_vector)
        result = self._as_dict(derived.real)
        result["imaginary_max"] = float(numpy.max(numpy.abs(derived.imag)))
        return result

    # Compatibility with earlier versions.
    def finalize(self, mean_vector):
        return self.finalise(mean_vector)

    def summary(self, numer_blocks, denom_blocks, log_shifts, block_size=1):
        """Ratio of the summed blocks and delete-a-group jackknife errors.

        Block b holds scaled complex sums; its physical sums are the stored
        values times exp(log_shifts[b]). Consecutive groups of `block_size`
        blocks are the jackknife units; they should be long enough to cover
        any correlation between successive blocks, and the cyclic cuts of one
        path always belong to one block. With one group the errors are NaN.

        Parameters
        ----------
        numer_blocks : :class:`numpy.ndarray`
            Numerators, shape (nblocks, size).
        denom_blocks : :class:`numpy.ndarray`
            Denominators, shape (nblocks,).
        log_shifts : :class:`numpy.ndarray`
            Log scales of the blocks, shape (nblocks,).
        block_size : int
            Blocks per jackknife group; must divide nblocks. Default 1.

        Returns
        -------
        summary : dict
            `mean` and `error` dictionaries, `nblocks`, `jackknife_groups`,
            `block_size` and `imaginary_max`.
        """
        numer = numpy.asarray(numer_blocks, dtype=numpy.complex128)
        denom = numpy.asarray(denom_blocks, dtype=numpy.complex128)
        shifts = numpy.asarray(log_shifts, dtype=float)
        if numer.ndim != 2 or numer.shape[1] != self.size or numer.shape[0] == 0:
            raise ValueError("numer_blocks must have nonempty shape (nblocks, size).")
        nblocks = len(numer)
        if denom.shape != (nblocks,) or shifts.shape != (nblocks,):
            raise ValueError("Each numerator block needs one denominator and log shift.")
        if not (
            numpy.isfinite(numer).all()
            and numpy.isfinite(denom).all()
            and numpy.isfinite(shifts).all()
        ):
            raise ValueError("Raw block sums and log shifts must be finite.")
        if int(block_size) != block_size or block_size < 1 or nblocks % int(block_size):
            raise ValueError("block_size must be positive and divide the block count.")
        block_size = int(block_size)
        scale = numpy.exp(shifts - numpy.max(shifts))
        numerator = numer * scale[:, None]
        denominator = denom * scale
        total_numer, total_denom = numerator.sum(axis=0), denominator.sum()
        if abs(total_denom) == 0:
            raise ValueError("The total trace denominator vanishes.")
        mean = self.finalise(total_numer / total_denom)
        ngroups = nblocks // block_size
        error = numpy.full(len(self._derived(total_numer / total_denom)), numpy.nan)
        if ngroups > 1:
            group_numer = numerator.reshape(ngroups, block_size, self.size).sum(axis=1)
            group_denom = denominator.reshape(ngroups, block_size).sum(axis=1)
            leave_numer, leave_denom = delete_group_sums(group_numer, group_denom)
            if numpy.any(numpy.abs(leave_denom) == 0):
                raise ValueError("A jackknife deletion has zero trace denominator.")
            leave = numpy.asarray(
                [self._derived(n / d).real for n, d in zip(leave_numer, leave_denom)]
            )
            error = numpy.sqrt(
                (ngroups - 1) / ngroups * numpy.sum((leave - leave.mean(axis=0)) ** 2, axis=0)
            )
        return {
            "mean": mean,
            "error": self._as_dict(error),
            "nblocks": nblocks,
            "jackknife_groups": ngroups,
            "block_size": block_size,
            "imaginary_max": mean["imaginary_max"],
        }


class HubbardSpinObservables(TraceRatioObservables):
    r"""Equal-time observables of a periodic rectangular Hubbard lattice.

    Energy per site, density, double occupancy, singles, site magnetizations
    and the spin correlations per distance shell,

    .. math::

        C_{zz}(d) = 4 \left(\langle S^z_i S^z_j \rangle - \langle S^z_i \rangle
                    \langle S^z_j \rangle\right), \qquad S^z = (n_\uparrow - n_\downarrow)/2,

    averaged over every directed pair at minimum-image squared distance d.
    The energy is :math:`\langle K + U \sum_i n_{i\uparrow} n_{i\downarrow}\rangle /
    N_{sites}` with the hopping K only: no chemical potential, pinning field or
    energy shift. Complex values are kept through all contractions and ratios.

    Parameters
    ----------
    hop : :class:`numpy.ndarray`
        Hopping matrix, shape (nsites, nsites) or (2, nsites, nsites).
    U : float
        Hubbard interaction.
    lx, ly : int
        Lattice dimensions; site index i = x + lx * y.
    shell_distances_squared : tuple
        Squared distances of the correlation shells. Default (1, 2, 4, 5).
    """

    def __init__(self, hop, U, lx, ly, shell_distances_squared=(1, 2, 4, 5)):
        if int(lx) != lx or int(ly) != ly or lx < 1 or ly < 1:
            raise ValueError("Lattice lengths must be positive integers.")
        self.lx, self.ly = int(lx), int(ly)
        self.nsites = self.lx * self.ly
        self.hop = numpy.asarray(to_host(hop))
        if self.hop.shape == (self.nsites, self.nsites):
            self.hop = numpy.broadcast_to(self.hop, (2, self.nsites, self.nsites))
        if self.hop.shape != (2, self.nsites, self.nsites):
            raise ValueError("hop must have shape (M,M) or (2,M,M).")
        self.U = float(U)
        shells = tuple(shell_distances_squared)
        if any(int(d) != d or d <= 0 for d in shells) or len(set(shells)) != len(shells):
            raise ValueError("Squared shell distances must be distinct positive integers.")
        self.shell_distances_squared = tuple(int(d) for d in shells)
        self.pairs = {}
        x = numpy.arange(self.nsites) % self.lx
        y = numpy.arange(self.nsites) // self.lx
        dx = numpy.abs(x[:, None] - x[None, :])
        dy = numpy.abs(y[:, None] - y[None, :])
        distance_squared = numpy.minimum(dx, self.lx - dx) ** 2
        distance_squared += numpy.minimum(dy, self.ly - dy) ** 2
        for d in self.shell_distances_squared:
            pairs = numpy.nonzero(distance_squared == d)
            if len(pairs[0]) == 0:
                raise ValueError(f"No periodic site pairs at squared distance {d}.")
            self.pairs[d] = pairs
        self.names = (
            "energy_per_site",
            "density",
            "double_occupancy",
            "singles",
            *(f"magnetization_{i}" for i in range(self.nsites)),
            *(f"raw_czz_d2_{d}" for d in self.shell_distances_squared),
        )
        self.size = len(self.names)
        self.metadata = {
            "lx": self.lx,
            "ly": self.ly,
            "site_index": "x + lx*y",
            "periodic": True,
            "spin_normalization": "4*(<Sz_i Sz_j>-<Sz_i><Sz_j>), Sz=(n_up-n_down)/2",
            "energy_convention": "<K_hop + U*sum_i n_up_i*n_down_i>/nsites",
            "shell_distances_squared": list(self.shell_distances_squared),
            "shell_pair_counts": {str(d): len(p[0]) for d, p in self.pairs.items()},
            "raw_names": list(self.names),
        }

    def measure(self, G):
        r"""Raw observable vector of one Green's function.

        Parameters
        ----------
        G : :class:`numpy.ndarray`
            Particle contraction :math:`G_{s, ij} = \langle c_j^\dagger c_i \rangle`,
            shape (2, nsites, nsites).

        Returns
        -------
        values : :class:`numpy.ndarray`
            Complex raw vector, shape (size,).
        """
        G = numpy.asarray(to_host(G))
        if G.shape != (2, self.nsites, self.nsites):
            raise ValueError("G must be the particle contraction with shape (2,M,M).")
        occupations = numpy.diagonal(G, axis1=-2, axis2=-1)
        density = occupations[0] + occupations[1]
        double = occupations[0] * occupations[1]
        magnetization = occupations[0] - occupations[1]
        energy = numpy.einsum("sij,sji->", self.hop, G) + self.U * double.sum()
        out = numpy.empty(self.size, dtype=numpy.complex128)
        out[:4] = energy / self.nsites, density.mean(), double.mean(), (density - 2 * double).mean()
        out[4 : 4 + self.nsites] = magnetization
        for k, (i, j) in enumerate(self.pairs.values()):
            # i != j by construction. Exchange products have no conjugation:
            # the trace/transition contraction need not be Hermitian.
            exchange = numpy.sum(G[:, i, j] * G[:, j, i], axis=0)
            out[4 + self.nsites + k] = numpy.mean(magnetization[i] * magnetization[j] - exchange)
        return out

    def _derived(self, mean_vector):
        values = numpy.asarray(mean_vector, dtype=numpy.complex128)
        if values.shape != (self.size,):
            raise ValueError("Mean vector does not match the raw observable schema.")
        mag = values[4 : 4 + self.nsites]
        raw = values[4 + self.nsites :]
        connected = numpy.asarray(
            [raw[k] - numpy.mean(mag[i] * mag[j]) for k, (i, j) in enumerate(self.pairs.values())],
            dtype=numpy.complex128,
        )
        return numpy.concatenate((values[:4], [mag.mean()], raw, connected))

    def _as_dict(self, values):
        nshells = len(self.pairs)
        result = dict(
            zip(
                ("energy_per_site", "density", "double_occupancy", "singles", "magnetization"),
                (float(v) for v in values[:5]),
            )
        )
        result["raw_czz"] = {str(d): float(v) for d, v in zip(self.pairs, values[5 : 5 + nshells])}
        result["czz"] = {str(d): float(v) for d, v in zip(self.pairs, values[5 + nshells :])}
        return result


class HubbardTraceEstimator(EstimatorBase):
    """Block sums of trace observables for the estimator handler.

    Holds the sums of one block of :class:`ThermofieldHubbardAFQMC`, already
    reduced over processes, so that the handler writes them to its HDF5 file.

    Parameters
    ----------
    observables : :class:`TraceRatioObservables`
        Observables of the driver.
    """

    def __init__(self, observables):
        super().__init__()
        size = observables.size
        self._data = {
            "Numer": numpy.zeros(size, dtype=numpy.complex128),
            "NumerEndpoint": numpy.zeros(size, dtype=numpy.complex128),
            "NumerMixed": numpy.zeros(size, dtype=numpy.complex128),
            "Denom": numpy.zeros(1, dtype=numpy.complex128),
            "DenomMixed": numpy.zeros(1, dtype=numpy.complex128),
            "AbsoluteWeight": numpy.zeros(1, dtype=numpy.complex128),
            "SquaredWeight": numpy.zeros(1, dtype=numpy.complex128),
            "LogShift": numpy.zeros(1, dtype=numpy.complex128),
        }
        self._shape = (3 * size + 5,)
        self.print_to_stdout = False
        self.ascii_filename = None
        self.scalar_estimator = False

    def set_block(self, result, comm):
        """Store the reduced sums of one block.

        Parameters
        ----------
        result : dict
            Block result of :class:`ThermofieldHubbardAFQMC`.
        comm : MPI.COMM_WORLD
            MPI communicator; the sums are global, so only the root contributes.
        """
        self.zero()
        if comm.rank != 0:
            return
        self._data["Numer"][:] = result["numer"]
        self._data["NumerEndpoint"][:] = result["numer_endpoint"]
        self._data["Denom"][:] = result["denom"]
        self._data["AbsoluteWeight"][:] = result["absolute_weight"]
        self._data["SquaredWeight"][:] = result["squared_weight"]
        self._data["LogShift"][:] = result["log_shift"]
        if "numer_mixed" in result:
            self._data["NumerMixed"][:] = result["numer_mixed"]
            self._data["DenomMixed"][:] = result["denom_mixed"]

    def compute_estimator(self, system=None, walkers=None, hamiltonian=None, trial=None):
        # Set by `set_block`.
        return self.data

    @property
    def data(self):
        # The block sums live on the host, also on the GPU.
        return numpy.concatenate(list(self._data.values()))
