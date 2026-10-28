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

import numpy

from ipie.addons.thermal.thermofield.trial.thermofield import ThermofieldThermalTrial
from ipie.addons.thermal.thermofield.utils.linalg import pivoted_qr, qr_batched, SCALE_FLOOR
from ipie.addons.thermal.thermofield.walkers.pop_controller import population_control
from ipie.config import config
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import to_host
from ipie.walkers.base_walkers import BaseWalkers


class ThermofieldWalkers(BaseWalkers):
    r"""Open thermofield Gaussian walkers.

    Each walker is an nbasis x nbasis matrix per spin representing the purified
    state

    .. math::

        |\Phi(\Delta)\rangle = \exp(c^\dagger \Delta \tilde{c}^\dagger)
            |0 \tilde{0}\rangle.

    Walkers start from the identity purification :math:`\Delta = I` (infinite
    temperature) and are propagated on the physical side only,
    :math:`\Delta \leftarrow B(x) \Delta`. To keep the small scales of the
    propagator product, each walker is stored factorized,

    .. math::

        \Delta = Q e^{D} T,

    with bounded Q and T and the scales D in the log domain. Propagation
    multiplies Q from the left; :meth:`stabilize` refactorizes by column-pivoted
    QR without changing :math:`\Delta`.

    Parameters
    ----------
    trial : :class:`ThermofieldThermalTrial`
        Guide used for the cached overlaps and Green's functions.
    nbasis : int
        Number of basis functions.
    nwalkers : int
        Number of walkers on this MPI process.
    mpi_handler : :class:`ipie.utils.mpi.MPIHandler`
        MPI handler.
    verbose : bool
        Print information. Default False.

    Attributes
    ----------
    Qmat, Tmat : :class:`numpy.ndarray`
        Bounded factors, shape (nwalkers, 2, nbasis, nbasis).
    log_d : :class:`numpy.ndarray`
        Log scales, shape (nwalkers, 2, nbasis).
    log_ovlp : :class:`numpy.ndarray`
        Complex log of the guide overlap, shape (nwalkers,).
    Ga, Gb : :class:`numpy.ndarray`
        Mixed Green's functions in the thermal hole convention
        :math:`G_{ij} = \langle c_i c_j^\dagger \rangle = (I - P^T)_{ij}`, as
        consumed by the thermal force bias.
    Qinv : :class:`numpy.ndarray`
        Cached inverses of the Q factors, shape (nwalkers, 2, nbasis, nbasis),
        kept by propagators that update them incrementally. None when unused.
    log_det_Q : :class:`numpy.ndarray`
        Cached log determinants of the Q factors, shape (nwalkers, 2). None
        when unused.
    extra_walker_arrays : list
        (array, walker_axis) pairs that population control reorders together
        with the walkers, e.g. per-walker path histories owned by a driver.
    """

    def __init__(
        self,
        trial: ThermofieldThermalTrial,
        nbasis: int,
        nwalkers: int,
        mpi_handler=None,
        verbose: bool = False,
    ):
        assert isinstance(trial, ThermofieldThermalTrial)
        super().__init__(nwalkers, verbose=verbose)
        # Weights, phases and overlaps live on the host; only the walker
        # matrices move to the GPU.
        for name in ("weight", "unscaled_weight", "phase", "hybrid_energy", "ovlp", "sgn_ovlp"):
            setattr(self, name, to_host(getattr(self, name)))

        self.nbasis = nbasis
        self.mpi_handler = mpi_handler

        # Walker matrices are created on the backend (the GPU with IPIE_USE_GPU).
        self.Qmat = xp.zeros((nwalkers, 2, nbasis, nbasis), dtype=xp.complex128)
        self.Qmat[:] = xp.eye(nbasis)
        self.log_d = xp.zeros((nwalkers, 2, nbasis), dtype=xp.float64)
        self.Tmat = xp.zeros((nwalkers, 2, nbasis, nbasis), dtype=xp.complex128)
        self.Tmat[:] = xp.eye(nbasis)
        # Complex, unlike the real overlap of the base class.
        self.log_ovlp = numpy.zeros(nwalkers, dtype=numpy.complex128)

        self.Ga = xp.zeros((nwalkers, nbasis, nbasis), dtype=xp.complex128)
        self.Gb = xp.zeros((nwalkers, nbasis, nbasis), dtype=xp.complex128)
        self._identity = xp.eye(nbasis)

        self.Qinv = None
        self.log_det_Q = None
        self.extra_walker_arrays = []
        self.update_walker_cache_batched(trial)

        self.buff_names += ["Qmat", "log_d", "Tmat", "Ga", "Gb"]
        self.buff_size = round(self.set_buff_size_single_walker() / float(self.nwalkers))
        self.walker_buffer = numpy.zeros(self.buff_size, dtype=numpy.complex128)

    def add_walker_array(self, name, value):
        """Add a per-walker array that population control copies with the walkers.

        Parameters
        ----------
        name : str
            Attribute name.
        value : :class:`numpy.ndarray`
            Initial value, shape (nwalkers, ...).
        """
        setattr(self, name, value)
        if name not in self.buff_names:
            self.buff_names += [name]
        # The array may replace one of another shape.
        self.buff_size = round(self.set_buff_size_single_walker() / float(self.nwalkers))
        self.walker_buffer = numpy.zeros(self.buff_size, dtype=numpy.complex128)

    def get_delta(self, iw):
        """Walker matrices of walker `iw`, shape (2, nbasis, nbasis).

        Can overflow for large scales; meant for estimators and tests at
        moderate beta.
        """
        return numpy.array(
            [
                self.Qmat[iw, s] @ (numpy.exp(self.log_d[iw, s])[:, None] * self.Tmat[iw, s])
                for s in range(2)
            ]
        )

    def set_delta(self, iw, Delta, trial=None):
        """Set walker `iw` from plain matrices of shape (2, nbasis, nbasis)."""
        self.Qmat[iw] = Delta
        self.log_d[iw] = 0.0
        self.Tmat[iw] = numpy.eye(self.nbasis)
        if trial is not None:
            self.update_walker_cache(trial, iw)

    def update_walker_cache(self, trial, iw):
        """Recompute the guide overlap and Green's functions of walker `iw`."""
        log_ovlp, G = trial.calc_log_overlap_and_greens_function(
            self.Qmat[iw], self.log_d[iw], self.Tmat[iw]
        )
        self.set_walker_cache(iw, log_ovlp, G)

    def set_walker_cache(self, iw, log_ovlp, G):
        """Store a guide overlap and Green's function (particle convention) for walker `iw`."""
        self.Ga[iw] = self._identity - G[0]
        self.Gb[iw] = self._identity - G[1]
        self.log_ovlp[iw] = log_ovlp

    def update_walker_cache_batched(self, trial):
        """Recompute the guide overlaps and Green's functions of all walkers."""
        log_ovlp, G = trial.calc_log_overlap_and_greens_function_batched(
            self.Qmat, self.log_d, self.Tmat
        )
        self.set_walker_cache_batched(log_ovlp, G)

    def set_walker_cache_batched(self, log_ovlp, G):
        """Store guide overlaps and Green's functions (particle convention) of all walkers.

        Parameters
        ----------
        log_ovlp : :class:`numpy.ndarray`
            Log overlaps, shape (nwalkers,). Kept on the host.
        G : :class:`numpy.ndarray`
            Green's functions, shape (nwalkers, 2, nbasis, nbasis).
        """
        self.Ga[:] = self._identity - G[:, 0]
        self.Gb[:] = self._identity - G[:, 1]
        self.log_ovlp[:] = to_host(log_ovlp)

    # Walker matrices live on the GPU; weights, phases and overlaps stay on the host.
    _walker_matrices = ("Qmat", "log_d", "Tmat", "Ga", "Gb")
    _device_arrays = _walker_matrices + ("_identity",)

    def cast_to_cupy(self, verbose=False):
        """Copy the walker matrices to the GPU."""
        if verbose:
            print(f"# {self.__class__.__name__}: copying walker matrices to the GPU.")
        for name in self._device_arrays:
            setattr(self, name, xp.asarray(getattr(self, name)))

    def cast_to_host(self):
        """Copy the walker matrices back to the host, e.g. for population control across processes."""
        for name in self._device_arrays:
            setattr(self, name, to_host(getattr(self, name)))

    def reorder(self, ids):
        """Replace walker `i` by walker `ids[i]` for every walker.

        Every per-walker array in `buff_names` is reordered, except the weights
        and phases, which population control sets itself.
        The arrays in `extra_walker_arrays` are reordered as well.
        Cached Q inverses are not: propagators that keep them recompute them
        after population control.

        Parameters
        ----------
        ids : :class:`numpy.ndarray`
            Source walker of every slot, shape (nwalkers,).
        """
        ids_x = xp.asarray(ids)
        for name in self._walker_matrices:
            arr = getattr(self, name)
            arr[:] = arr[ids_x]
        # Host arrays: overlaps, traces and those registered by propagators or
        # drivers, in place so that references held elsewhere stay valid.
        for name in self.buff_names:
            if name not in ("weight", "unscaled_weight", "phase") + self._walker_matrices:
                arr = getattr(self, name)
                arr[:] = arr[ids]
        # Reorder leading index by leading index to bound the temporary.
        for arr, axis in self.extra_walker_arrays:
            if axis == 0:
                arr[:] = arr[ids_x]
            else:
                for lead in numpy.ndindex(*arr.shape[:axis]):
                    view = arr[lead]
                    view[:] = view[ids_x]

    def _per_walker_arrays(self):
        # Every array that moves with a walker, with the walker axis first; host
        # arrays stay on the host.
        arrays = [(getattr(self, name), 0) for name in self.buff_names]
        arrays += self.extra_walker_arrays
        return [
            (
                numpy.moveaxis(arr, axis, 0)
                if isinstance(arr, numpy.ndarray)
                else xp.moveaxis(arr, axis, 0)
            )
            for arr, axis in arrays
        ]

    def packed_size(self):
        """Length of one walker in :meth:`pack_walkers`."""
        return sum(view[0].size for view in self._per_walker_arrays())

    def pack_walkers(self, ids):
        """Copy of the walkers `ids` on the host, to send them to another process.

        Parameters
        ----------
        ids : :class:`numpy.ndarray`
            Walker indices.

        Returns
        -------
        buff : :class:`numpy.ndarray`
            Every per-walker array (`buff_names` and `extra_walker_arrays`) of
            the walkers, one row per walker, shape (len(ids), size).
        """
        parts = []
        for view in self._per_walker_arrays():
            rows = view[ids] if isinstance(view, numpy.ndarray) else to_host(view[xp.asarray(ids)])
            parts.append(rows.reshape(len(ids), -1))
        return numpy.concatenate(parts, axis=1).astype(numpy.complex128)

    def unpack_walkers(self, ids, buff):
        """Overwrite the walkers `ids` with rows of :meth:`pack_walkers`.

        Parameters
        ----------
        ids : :class:`numpy.ndarray`
            Walker indices, one per row of `buff`.
        buff : :class:`numpy.ndarray`
            Packed walkers, shape (len(ids), size).
        """
        start = 0
        for view in self._per_walker_arrays():
            size = view[0].size
            rows = buff[:, start : start + size].reshape((len(ids),) + view.shape[1:])
            if not numpy.iscomplexobj(view):
                rows = rows.real
            if isinstance(view, numpy.ndarray):
                view[ids] = rows
            else:
                view[xp.asarray(ids)] = xp.asarray(rows)
            start += size

    def stabilize(self):
        """Refactorize every walker by column-pivoted QR without changing it.

        The propagator factors accumulated in Q move into a fresh unitary Q, log
        scales D and a bounded factor merged into T.
        On the GPU a batched non-pivoted Householder QR (:func:`qr_batched`) is
        used: the factorization differs, the walker does not.
        """
        nwalkers, nbasis = self.nwalkers, self.nbasis
        if config.get_option("use_gpu"):
            shift = xp.max(self.log_d, axis=-1)
            F = self.Qmat * xp.exp(self.log_d - shift[..., None])[..., None, :]
            Qs, R = qr_batched(F)
            absd = xp.abs(xp.diagonal(R, axis1=-2, axis2=-1))
            floor = xp.maximum(absd.max(axis=-1), 1.0) * SCALE_FLOOR
            absd = xp.maximum(absd, floor[..., None])
            # Unit-scale rows; S keeps the phases of R.
            S = (1.0 / absd)[..., :, None] * R
            self.Qmat[:] = Qs
            self.log_d[:] = xp.log(absd) + shift[..., None]
            self.Tmat[:] = S @ self.Tmat
            return
        shift = numpy.max(self.log_d, axis=-1)
        # FT[iw, s].T is the Fortran-ordered F = Q e^{D - shift} that LAPACK
        # factorizes in place.
        FT = (
            numpy.swapaxes(self.Qmat, -1, -2)
            * numpy.exp(self.log_d - shift[..., None])[..., :, None]
        )

        # Walkers whose spin blocks are bitwise identical share the QR.
        same = (
            numpy.all(self.Qmat[:, 0] == self.Qmat[:, 1], axis=(-2, -1))
            & numpy.all(self.log_d[:, 0] == self.log_d[:, 1], axis=-1)
            & numpy.all(self.Tmat[:, 0] == self.Tmat[:, 1], axis=(-2, -1))
        )

        Qs = numpy.empty_like(self.Qmat)
        R = numpy.empty_like(self.Qmat)
        perm = numpy.empty((nwalkers, 2, nbasis), dtype=numpy.int64)
        for iw in range(nwalkers):
            for s in range(2):
                if s == 1 and same[iw]:
                    Qs[iw, 1] = Qs[iw, 0]
                    R[iw, 1] = R[iw, 0]
                    perm[iw, 1] = perm[iw, 0]
                    continue
                Qs[iw, s], R[iw, s], perm[iw, s] = pivoted_qr(FT[iw, s].T)

        # Floor tiny scales so 1/d and log(d) stay finite at large beta. Pivoting
        # makes |R_ii| dominate its row, so the deficit moves exactly into the
        # bounded rows of S and S diag(d) = R still holds.
        absd = numpy.abs(numpy.diagonal(R, axis1=-2, axis2=-1))
        floor = numpy.maximum(absd.max(axis=-1), 1.0) * SCALE_FLOOR
        absd = numpy.maximum(absd, floor[..., None])
        # S[:, perm] = diag(1/d) R, i.e. S = (diag(1/d) R)[:, invperm].
        invperm = numpy.empty_like(perm)
        numpy.put_along_axis(invperm, perm, numpy.arange(nbasis)[None, None, :], axis=-1)
        S = numpy.take_along_axis((1.0 / absd)[..., :, None] * R, invperm[..., None, :], axis=-1)
        self.Qmat[:] = Qs
        self.log_d[:] = numpy.log(absd) + shift[..., None]
        self.Tmat[:] = S @ self.Tmat

    def reset(self, trial, compute_cache=True):
        """Reset all walkers to the identity purification with unit weight.

        Parameters
        ----------
        trial : :class:`ThermofieldThermalTrial`
            Guide for the cached overlaps.
        compute_cache : bool
            Recompute the cached overlaps. False is for callers that recompute
            them at another guide time right away. Default True.
        """
        self.weight = numpy.ones(self.nwalkers)
        self.unscaled_weight = numpy.ones(self.nwalkers)
        self.phase = numpy.ones(self.nwalkers, dtype=numpy.complex128)
        self.Qmat[:] = 0.0
        self.Qmat[:] = xp.eye(self.nbasis)
        self.log_d[:] = 0.0
        self.Tmat[:] = 0.0
        self.Tmat[:] = xp.eye(self.nbasis)
        if compute_cache:
            self.update_walker_cache_batched(trial)

    # For compatibility with BaseWalkers: stabilization is the QDT refactorization.
    def reortho(self):
        pass

    def reortho_batched(self):
        pass

    # Compatibility with the per-array GPU interface of earlier versions.
    @property
    def on_device(self):
        return not isinstance(self.Qmat, numpy.ndarray)

    def to_device(self):
        """Copy the walker matrices to the GPU; needs IPIE_USE_GPU=1. Use `cast_to_cupy`."""
        if not config.get_option("use_gpu"):
            raise RuntimeError("to_device needs the GPU backend: set IPIE_USE_GPU=1.")
        self.cast_to_cupy()
        return self

    def to_host(self):
        """Copy the walker matrices to the host. Use `cast_to_host`."""
        self.cast_to_host()
        return self

    def pop_control_inplace(self, pcontrol, comm):
        """Population control by reordering the walkers. Use :class:`ThermofieldPopController`."""
        population_control(pcontrol, self, comm)
