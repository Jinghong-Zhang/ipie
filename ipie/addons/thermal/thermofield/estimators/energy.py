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

from ipie.addons.thermal.thermofield.estimators.local_energy import local_energy_batched
from ipie.config import MPI
from ipie.estimators.estimator_base import EstimatorBase
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import to_host
from ipie.utils.io import format_fixed_width_floats, format_fixed_width_strings


class ThermofieldMixedEstimator(EstimatorBase):
    r"""Energy and particle number with the trial at imaginary time eta.

    Measured on the walkers at :math:`\theta = \beta - \eta` with the bra
    :math:`e^{-\eta k_T}` in place of the guide,

    .. math::

        E(\eta) = \frac{\sum_i w_i F_i E_L(\Delta_i)}{\sum_i w_i F_i}, \qquad
        F_i = \frac{S_\eta(\Delta_i)}{S_T(\Delta_i)},

    where :math:`E_L` uses the Green's function of the bra at the walker
    frontier. Numerator and denominator are kept separately, scaled by
    :math:`e^{-s}` with a log scale s common to all processes (the largest
    :math:`\log F` plus the population-control normalization), so that blocks
    combine as a pooled ratio of means.

    Parameters
    ----------
    eta : float
        Trial time. 0 gives the closed trace :math:`\det(I + \Delta)`.
    hamiltonian :
        Hamiltonian.
    trial : :class:`ThermofieldThermalTrial`
        Trial providing :math:`k_T` and the guide.
    filename : str
        Text file for the block data. Default None.

    Attributes
    ----------
    global_sums : :class:`numpy.ndarray`
        Denominator, energy and particle-number numerators of the last
        measurement summed over all processes.
    log_scale : float
        Log scale of `global_sums`.
    """

    def __init__(self, eta, hamiltonian=None, trial=None, filename=None):
        super().__init__()
        self.eta = eta
        self.label = f"eta={eta:g}"
        self._data = {
            "ENumer": 0.0j,
            "EDenom": 0.0j,
            "NavNumer": 0.0j,
            "NavDenom": 0.0j,
            "LogScale": 0.0j,
            "ETotal": 0.0j,
            "Nav": 0.0j,
        }
        self._shape = (len(self.names),)
        self._data_index = {k: i for i, k in enumerate(list(self._data.keys()))}
        self.print_to_stdout = True
        self.ascii_filename = filename
        self.scalar_estimator = True
        self.global_sums = numpy.zeros(3, dtype=numpy.complex128)
        self.log_scale = 0.0

    def calc_local_quantities(self, walkers, hamiltonian, trial):
        r"""Log reweighting factor, local energy and particle number of every walker.

        Returns
        -------
        log_factor : :class:`numpy.ndarray`
            :math:`\log F`, shape (nwalkers,).
        energy, nav : :class:`numpy.ndarray`
            Local energies and particle numbers, shape (nwalkers,).
        """
        # The observable sits at the walker frontier, whatever split the force
        # bias uses.
        log_ovlp, G = trial.calc_log_overlap_and_greens_function_batched(
            walkers.Qmat, walkers.log_d, walkers.Tmat, guide_time=self.eta, split="frontier"
        )
        P = G.transpose(0, 1, 3, 2)  # P[s] = G[s]^T.
        energy = to_host(local_energy_batched(hamiltonian, P))[:, 0]
        nav = to_host(xp.einsum("wsii->w", P))
        return to_host(log_ovlp) - walkers.log_ovlp, energy, nav

    def measure(self, walkers, hamiltonian, trial, comm, log_norm=0.0):
        """Measure on the current walkers; collective over `comm`.

        Walkers with zero weight or a non-finite factor or observable are
        dropped.

        Parameters
        ----------
        walkers : :class:`ThermofieldWalkers`
            Walkers.
        hamiltonian :
            Hamiltonian.
        trial : :class:`ThermofieldThermalTrial`
            Trial.
        comm : MPI.COMM_WORLD
            MPI communicator.
        log_norm : float
            Log of the weight removed by population control so far in the walk.
            Default 0.0.
        """
        log_factor, energy, nav = self.calc_local_quantities(walkers, hamiltonian, trial)
        weight = walkers.weight * walkers.phase
        alive = numpy.isfinite(log_factor) & (weight != 0.0)
        for obs in (energy, nav):
            alive &= numpy.isfinite(obs)
        local_max = float(log_factor.real[alive].max()) if alive.any() else -numpy.inf
        shift = comm.allreduce(local_max, op=MPI.MAX)
        if not numpy.isfinite(shift):
            shift = 0.0
        with numpy.errstate(over="ignore", invalid="ignore"):
            wf = numpy.where(alive, weight * numpy.exp(log_factor - shift), 0.0)
        local = numpy.array(
            [numpy.sum(wf)]
            + [numpy.sum(wf * numpy.where(alive, obs, 0.0)) for obs in (energy, nav)],
            dtype=numpy.complex128,
        )
        comm.Allreduce(local, self.global_sums)
        self.log_scale = shift + log_norm
        self._data["EDenom"] = local[0]
        self._data["ENumer"] = local[1]
        self._data["NavDenom"] = local[0]
        self._data["NavNumer"] = local[2]
        # Same on every process, so only the root contributes to the reduction.
        self._data["LogScale"] = self.log_scale if comm.rank == 0 else 0.0

    def compute_estimator(self, system=None, walkers=None, hamiltonian=None, trial=None):
        # Measured during the walk by `measure`.
        return self.data

    def get_index(self, name):
        index = self._data_index.get(name, None)
        if index is None:
            raise RuntimeError(f"Unknown estimator {name}")
        return index

    @property
    def header_to_text(self) -> str:
        return format_fixed_width_strings([f"E({self.label})", f"N({self.label})"])

    def data_to_text(self, vals) -> str:
        return format_fixed_width_floats(
            vals[[self.get_index("ETotal"), self.get_index("Nav")]].real
        )

    def post_reduce_hook(self, data):
        with numpy.errstate(divide="ignore", invalid="ignore"):
            data[self.get_index("ETotal")] = (
                data[self.get_index("ENumer")] / data[self.get_index("EDenom")]
            )
            data[self.get_index("Nav")] = (
                data[self.get_index("NavNumer")] / data[self.get_index("NavDenom")]
            )


class ThermofieldTraceEstimator(ThermofieldMixedEstimator):
    r"""Closed-trace energy and particle number at the end of the walk.

    The :math:`\eta = 0` member: the bra is the identity and F is the endpoint
    correction :math:`Q = \det(I + \Delta) / S_T(\Delta)`.

    Parameters
    ----------
    hamiltonian :
        Hamiltonian.
    trial : :class:`ThermofieldThermalTrial`
        Trial.
    filename : str
        Text file for the block data. Default None.
    """

    def __init__(self, hamiltonian=None, trial=None, filename=None):
        super().__init__(0.0, hamiltonian=hamiltonian, trial=trial, filename=filename)
        self.label = "trace"
