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

from ipie.addons.thermal.thermofield.estimators.energy import (
    ThermofieldMixedEstimator,
    ThermofieldTraceEstimator,
)
from ipie.config import config
from ipie.estimators.handler import EstimatorHandler
from ipie.utils.io import format_fixed_width_strings


class ThermofieldEstimatorHandler(EstimatorHandler):
    """Container for the thermofield path estimators.

    Holds one :class:`ThermofieldMixedEstimator` per trial time, named
    "eta=<eta>", and the :class:`ThermofieldTraceEstimator` "trace". Estimates
    are reduced over processes and written once per block (one walk).

    Parameters
    ----------
    comm : MPI.COMM_WORLD
        MPI Communicator.
    hamiltonian :
        Hamiltonian describing the system.
    trial : :class:`ThermofieldThermalTrial`
        Trial.
    eta_members : tuple
        Trial times of the mixed estimators. Default ().
    measure_trace : bool
        Add the trace estimator. Default True.
    trace_branching : bool
        The walker weights carry the trace magnitude. Default False.
    walker_state : :class:`WalkerAccumulator` object
        WalkerAccumulator class.
    verbose : bool
        If true we print out additional setup information.
    filename : str
        .h5 file name for saving data. Default None: nothing is written; the
        driver returns the block sums.
    """

    def __init__(
        self,
        comm,
        hamiltonian,
        trial,
        eta_members=(),
        measure_trace: bool = True,
        trace_branching: bool = False,
        walker_state=None,
        verbose: bool = False,
        filename=None,
    ):
        if verbose:
            print("# Setting up estimator object.")
        self.write_output = filename is not None
        self.filename = filename if comm.rank == 0 else None
        if verbose and self.write_output:
            print(f"# Writing estimator data to {self.filename}")
        self.buffer_size = config.get_option("estimator_buffer_size")
        if walker_state is not None:
            self.num_walker_props = walker_state.size
            self.walker_header = walker_state.names
        else:
            self.num_walker_props = 0
            self.walker_header = ""
        self._estimators = {}
        self._shapes = []
        self._offsets = {}
        self.json_string = "{}"
        for eta in eta_members:
            est = ThermofieldMixedEstimator(eta, hamiltonian=hamiltonian, trial=trial)
            self[est.label] = est
        if measure_trace:
            est = ThermofieldTraceEstimator(
                hamiltonian=hamiltonian, trial=trial, trace_branching=trace_branching
            )
            self[est.label] = est
        if verbose:
            print("# Finished settting up estimator object.")

    def initialize(self, comm):
        if self.write_output:
            super().initialize(comm)
            return
        # The buffers and header of EstimatorHandler.initialize, without the file.
        self.local_estimates = numpy.zeros(
            (self.size + self.num_walker_props), dtype=numpy.complex128
        )
        self.global_estimates = numpy.zeros(
            (self.size + self.num_walker_props), dtype=numpy.complex128
        )
        self.output = _NullOutput()
        header = f"{'Block':>17s}  "
        header += format_fixed_width_strings(self.walker_header)
        header += " "
        for _, e in self.items():
            if e.print_to_stdout:
                header += e.header_to_text
        if comm.rank == 0:
            print(header)


class _NullOutput:
    """Stand-in for the HDF5 writer when no estimator file is requested."""

    def push_to_chunk(self, data, name):
        pass

    def increment(self):
        pass
