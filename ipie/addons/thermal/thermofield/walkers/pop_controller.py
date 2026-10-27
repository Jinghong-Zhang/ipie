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

from ipie.config import config
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import to_host
from ipie.walkers.pop_controller import PopController


class ThermofieldPopController(PopController):
    """Population control that reorders thermofield walkers in place.

    The core controller copies whole walker buffers between slots. On a single
    MPI process this controller runs it on a stand-in carrying only the
    weights, phases and walker ids, then applies the resulting slot to source
    map to the walker matrices in one gather, so device-resident walkers never
    move to the host. With several processes walkers move between ranks through
    the full buffers, staged through the host on the GPU.
    """

    def pop_control(self, walkers, comm):
        if comm.size > 1:
            if config.get_option("use_gpu"):
                walkers.cast_to_host()
                super().pop_control(walkers, comm)
                walkers.cast_to_cupy()
            else:
                super().pop_control(walkers, comm)
            return
        reorder_population(self, walkers, comm)


def reorder_population(pcontrol, walkers, comm):
    """Population control on one MPI process by reordering the walkers in place.

    Runs the population control of `pcontrol` on a stand-in with the weights,
    phases and walker ids, then reorders the walkers with the resulting map.

    Parameters
    ----------
    pcontrol : :class:`ipie.walkers.pop_controller.PopController`
        Population controller.
    walkers : :class:`ThermofieldWalkers`
        Walkers.
    comm : MPI.COMM_WORLD
        MPI communicator of size 1.
    """
    proxy = _PopControlProxy(walkers)
    PopController.pop_control(pcontrol, proxy, comm)
    ids = numpy.rint(to_host(proxy.wid).real).astype(numpy.int64)
    walkers.weight = to_host(proxy.weight)
    walkers.unscaled_weight = to_host(proxy.unscaled_weight)
    walkers.phase = to_host(proxy.phase)
    if not numpy.array_equal(ids, numpy.arange(walkers.nwalkers)):
        walkers.reorder(ids)


class _PopControlProxy:
    """Walker stand-in for population control: weights, phases and walker ids.

    The arrays live on the backend, where the core controller works on them.
    """

    def __init__(self, walkers):
        nwalkers = walkers.nwalkers
        self.nwalkers = nwalkers
        self.weight = xp.array(walkers.weight, dtype=xp.float64)
        self.unscaled_weight = xp.array(walkers.unscaled_weight, dtype=xp.float64)
        self.phase = xp.array(walkers.phase, dtype=xp.complex128)
        self.wid = xp.arange(nwalkers, dtype=xp.complex128)
        self.buff_names = ["weight", "unscaled_weight", "phase", "wid"]
        self.buff_size = 4
        self.walker_buffer = numpy.zeros(self.buff_size, dtype=numpy.complex128)

    def get_buffer(self, iw):
        return numpy.array(
            [
                to_host(self.weight[iw]),
                to_host(self.unscaled_weight[iw]),
                to_host(self.phase[iw]),
                to_host(self.wid[iw]),
            ],
            dtype=numpy.complex128,
        )

    def set_buffer(self, iw, buff):
        self.weight[iw] = buff[0].real
        self.unscaled_weight[iw] = buff[1].real
        self.phase[iw] = buff[2]
        self.wid[iw] = buff[3]
