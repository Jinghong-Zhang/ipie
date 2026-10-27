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

from ipie.config import config, MPI
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import to_host
from ipie.walkers.pop_controller import PopController


class ThermofieldPopController(PopController):
    """Population control that reorders thermofield walkers in place.

    The core controller copies whole walker buffers between slots, one walker
    and one MPI message at a time. This controller finds the slot to source map
    from the weights alone and applies it to the walker matrices in one
    gather, so device-resident walkers move to the host only to change
    process: pair branching runs on the host, other methods run the core
    controller on a stand-in carrying the weights, phases and walker ids on one
    process, and the core controller itself with several processes (on the
    CPU only).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if config.get_option("use_gpu") and self.size > 1 and self.method != "pair_branch":
            raise NotImplementedError(
                f"{self.method} population control across processes is not available on the GPU."
            )

    def pop_control(self, walkers, comm):
        if self.method == "pair_branch":
            pair_branch_population(self, walkers, comm)
        elif comm.size > 1:
            super().pop_control(walkers, comm)
        else:
            reorder_population(self, walkers, comm)


def pair_branch_population(pcontrol, walkers, comm):
    """Pair-branch population control by reordering the walkers.

    The arithmetic and random numbers of the core controller with
    :func:`ipie.walkers.pop_controller.pair_branch`: the weights are rescaled
    to the target, the root pairs the smallest and largest weights of the
    whole population, and a walker cloned on rank a for rank b fills the next
    killed slot on b whose partner is on a, both in index order, as MPI
    matches the core controller's messages. Clones that stay on their process
    are gathered in place; the others travel as one host buffer per pair of
    processes, so no GPU-aware MPI is needed.

    Parameters
    ----------
    pcontrol : :class:`ipie.walkers.pop_controller.PopController`
        Population controller.
    walkers : :class:`ThermofieldWalkers`
        Walkers.
    comm : MPI.COMM_WORLD
        MPI communicator.
    """
    if pcontrol.ntot_walkers == 1:
        return
    nwalkers, rank, size = walkers.nwalkers, comm.rank, comm.size
    unscaled = numpy.array(walkers.weight, dtype=numpy.float64)
    total = numpy.empty(1, dtype=numpy.float64)
    comm.Reduce(numpy.sum(numpy.abs(unscaled)), total, op=MPI.SUM, root=0)
    comm.Bcast(total, root=0)
    weight = unscaled / (total[0] / pcontrol.target_weight)
    pcontrol.total_weight = pcontrol.target_weight
    gathered = numpy.empty((size, nwalkers), dtype=numpy.float64) if rank == 0 else None
    comm.Gather(numpy.abs(weight), gathered, root=0)
    # Columns: weight, status (0 killed, 1 kept, 2 cloned), rank of the partner.
    info = None
    if rank == 0:
        info = numpy.zeros((size * nwalkers, 3))
        info[:, 0] = gathered.ravel()
        info[:, 1] = 1
        info[:, 2] = numpy.repeat(numpy.arange(size), nwalkers)
        sort = numpy.argsort(info[:, 0], kind="mergesort")
        isort = numpy.argsort(sort, kind="mergesort")
        info = info[sort]
        s, e = 0, len(info) - 1
        while s < e:
            if info[s, 0] < pcontrol.min_weight or info[e, 0] > pcontrol.max_weight:
                wab = info[s, 0] + info[e, 0]
                r = numpy.random.rand()
                clone, kill = (e, s) if r < info[e, 0] / wab else (s, e)
                rank_clone, rank_kill = info[clone, 2], info[kill, 2]
                info[clone] = 0.5 * wab, 2, rank_kill
                info[kill] = 0.0, 0, rank_clone
                s += 1
                e -= 1
            else:
                break
        info = info[isort]
    data = numpy.empty((nwalkers, 3), dtype=numpy.float64)
    comm.Scatter(info, data, root=0)
    status = data[:, 1]
    partner = numpy.rint(data[:, 2]).astype(numpy.int64)
    clones = numpy.flatnonzero(status > 1)
    kills = numpy.flatnonzero(status == 0)
    weight[clones] = data[clones, 0]
    walkers.weight = weight
    walkers.unscaled_weight = unscaled
    walkers.phase = numpy.array(walkers.phase)
    sends, recvs = {}, {}
    ids = numpy.arange(nwalkers)
    for other in range(size):
        out, into = clones[partner[clones] == other], kills[partner[kills] == other]
        if other == rank:
            ids[into] = out
        elif len(out) or len(into):
            sends[other], recvs[other] = out, into
    buffers = {other: walkers.pack_walkers(out) for other, out in sends.items() if len(out)}
    requests = [comm.Isend(buff, dest=other) for other, buff in buffers.items()]
    if not numpy.array_equal(ids, numpy.arange(nwalkers)):
        walkers.weight = walkers.weight[ids]
        walkers.unscaled_weight = walkers.unscaled_weight[ids]
        walkers.phase = walkers.phase[ids]
        walkers.reorder(ids)
    for other, into in recvs.items():
        if len(into):
            buff = numpy.empty((len(into), walkers.packed_size()), dtype=numpy.complex128)
            comm.Recv(buff, source=other)
            walkers.unpack_walkers(into, buff)
    MPI.Request.Waitall(requests)
    pcontrol.pop_control_counter += 1


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
