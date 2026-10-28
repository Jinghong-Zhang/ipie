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

from ipie.config import MPI
from ipie.walkers.pop_controller import minimize_communication, PopController


class ThermofieldPopController(PopController):
    """Population control that reorders thermofield walkers in place.

    The core controller copies whole walker buffers between slots, one walker
    and one MPI message at a time. This controller finds the slot to source map
    from the weights alone, with the arithmetic and random numbers of the core
    controller, and applies it to the walker matrices in one gather on every
    process; walkers that change process travel as one host buffer per pair of
    processes, so device-resident walkers need no GPU-aware MPI. Comb copies
    every walker as often as the comb selects it, which the core controller
    does only for up to two copies.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.method not in ("pair_branch", "comb", "stochastic_reconfiguration"):
            raise ValueError(f"Unknown population control method {self.method}.")
        if self.correlated_samp:
            raise NotImplementedError(
                "Correlated sampling is not available for thermofield walkers."
            )

    def pop_control(self, walkers, comm):
        population_control(self, walkers, comm)


def population_control(pcontrol, walkers, comm):
    """Population control of `pcontrol.method` by reordering the walkers.

    Parameters
    ----------
    pcontrol : :class:`ipie.walkers.pop_controller.PopController`
        Population controller.
    walkers : :class:`ThermofieldWalkers`
        Walkers.
    comm : MPI.COMM_WORLD
        MPI communicator.
    """
    if pcontrol.method == "pair_branch":
        pair_branch_population(pcontrol, walkers, comm)
    elif pcontrol.method == "comb":
        comb_population(pcontrol, walkers, comm)
    elif pcontrol.method == "stochastic_reconfiguration":
        stochastic_reconfiguration_population(pcontrol, walkers, comm)
    else:
        raise ValueError(f"Unknown population control method {pcontrol.method}.")


def pair_branch_population(pcontrol, walkers, comm):
    """Pair-branch population control by reordering the walkers.

    The arithmetic and random numbers of the core controller with
    :func:`ipie.walkers.pop_controller.pair_branch`: the weights are rescaled
    to the target, the root pairs the smallest and largest weights of the
    whole population, and a walker cloned on rank a for rank b fills the next
    killed slot on b whose partner is on a, both in index order, as MPI
    matches the core controller's messages.

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
    _exchange_walkers(walkers, comm, ids, sends, recvs)
    pcontrol.pop_control_counter += 1


def comb_population(pcontrol, walkers, comm):
    """Comb population control by reordering the walkers.

    The arithmetic and random numbers of :func:`ipie.walkers.pop_controller.comb`
    (Booth and Gubernatis, PRE 80, 046704 (2009)): the root lays a comb of
    ntot_walkers evenly spaced teeth with one random offset over the
    cumulative weights, every walker is copied once per tooth on its weight,
    and every copy has unit weight. The copies keep their slots where they can
    (:func:`ipie.walkers.pop_controller.minimize_communication`), which equals
    the core controller when no walker is copied more than twice.

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
    unscaled = numpy.array(walkers.weight, dtype=numpy.float64)
    weights = numpy.empty(walkers.nwalkers * comm.size)
    comm.Allgather(numpy.abs(unscaled), weights)
    # Sequential sums, as the core controller.
    scale = sum(weights) / pcontrol.target_weight
    weights = weights / scale
    source = None
    if comm.rank == 0:
        total = sum(weights)
        cprobs = numpy.cumsum(weights)
        r = numpy.random.random()
        teeth = (numpy.arange(pcontrol.target_weight) + r) * (total / pcontrol.target_weight)
        # Tooth c belongs to the first walker whose cumulative weight exceeds it.
        copies = numpy.bincount(
            numpy.searchsorted(cprobs, teeth, side="right"), minlength=len(weights)
        )
        source, _ = minimize_communication(numpy.repeat(numpy.arange(len(weights)), copies))
    source = comm.bcast(source, root=0)
    pcontrol.total_weight = pcontrol.target_weight
    walkers.unscaled_weight = unscaled
    walkers.phase = numpy.array(walkers.phase)
    _move_walkers(walkers, comm, source)
    walkers.weight = numpy.ones(walkers.nwalkers)
    pcontrol.pop_control_counter += 1


def stochastic_reconfiguration_population(pcontrol, walkers, comm):
    """Stochastic reconfiguration by reordering the walkers.

    The arithmetic and random numbers of
    :func:`ipie.walkers.pop_controller.stochastic_reconfiguration`: the root
    draws ntot_walkers evenly spaced points with one random offset on the
    cumulative weights, slot i takes the walker of point i where
    :func:`ipie.walkers.pop_controller.minimize_communication` places it, and
    every walker gets the average weight.

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
    nwalkers, size = walkers.nwalkers, comm.size
    unscaled = numpy.array(walkers.weight, dtype=numpy.float64)
    total = numpy.empty(1, dtype=numpy.float64)
    comm.Reduce(numpy.sum(numpy.abs(unscaled)), total, op=MPI.SUM, root=0)
    comm.Bcast(total, root=0)
    weight = unscaled / (total[0] / pcontrol.target_weight)
    gathered = numpy.empty((size, nwalkers), dtype=numpy.float64) if comm.rank == 0 else None
    comm.Gather(weight, gathered, root=0)
    source = average = None
    if comm.rank == 0:
        cumulative = numpy.cumsum(numpy.abs(gathered))
        average = cumulative[-1] / nwalkers / size
        zeta = numpy.random.rand()
        points = (numpy.arange(size * nwalkers) + zeta) / nwalkers / size * cumulative[-1]
        source, _ = minimize_communication(numpy.searchsorted(cumulative, points))
    source, average = comm.bcast((source, average), root=0)
    pcontrol.total_weight = pcontrol.target_weight
    walkers.unscaled_weight = unscaled
    walkers.phase = numpy.array(walkers.phase)
    _move_walkers(walkers, comm, source)
    walkers.weight = numpy.full(nwalkers, average)
    pcontrol.pop_control_counter += 1


def _move_walkers(walkers, comm, source):
    """Replace every walker by its source, given in the global walker order.

    Parameters
    ----------
    walkers : :class:`ThermofieldWalkers`
        Walkers.
    comm : MPI.COMM_WORLD
        MPI communicator.
    source : :class:`numpy.ndarray`
        Global index of the source of every global slot, shape (nwalkers *
        comm.size,).
    """
    nwalkers, rank = walkers.nwalkers, comm.rank
    owner = (source // nwalkers).reshape(comm.size, nwalkers)
    local = (source % nwalkers).reshape(comm.size, nwalkers)
    sends, recvs = {}, {}
    ids = numpy.arange(nwalkers)
    for other in range(comm.size):
        into = numpy.flatnonzero(owner[rank] == other)
        if other == rank:
            ids[into] = local[rank, into]
            continue
        # The slots of `other` that take walkers of this process, in slot order.
        out = local[other, owner[other] == rank]
        if len(out) or len(into):
            sends[other], recvs[other] = out, into
    _exchange_walkers(walkers, comm, ids, sends, recvs)


def _exchange_walkers(walkers, comm, ids, sends, recvs):
    """Gather the walkers of this process and exchange walkers with the others.

    Outgoing walkers are copied to host buffers first, one buffer per
    process, then the walkers of this process are reordered, then the
    incoming walkers overwrite their slots.

    Parameters
    ----------
    walkers : :class:`ThermofieldWalkers`
        Walkers.
    comm : MPI.COMM_WORLD
        MPI communicator.
    ids : :class:`numpy.ndarray`
        Local source of every slot; slots in `recvs` keep their own index.
    sends : dict
        Walkers sent to each other process, in the order of its receiving slots.
    recvs : dict
        Slots that receive from each other process, in order.
    """
    buffers = {other: walkers.pack_walkers(out) for other, out in sends.items() if len(out)}
    requests = [comm.Isend(buff, dest=other) for other, buff in buffers.items()]
    if not numpy.array_equal(ids, numpy.arange(walkers.nwalkers)):
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
