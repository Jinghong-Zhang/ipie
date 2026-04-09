import time
import os
import h5py
import numpy

from ipie.config import MPI
from ipie.utils.backend import arraylib as xp


class PopControllerTimer:
    def __init__(self):
        self.start_time_const = 0.0
        self.communication_time = 0.0
        self.non_communication_time = 0.0
        self.recv_time = 0.0
        self.send_time = 0.0

    def start_time(self):
        self.start_time_const = time.time()

    def add_non_communication(self):
        self.non_communication_time += time.time() - self.start_time_const

    def add_communication(self):
        self.communication_time += time.time() - self.start_time_const

    def add_recv_time(self):
        self.recv_time += time.time() - self.start_time_const

    def add_send_time(self):
        self.send_time += time.time() - self.start_time_const


class PopController:
    def __init__(
        self,
        num_walkers_local,
        num_steps,
        mpi_handler=None,
        pop_control_method="pair_branch",
        min_weight=0.1,
        max_weight=4,
        verbose=False,
        correlated_samp=False,
        reference_run=False,
        walkermap_filepath=None,
    ):
        self.verbose = verbose

        self.num_walkers_local = num_walkers_local
        self.num_steps = num_steps
        self.mpi_handler = mpi_handler

        self.method = pop_control_method
        if verbose:
            print(f"# Using {self.method} population control " "algorithm.")

        self.min_weight = min_weight
        self.max_weight = max_weight
        self.pop_control_counter = 0

        self.mpi_handler = mpi_handler

        if self.mpi_handler is not None:
            self.size = self.mpi_handler.size
            self.ntot_walkers = num_walkers_local * self.size
        else:
            self.size = 1
            self.ntot_walkers = num_walkers_local

        self.target_weight = self.ntot_walkers
        self.total_weight = self.ntot_walkers
        self.correlated_samp = correlated_samp
        self.reference_run = reference_run
        self.walkermap_filepath = walkermap_filepath

        if verbose:
            print(f"# target weight is {self.target_weight}")
            print(f"# total weight is {self.total_weight}")

        self.timer = PopControllerTimer()

    def pop_control(self, walkers, comm):
        self.timer.start_time()
        if self.ntot_walkers == 1:
            return
        weights = numpy.abs(xp.array(walkers.weight))
        global_weights = numpy.empty(len(weights) * comm.size)
        self.timer.add_non_communication()
        self.timer.start_time()
        if self.method == "comb":
            comm.Allgather(weights, global_weights)
            total_weight = sum(global_weights)
        else:
            sum_weights = numpy.sum(weights)
            total_weight = numpy.empty(1, dtype=numpy.float64)
            if hasattr(sum_weights, "get"):
                sum_weights = sum_weights.get()
            comm.Reduce(sum_weights, total_weight, op=MPI.SUM, root=0)
            comm.Bcast(total_weight, root=0)
            total_weight = total_weight[0]

        self.timer.add_communication()
        self.timer.start_time()

        # Rescale weights to combat exponential decay/growth.
        scale = total_weight / self.target_weight
        if total_weight < 1e-8:
            if comm.rank == 0:
                print(f"# Warning: Total weight is {total_weight:13.8e}")
                print("# Something is seriously wrong.")
            # raise ValueError
        self.total_weight = total_weight
        
        # Todo: Just standardise information we want to send between routines.
        walkers.unscaled_weight = walkers.weight
        walkers.weight = walkers.weight / scale
        if hasattr(walkers, "walkers_A") and hasattr(walkers, "walkers_B"):
            walkers.walkers_A.unscaled_weight = walkers.walkers_A.weight
            walkers.walkers_B.unscaled_weight = walkers.walkers_B.weight
            walkers.walkers_A.weight = walkers.walkers_A.weight / scale
            walkers.walkers_B.weight = walkers.walkers_B.weight / scale
        self.total_weight = self.target_weight
        if self.method == "comb":
            global_weights = global_weights / scale
            self.timer.add_non_communication()
            comb(walkers, comm, global_weights, self.target_weight, self.timer)
        elif self.method == "pair_branch":
            pair_branch(walkers, comm, self.max_weight, self.min_weight, self.timer)
        elif self.method == "stochastic_reconfiguration":
            if not self.correlated_samp:
                stochastic_reconfiguration(walkers, comm, self.timer, self.pop_control_counter)
            else:
                if self.reference_run:
                    stochastic_reconfiguration(
                        walkers,
                        comm,
                        self.timer,
                        self.pop_control_counter,
                        store_walkermap=True,
                        walkermap_file=self.walkermap_filepath,
                    )
                else:
                    stochastic_reconfiguration(
                        walkers,
                        comm,
                        self.timer,
                        self.pop_control_counter,
                        read_walkermap=True,
                        walkermap_file=self.walkermap_filepath,
                    )
        elif self.method == "stochastic_reconfiguration_independent_repairing":
            stochastic_reconfiguration_independent_repairing(
                walkers,
                comm,
                self.timer,
                self.pop_control_counter,
                store_walkermap=self.reference_run,
                read_walkermap=(self.correlated_samp and not self.reference_run),
                walkermap_file=self.walkermap_filepath,
            )
        else:
            if comm.rank == 0:
                print("Unknown population control method.")
        self.pop_control_counter += 1


def get_buffer(walkers, iw):
    """Get iw-th walker buffer for MPI communication
    iw : int
        the walker index of interest
    Returns
    -------
    buff : dict
        Relevant walker information for population control.
    """
    if hasattr(walkers, "get_buffer"):
        return walkers.get_buffer(iw)

    s = 0
    buff = xp.zeros(walkers.buff_size, dtype=numpy.complex128)
    for d in walkers.buff_names:
        data = walkers.__dict__[d]
        if data is None:
            continue
        assert data.size % walkers.nwalkers == 0  # Only walker-specific data is being communicated
        if isinstance(data[iw], (xp.ndarray)):
            buff[s : s + data[iw].size] = xp.array(data[iw].ravel())
            s += data[iw].size
        elif isinstance(data[iw], list):  # when data is list
            for l in data[iw]:
                if isinstance(l, (xp.ndarray)):
                    buff[s : s + l.size] = xp.array(l.ravel())
                    s += l.size
                elif isinstance(l, (int, float, complex, numpy.float64, numpy.complex128)):
                    buff[s : s + 1] = l
                    s += 1
        else:
            buff[s : s + 1] = xp.array(data[iw])
            s += 1
    return buff


def set_buffer(walkers, iw, buff):
    """Set walker buffer following MPI communication
    Parameters
    -------
    buff : dict
        Relevant walker information for population control.
    """
    if hasattr(walkers, "set_buffer"):
        walkers.set_buffer(iw, buff)
        return

    s = 0
    for d in walkers.buff_names:
        data = walkers.__dict__[d]
        if data is None:
            continue
        assert data.size % walkers.nwalkers == 0  # Only walker-specific data is being communicated
        if isinstance(data[iw], xp.ndarray):
            if walkers.__dict__[d][iw].dtype == numpy.float64:
                walkers.__dict__[d][iw] = xp.array(
                    buff[s : s + data[iw].size].reshape(data[iw].shape).real.copy(),
                )
            elif walkers.__dict__[d][iw].dtype == numpy.complex128:
                walkers.__dict__[d][iw] = xp.array(
                    buff[s : s + data[iw].size].reshape(data[iw].shape).copy(),
                )
            s += data[iw].size
        elif isinstance(data[iw], list):
            for ix, l in enumerate(data[iw]):
                if isinstance(l, (xp.ndarray)):
                    walkers.__dict__[d][iw][ix] = xp.array(
                        buff[s : s + l.size].reshape(l.shape).copy()
                    )
                    s += l.size
                elif isinstance(l, (int, float, complex)):
                    walkers.__dict__[d][iw][ix] = buff[s]
                    s += 1
        else:
            if isinstance(walkers.__dict__[d][iw], (int, numpy.int64)):
                walkers.__dict__[d][iw] = int(buff[s].real)
            elif isinstance(walkers.__dict__[d][iw], (float, numpy.float64)):
                walkers.__dict__[d][iw] = buff[s].real
            else:
                walkers.__dict__[d][iw] = buff[s]
            s += 1


def minimize_communication(new_idx):
    """
    Given new_idx, a 1D int array of length N (monotonic or not),
    returns a permutation out of the same multiset that maximizes
    the count of i where out[i] == i.
    """
    N = new_idx.size
    counts = numpy.bincount(new_idx, minlength=N)
    counts_out = counts.copy()

    out = -numpy.ones(N, dtype=int)

    for i in range(N):
        if counts[i] > 0:
            out[i] = i
            counts[i] -= 1

    leftovers = numpy.repeat(numpy.arange(N), counts)
    holes = numpy.where(out < 0)[0]
    out[holes] = leftovers[: holes.size]

    return out, counts_out


def comb(walkers, comm, weights, target_weight, timer=PopControllerTimer()):
    """Apply the comb method of population control / branching.

    See Booth & Gubernatis PRE 80, 046704 (2009).

    Parameters
    ----------
    comm : MPI communicator
    """
    # Need make a copy to since the elements in psi are only references to
    # walker objects in memory. We don't want future changes in a given
    # element of psi having unintended consequences.
    # todo : add phase to walker for free projection
    timer.start_time()
    if comm.rank == 0:
        parent_ix = numpy.zeros(len(weights), dtype="i")
    else:
        parent_ix = numpy.empty(len(weights), dtype="i")
    if comm.rank == 0:
        total_weight = sum(weights)
        cprobs = numpy.cumsum(weights)
        r = numpy.random.random()
        comb = [(i + r) * (total_weight / target_weight) for i in range(target_weight)]
        iw = 0
        ic = 0
        while ic < len(comb):
            if comb[ic] < cprobs[iw]:
                parent_ix[iw] += 1
                ic += 1
            else:
                iw += 1
        data = {"ix": parent_ix}
    else:
        data = None

    timer.add_non_communication()

    timer.start_time()
    data = comm.bcast(data, root=0)
    timer.add_communication()
    timer.start_time()
    parent_ix = data["ix"]
    # Keep total weight saved for capping purposes.
    # where returns a tuple (array,), selecting first element.
    kill = numpy.where(parent_ix == 0)[0]
    clone = numpy.where(parent_ix > 1)[0]
    reqs = []
    # First initiate non-blocking sends of walkers.
    timer.add_non_communication()
    timer.start_time()
    comm.barrier()
    timer.add_communication()
    for i, (c, k) in enumerate(zip(clone, kill)):
        # Sending from current processor?
        if c // walkers.nwalkers == comm.rank:
            timer.start_time()
            # Location of walker to clone in local list.
            clone_pos = c % walkers.nwalkers
            # copying walker data to intermediate buffer to avoid issues
            # with accessing walker data during send. Might not be
            # necessary.
            dest_proc = k // walkers.nwalkers
            buff = get_buffer(walkers, clone_pos)
            timer.add_non_communication()
            timer.start_time()
            reqs.append(comm.Isend(buff, dest=dest_proc, tag=i))
            timer.add_send_time()
    # Now receive walkers on processors where walkers are to be killed.
    for i, (c, k) in enumerate(zip(clone, kill)):
        # Receiving to current processor?
        if k // walkers.nwalkers == comm.rank:
            timer.start_time()
            # Processor we are receiving from.
            source_proc = c // walkers.nwalkers
            # Location of walker to kill in local list of walkers.
            kill_pos = k % walkers.nwalkers
            timer.add_non_communication()
            timer.start_time()
            comm.Recv(walkers.walker_buffer, source=source_proc, tag=i)
            # with h5py.File('walkers_recv.h5', 'w') as fh5:
            # fh5['walk_{}'.format(k)] = walkers.walker_buffer.copy()
            timer.add_recv_time()
            timer.start_time()
            set_buffer(walkers, kill_pos, walkers.walker_buffer)
            timer.add_non_communication()
            # with h5py.File('after_{}.h5'.format(comm.rank), 'a') as fh5:
            # fh5['walker_{}_{}_{}'.format(c,k,comm.rank)] = walkers.walkers[kill_pos].get_buffer()
    timer.start_time()
    # Complete non-blocking send.
    for rs in reqs:
        rs.wait()
    # Necessary?
    # if len(kill) > 0 or len(clone) > 0:
    # sys.exit()
    comm.Barrier()
    timer.add_communication()
    # Reset walker weight.
    # TODO: check this.
    # for w in walkers.walkers:
    # w.weight = 1.0
    timer.start_time()
    if hasattr(walkers, "fill_combined_weight"):
        walkers.fill_combined_weight(1.0)
    else:
        walkers.weight.fill(1.0)
    timer.add_non_communication()


def pair_branch(walkers, comm, max_weight, min_weight, timer=PopControllerTimer()):
    timer.start_time()
    walker_info_0 = xp.array(xp.abs(walkers.weight))
    timer.add_non_communication()

    timer.start_time()
    glob_inf = None
    glob_inf_0 = None
    glob_inf_1 = None
    glob_inf_2 = None
    glob_inf_3 = None
    if comm.rank == 0:
        glob_inf_0 = numpy.empty([comm.size, walkers.nwalkers], dtype=numpy.float64)
        glob_inf_1 = numpy.empty([comm.size, walkers.nwalkers], dtype=numpy.int64)
        glob_inf_1.fill(1)
        glob_inf_2 = numpy.array(
            [[r for i in range(walkers.nwalkers)] for r in range(comm.size)], dtype=numpy.int64
        )
        glob_inf_3 = numpy.array(
            [[r for i in range(walkers.nwalkers)] for r in range(comm.size)], dtype=numpy.int64
        )

    timer.add_non_communication()

    timer.start_time()
    if hasattr(walker_info_0, "get"):
        walker_info_0 = walker_info_0.get()
    comm.Gather(
        walker_info_0, glob_inf_0, root=0
    )  # gather |w_i| from all processors (comm.size x nwalkers)
    timer.add_communication()

    # Want same random number seed used on all processors
    timer.start_time()
    if comm.rank == 0:
        # Rescale weights.
        glob_inf = numpy.zeros((walkers.nwalkers * comm.size, 4), dtype=numpy.float64)
        glob_inf[:, 0] = glob_inf_0.ravel()  # contains walker |w_i|
        glob_inf[:, 1] = (
            glob_inf_1.ravel()
        )  # all initialized to 1 when it becomes 2 then it will be "branched"
        glob_inf[:, 2] = (
            glob_inf_2.ravel()
        )  # contain processor+walker indices (initial) (i.e., where walkers live)
        glob_inf[:, 3] = (
            glob_inf_3.ravel()
        )  # contain processor+walker indices (final) (i.e., where walkers live)
        sort = numpy.argsort(glob_inf[:, 0], kind="mergesort")
        isort = numpy.argsort(sort, kind="mergesort")
        glob_inf = glob_inf[sort]
        s = 0
        e = len(glob_inf) - 1
        tags = []
        # go through walkers pair-wise
        while s < e:
            if glob_inf[s][0] < min_weight or glob_inf[e][0] > max_weight:
                # sum of paired walker weights
                wab = glob_inf[s][0] + glob_inf[e][0]
                r = numpy.random.rand()
                if r < glob_inf[e][0] / wab:
                    # clone large weight walker
                    glob_inf[e][0] = 0.5 * wab
                    glob_inf[e][1] = 2
                    # Processor we will send duplicated walker to
                    glob_inf[e][3] = glob_inf[s][2]
                    send = glob_inf[s][2]
                    # Kill small weight walker
                    glob_inf[s][0] = 0.0
                    glob_inf[s][1] = 0
                    glob_inf[s][3] = glob_inf[e][2]
                else:
                    # clone small weight walker
                    glob_inf[s][0] = 0.5 * wab
                    glob_inf[s][1] = 2
                    # Processor we will send duplicated walker to
                    glob_inf[s][3] = glob_inf[e][2]
                    send = glob_inf[e][2]
                    # Kill small weight walker
                    glob_inf[e][0] = 0.0
                    glob_inf[e][1] = 0
                    glob_inf[e][3] = glob_inf[s][2]
                tags.append([send])
                s += 1
                e -= 1
            else:
                break
        nw = walkers.nwalkers
        glob_inf = glob_inf[isort].reshape((comm.size, nw, 4))
    else:
        data = None
        glob_inf = None
    timer.add_non_communication()
    timer.start_time()

    data = numpy.empty([walkers.nwalkers, 4], dtype=numpy.float64)
    # 0 = weight, 1 = status (live, branched, die), 2 = initial index, 3 = final index
    comm.Scatter(glob_inf, data, root=0)

    timer.add_communication()
    # Keep total weight saved for capping purposes.
    reqs = []
    for iw, walker in enumerate(data):
        if walker[1] > 1:
            timer.start_time()
            tag = comm.rank * walkers.nwalkers + walker[3]
            if hasattr(walkers, "set_combined_weight"):
                walkers.set_combined_weight(iw, walker[0])
            else:
                walkers.weight[iw] = walker[0]
            buff = get_buffer(walkers, iw)
            timer.add_non_communication()
            timer.start_time()
            reqs.append(comm.Isend(buff, dest=int(round(walker[3])), tag=tag))
            timer.add_send_time()
    for iw, walker in enumerate(data):
        if walker[1] == 0:
            timer.start_time()
            tag = walker[3] * walkers.nwalkers + comm.rank
            timer.add_non_communication()
            timer.start_time()
            comm.Recv(walkers.walker_buffer, source=int(round(walker[3])), tag=tag)
            timer.add_recv_time()
            timer.start_time()
            set_buffer(walkers, iw, walkers.walker_buffer)
            timer.add_non_communication()
    timer.start_time()
    for r in reqs:
        r.wait()
    timer.add_communication()


def stochastic_reconfiguration(
    walkers,
    comm,
    timer=PopControllerTimer(),
    pop_control_counter=0,
    store_walkermap=False,
    read_walkermap=False,
    walkermap_file=None,
):
    timer.start_time()
    nwalkers = walkers.nwalkers
    local_weight = walkers.weight.get() if hasattr(walkers.weight, "get") else walkers.weight
    global_weight = None
    if comm.rank == 0:
        global_weight = numpy.zeros((comm.size, nwalkers), dtype=local_weight.dtype)
    timer.add_non_communication()

    timer.start_time()
    comm.Gather(local_weight, global_weight, root=0)
    timer.add_communication()

    # perform sr on the root
    timer.start_time()
    new_average_weight = None
    if comm.rank == 0:
        cumulative_weights = numpy.cumsum(abs(global_weight))
        total_weight = cumulative_weights[-1]
        new_average_weight = total_weight / nwalkers / comm.size
        if not read_walkermap:
            zeta = numpy.random.rand()
            new_indices = numpy.zeros(comm.size * nwalkers, dtype=numpy.int64)
            for i in range(comm.size * nwalkers):
                z = (i + zeta) / nwalkers / comm.size
                new_indices[i] = numpy.searchsorted(cumulative_weights, z * total_weight)
            reordered_indices, _ = minimize_communication(new_indices)
            if store_walkermap:
                assert walkermap_file is not None, "Must provide filename to store the walker map."
                with h5py.File(walkermap_file, "a") as f:
                    name = f"walker_map_{pop_control_counter}"
                    if name in f:
                        f[name][...] = reordered_indices
                    else:
                        f.create_dataset(name, data=reordered_indices)
        else:
            assert walkermap_file is not None, "Must provide filename to read the walker map."
            with h5py.File(walkermap_file, "r") as f:
                reordered_indices = f[f"walker_map_{pop_control_counter}"][:]
    timer.add_non_communication()

    timer.start_time()
    glob_inf = None
    if comm.rank == 0:
        glob_indices = numpy.arange(comm.size * nwalkers, dtype=numpy.int64)
        mask = reordered_indices != glob_indices
        sendidx = reordered_indices[mask]
        destidx = glob_indices[mask]
        glob_inf = numpy.column_stack((sendidx, destidx))

    timer.add_non_communication()
    timer.start_time()
    glob_inf = comm.bcast(glob_inf, root=0)
    new_average_weight = comm.bcast(new_average_weight, root=0)
    timer.add_communication()

    timer.start_time()
    local_sends = [glob_inf[(glob_inf[:, 0] // nwalkers == i)] for i in range(comm.size)]
    local_recvs = [glob_inf[(glob_inf[:, 1] // nwalkers == i)] for i in range(comm.size)]
    num_local_sends = numpy.array([len(s) for s in local_sends])
    cumsum_local_sends = numpy.cumsum(num_local_sends) - num_local_sends
    num_local_recvs = numpy.array([len(r) for r in local_recvs])
    cumsum_local_recvs = numpy.cumsum(num_local_recvs) - num_local_recvs

    buflis = {}
    local_send = local_sends[comm.rank]
    local_send_loc_idx = local_send[:, 0] % nwalkers
    local_recv = local_recvs[comm.rank]
    for i in range(nwalkers):
        if i in local_send_loc_idx:
            buflis[i] = get_buffer(walkers, i)
    timer.add_non_communication()
    comm.barrier()
    send_reqs = []
    for isend, (src_idx, dest_idx) in enumerate(local_send):
        src_loc = src_idx % nwalkers
        dest_rk = dest_idx // nwalkers
        tag = isend + cumsum_local_sends[comm.rank]

        buf = buflis[src_loc]
        req = comm.Issend(buf, dest=int(dest_rk), tag=int(tag))
        send_reqs.append(req)

    # Post all nonblocking recvs, saving a Status for each to inspect later
    walker_len = get_buffer(walkers, 0).shape[0]
    recv_reqs = []
    for irecv, (src_idx, dest_idx) in enumerate(local_recv):
        iw = dest_idx % nwalkers
        src_rank = src_idx // nwalkers
        tag_recv = irecv + cumsum_local_recvs[comm.rank]

        recv_buf = numpy.empty(walker_len, dtype=numpy.complex128)
        status = MPI.Status()
        req = comm.Irecv(recv_buf, source=int(src_rank), tag=int(tag_recv))
        recv_reqs.append((iw, recv_buf, status, req))

    debug_copy = os.environ.get("IPIE_SR_COPY_DEBUG", "0") == "1"
    copy_checks = []

    # Wait on recvs and inspect their Status
    for iw, buf, status, req in recv_reqs:
        req.Wait(status)
        set_buffer(walkers, iw, buf)
        if debug_copy and len(copy_checks) < 3:
            repacked = get_buffer(walkers, iw)
            if hasattr(status, "Get_source"):
                src_rank = int(status.Get_source())
            else:
                src_rank = int(getattr(status, "source", comm.rank))
            copy_checks.append(
                (
                    int(iw),
                    src_rank,
                    float(numpy.linalg.norm(buf)),
                    float(numpy.linalg.norm(repacked - buf)),
                )
            )

    # 4) Wait on sends
    MPI.Request.Waitall(send_reqs)

    comm.Barrier()

    timer.start_time()
    walkers.weight[:] = new_average_weight
    if hasattr(walkers, "walkers_A") and hasattr(walkers, "walkers_B"):
        walkers.walkers_A.weight[:] = new_average_weight
        walkers.walkers_B.weight[:] = new_average_weight
    timer.add_non_communication()


def stochastic_reconfiguration_independent_repairing(
    walkers,
    comm,
    timer=PopControllerTimer(),
    pop_control_counter=0,
    store_walkermap=False,
    read_walkermap=False,
    walkermap_file=None,
):
    """
    Independent population control on walkers_A and walkers_B, followed by
    post-SR rematching based on the squared norm of the difference between
    the last-block auxiliary fields.

    Assumptions:
      - walkers has walkers_A and walkers_B
      - walkers.auxfield stores the last sampled shared auxiliary field
      - pre-SR original correlated pairs share the same global index
      - get_buffer/set_buffer move the main walker state
      - auxfield is not necessarily included in get_buffer/set_buffer, so it is
        explicitly reordered at the end from the saved pre-SR auxfield arrays
    """
    assert hasattr(walkers, "walkers_A") and hasattr(walkers, "walkers_B")

    timer.start_time()
    nwalkers = walkers.nwalkers
    ntot = comm.size * nwalkers

    # Local weights
    local_weight_A = (
        walkers.walkers_A.weight.get()
        if hasattr(walkers.walkers_A.weight, "get")
        else walkers.walkers_A.weight
    )
    local_weight_B = (
        walkers.walkers_B.weight.get()
        if hasattr(walkers.walkers_B.weight, "get")
        else walkers.walkers_B.weight
    )

    # Save the pre-SR shared auxfields; these define the rematching metric.
    if hasattr(walkers, "auxfield") and walkers.auxfield is not None:
        local_aux = walkers.auxfield.get() if hasattr(walkers.auxfield, "get") else walkers.auxfield
    elif hasattr(walkers.walkers_A, "auxfield"):
        local_aux = (
            walkers.walkers_A.auxfield.get()
            if hasattr(walkers.walkers_A.auxfield, "get")
            else walkers.walkers_A.auxfield
        )
    else:
        raise AttributeError(
            "stochastic_reconfiguration_independent_repairing requires walkers.auxfield "
            "or walkers.walkers_A.auxfield to be defined."
        )

    local_aux = numpy.asarray(local_aux)
    aux_shape = local_aux.shape[1:]
    local_aux_flat = local_aux.reshape(nwalkers, -1)

    # Gather weights and pre-SR auxfields to root.
    global_weight_A = None
    global_weight_B = None
    global_aux = None
    if comm.rank == 0:
        global_weight_A = numpy.zeros((comm.size, nwalkers), dtype=local_weight_A.dtype)
        global_weight_B = numpy.zeros((comm.size, nwalkers), dtype=local_weight_B.dtype)
        global_aux = numpy.zeros((comm.size, nwalkers, local_aux_flat.shape[1]), dtype=local_aux_flat.dtype)
    timer.add_non_communication()

    timer.start_time()
    comm.Gather(local_weight_A, global_weight_A, root=0)
    comm.Gather(local_weight_B, global_weight_B, root=0)
    comm.Gather(local_aux_flat, global_aux, root=0)
    timer.add_communication()

    # Root computes final source maps:
    #   reordered_indices_A[dest] = pre-SR source for final A dest slot
    #   reordered_indices_B[dest] = pre-SR source for final B dest slot
    timer.start_time()
    reordered_indices_A = None
    reordered_indices_B = None
    new_average_weight_A = None
    new_average_weight_B = None

    if comm.rank == 0:
        flat_abs_A = numpy.abs(global_weight_A).ravel()
        flat_abs_B = numpy.abs(global_weight_B).ravel()
        pre_aux_flat = global_aux.reshape(ntot, -1)

        total_weight_A = flat_abs_A.sum()
        total_weight_B = flat_abs_B.sum()
        new_average_weight_A = total_weight_A / ntot
        new_average_weight_B = total_weight_B / ntot

        if not read_walkermap:
            # Independent SR on A
            cumulative_weights_A = numpy.cumsum(flat_abs_A)
            zeta = numpy.random.rand()
            new_indices_A = numpy.zeros(ntot, dtype=numpy.int64)
            for i in range(ntot):
                z = (i + zeta) / ntot
                new_indices_A[i] = numpy.searchsorted(cumulative_weights_A, z * total_weight_A)
            reordered_indices_A, _ = minimize_communication(new_indices_A)

            # Independent SR on B (raw map before repair)
            cumulative_weights_B = numpy.cumsum(flat_abs_B)
            new_indices_B = numpy.zeros(ntot, dtype=numpy.int64)
            for i in range(ntot):
                z = (i + zeta) / ntot
                new_indices_B[i] = numpy.searchsorted(cumulative_weights_B, z * total_weight_B)
            reordered_indices_B_raw, _ = minimize_communication(new_indices_B)

            # Auxfields inherited by the post-SR populations.
            post_aux_A = pre_aux_flat[reordered_indices_A]
            post_aux_B = pre_aux_flat[reordered_indices_B_raw]

            # Stage 1: preserve same-parent survivors first.
            children_A = [[] for _ in range(ntot)]
            children_B = [[] for _ in range(ntot)]

            for gA, pA in enumerate(reordered_indices_A):
                children_A[int(pA)].append(gA)
            for gB, pB in enumerate(reordered_indices_B_raw):
                children_B[int(pB)].append(gB)

            # repair_map_B[gA] = current post-SR B slot gB that should pair with final A slot gA
            repair_map_B = -numpy.ones(ntot, dtype=numpy.int64)
            used_B = numpy.zeros(ntot, dtype=bool)

            for parent in range(ntot):
                listA = children_A[parent]
                listB = children_B[parent]
                m = min(len(listA), len(listB))
                for t in range(m):
                    gA = listA[t]
                    gB = listB[t]
                    repair_map_B[gA] = gB
                    used_B[gB] = True

            # Stage 2: match leftovers by ||auxA - auxB||^2
            leftover_A = numpy.where(repair_map_B < 0)[0]
            leftover_B = numpy.where(~used_B)[0]

            if leftover_A.size > 0:
                XA = post_aux_A[leftover_A]
                XB = post_aux_B[leftover_B]

                diff = XA[:, None, :] - XB[None, :, :]
                if numpy.iscomplexobj(diff):
                    cost = numpy.sum(numpy.abs(diff) ** 2, axis=2).real
                else:
                    cost = numpy.sum(diff * diff, axis=2, dtype=numpy.float64)

                # Try Hungarian. Fall back to greedy if scipy is unavailable.
                try:
                    from scipy.optimize import linear_sum_assignment

                    row_ind, col_ind = linear_sum_assignment(cost)
                except Exception:
                    m = cost.shape[0]
                    used_cols = numpy.zeros(m, dtype=bool)
                    row_ind = []
                    col_ind = []
                    for i in range(m):
                        row_cost = cost[i].copy()
                        row_cost[used_cols] = numpy.inf
                        j = int(numpy.argmin(row_cost))
                        used_cols[j] = True
                        row_ind.append(i)
                        col_ind.append(j)
                    row_ind = numpy.asarray(row_ind, dtype=numpy.int64)
                    col_ind = numpy.asarray(col_ind, dtype=numpy.int64)

                for r, c in zip(row_ind, col_ind):
                    gA = int(leftover_A[r])
                    gB = int(leftover_B[c])
                    repair_map_B[gA] = gB
                    used_B[gB] = True

            assert numpy.all(repair_map_B >= 0)
            assert numpy.unique(repair_map_B).size == ntot

            # Compose the repair directly into the final B source map.
            reordered_indices_B = reordered_indices_B_raw[repair_map_B]

            if store_walkermap:
                assert walkermap_file is not None, "Must provide filename to store the walker maps."
                with h5py.File(walkermap_file, "a") as f:
                    nameA = f"walker_map_A_{pop_control_counter}"
                    nameB = f"walker_map_B_{pop_control_counter}"
                    if nameA in f:
                        f[nameA][...] = reordered_indices_A
                    else:
                        f.create_dataset(nameA, data=reordered_indices_A)
                    if nameB in f:
                        f[nameB][...] = reordered_indices_B
                    else:
                        f.create_dataset(nameB, data=reordered_indices_B)
        else:
            assert walkermap_file is not None, "Must provide filename to read the walker maps."
            with h5py.File(walkermap_file, "r") as f:
                reordered_indices_A = f[f"walker_map_A_{pop_control_counter}"][:]
                reordered_indices_B = f[f"walker_map_B_{pop_control_counter}"][:]

    timer.add_non_communication()

    # Broadcast final source maps and new weights.
    timer.start_time()
    reordered_indices_A = comm.bcast(reordered_indices_A, root=0)
    reordered_indices_B = comm.bcast(reordered_indices_B, root=0)
    new_average_weight_A = comm.bcast(new_average_weight_A, root=0)
    new_average_weight_B = comm.bcast(new_average_weight_B, root=0)
    timer.add_communication()

    # -------------------------------------------------------------------------
    # Move walkers_A according to reordered_indices_A
    # -------------------------------------------------------------------------
    timer.start_time()
    glob_inf_A = None
    if comm.rank == 0:
        glob_indices = numpy.arange(ntot, dtype=numpy.int64)
        mask = reordered_indices_A != glob_indices
        sendidx = reordered_indices_A[mask]
        destidx = glob_indices[mask]
        tags = numpy.arange(sendidx.size, dtype=numpy.int64)
        glob_inf_A = numpy.column_stack((sendidx, destidx, tags))
    timer.add_non_communication()

    timer.start_time()
    glob_inf_A = comm.bcast(glob_inf_A, root=0)
    timer.add_communication()

    timer.start_time()
    local_sends_A = [glob_inf_A[(glob_inf_A[:, 0] // nwalkers == i)] for i in range(comm.size)]
    local_recvs_A = [glob_inf_A[(glob_inf_A[:, 1] // nwalkers == i)] for i in range(comm.size)]

    buflis_A = {}
    local_send_A = local_sends_A[comm.rank]
    local_send_loc_idx_A = local_send_A[:, 0] % nwalkers if len(local_send_A) > 0 else numpy.array([], dtype=numpy.int64)
    local_recv_A = local_recvs_A[comm.rank]
    for i in range(nwalkers):
        if i in local_send_loc_idx_A:
            buflis_A[i] = get_buffer(walkers.walkers_A, i)
    timer.add_non_communication()

    comm.barrier()
    send_reqs_A = []
    for src_idx, dest_idx, tag in local_send_A:
        src_loc = src_idx % nwalkers
        dest_rk = dest_idx // nwalkers
        buf = buflis_A[src_loc]
        req = comm.Issend(buf, dest=int(dest_rk), tag=int(tag))
        send_reqs_A.append(req)

    walker_len_A = get_buffer(walkers.walkers_A, 0).shape[0]
    recv_reqs_A = []
    for src_idx, dest_idx, tag_recv in local_recv_A:
        iw = dest_idx % nwalkers
        src_rank = src_idx // nwalkers

        recv_buf = numpy.empty(walker_len_A, dtype=numpy.complex128)
        status = MPI.Status()
        req = comm.Irecv(recv_buf, source=int(src_rank), tag=int(tag_recv))
        recv_reqs_A.append((iw, recv_buf, status, req))

    for iw, buf, status, req in recv_reqs_A:
        req.Wait(status)
        set_buffer(walkers.walkers_A, iw, buf)

    MPI.Request.Waitall(send_reqs_A)
    comm.Barrier()

    # -------------------------------------------------------------------------
    # Move walkers_B according to reordered_indices_B (already composed with repair)
    # -------------------------------------------------------------------------
    timer.start_time()
    glob_inf_B = None
    if comm.rank == 0:
        glob_indices = numpy.arange(ntot, dtype=numpy.int64)
        mask = reordered_indices_B != glob_indices
        sendidx = reordered_indices_B[mask]
        destidx = glob_indices[mask]
        tags = numpy.arange(sendidx.size, dtype=numpy.int64)
        glob_inf_B = numpy.column_stack((sendidx, destidx, tags))
    timer.add_non_communication()

    timer.start_time()
    glob_inf_B = comm.bcast(glob_inf_B, root=0)
    timer.add_communication()

    timer.start_time()
    local_sends_B = [glob_inf_B[(glob_inf_B[:, 0] // nwalkers == i)] for i in range(comm.size)]
    local_recvs_B = [glob_inf_B[(glob_inf_B[:, 1] // nwalkers == i)] for i in range(comm.size)]

    buflis_B = {}
    local_send_B = local_sends_B[comm.rank]
    local_send_loc_idx_B = local_send_B[:, 0] % nwalkers if len(local_send_B) > 0 else numpy.array([], dtype=numpy.int64)
    local_recv_B = local_recvs_B[comm.rank]
    for i in range(nwalkers):
        if i in local_send_loc_idx_B:
            buflis_B[i] = get_buffer(walkers.walkers_B, i)
    timer.add_non_communication()

    comm.barrier()
    send_reqs_B = []
    for src_idx, dest_idx, tag in local_send_B:
        src_loc = src_idx % nwalkers
        dest_rk = dest_idx // nwalkers
        buf = buflis_B[src_loc]
        req = comm.Issend(buf, dest=int(dest_rk), tag=int(tag))
        send_reqs_B.append(req)

    walker_len_B = get_buffer(walkers.walkers_B, 0).shape[0]
    recv_reqs_B = []
    for src_idx, dest_idx, tag_recv in local_recv_B:
        iw = dest_idx % nwalkers
        src_rank = src_idx // nwalkers

        recv_buf = numpy.empty(walker_len_B, dtype=numpy.complex128)
        status = MPI.Status()
        req = comm.Irecv(recv_buf, source=int(src_rank), tag=int(tag_recv))
        recv_reqs_B.append((iw, recv_buf, status, req))

    for iw, buf, status, req in recv_reqs_B:
        req.Wait(status)
        set_buffer(walkers.walkers_B, iw, buf)

    MPI.Request.Waitall(send_reqs_B)
    comm.Barrier()

    # -------------------------------------------------------------------------
    # Reset weights
    # -------------------------------------------------------------------------
    timer.start_time()
    walkers.walkers_A.weight[:] = new_average_weight_A
    walkers.walkers_B.weight[:] = new_average_weight_B
    if hasattr(walkers, "weight"):
        # Pair-level weight convention is not unique once A/B SR are independent.
        walkers.weight[:] = 0.5 * (new_average_weight_A + new_average_weight_B)
    timer.add_non_communication()

    # -------------------------------------------------------------------------
    # Reorder auxfield explicitly so it stays aligned with the moved walkers.
    # -------------------------------------------------------------------------
    timer.start_time()
    if comm.rank == 0:
        pre_aux_flat = global_aux.reshape(ntot, -1)

        final_aux_A = pre_aux_flat[reordered_indices_A].reshape((comm.size, nwalkers) + aux_shape)
        final_aux_B = pre_aux_flat[reordered_indices_B].reshape((comm.size, nwalkers) + aux_shape)
    else:
        final_aux_A = None
        final_aux_B = None

    recv_aux_A = numpy.empty_like(local_aux)
    recv_aux_B = numpy.empty_like(local_aux)
    comm.Scatter(final_aux_A, recv_aux_A, root=0)
    comm.Scatter(final_aux_B, recv_aux_B, root=0)

    if hasattr(walkers.walkers_A, "auxfield") and hasattr(walkers.walkers_A.auxfield, "set"):
        walkers.walkers_A.auxfield.set(recv_aux_A)
    else:
        walkers.walkers_A.auxfield = recv_aux_A.copy()

    if hasattr(walkers.walkers_B, "auxfield") and hasattr(walkers.walkers_B.auxfield, "set"):
        walkers.walkers_B.auxfield.set(recv_aux_B)
    else:
        walkers.walkers_B.auxfield = recv_aux_B.copy()

    if hasattr(walkers, "auxfield"):
        if hasattr(walkers.auxfield, "set"):
            walkers.auxfield.set(recv_aux_A)
        else:
            walkers.auxfield = recv_aux_A.copy()

    if hasattr(walkers, "sync_combined_state"):
        walkers.sync_combined_state()
    timer.add_communication()
