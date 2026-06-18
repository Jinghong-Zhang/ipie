"""Run AFQMC on one LNO fragment exported by qcpbc (postscf_lnoeri, ao_isdf).

Reads hpq_<iF>_<thr>.h5 / Lpq_<iF>_<thr>.h5 from the qcpbc scratch dir, runs AFQMC
in the active LNO basis, and accumulates the fragment (pinned-orbital) correlation
energy via the LNO kernels in ipie/estimators/local_energy_sd.py.

Usage:
    python run_lno_afqmc.py <scratch_dir> <iF> <thr_idx> <n_frag>

n_frag = number of fragment occupied LNOs (frag_def[iF].n_elem). qcpbc's ao_isdf
Lpq_meta does not store it; pass it explicitly (or add it to lpq_meta in the C++).
"""

import sys

import h5py
import numpy

from ipie.config import MPI
from ipie.estimators.estimator_base import EstimatorBase
from ipie.estimators.local_energy_sd import (
    ecorrcoul_lno_real_rchol_uhf,
    ecorrxx_lno_real_rchol,
)
from ipie.hamiltonians.generic import Generic as HamGeneric
from ipie.qmc.afqmc import AFQMC
from ipie.trial_wavefunction.single_det import SingleDet
from ipie.utils.backend import arraylib as xp


def _frag_2body(trial, walkers, n_frag):
    """Per-walker fragment (pinned) two-body energy: sum_{I in frag} (ecoul - exx)."""
    naux = trial._rchola.shape[0]
    nocc = walkers.Ghalfa.shape[1]
    nbasis = walkers.Ghalfa.shape[2]
    rchola = trial._rchola.reshape(naux, nocc, nbasis)
    rcholb = trial._rcholb.reshape(naux, nocc, nbasis)
    Ga = walkers.Ghalfa  # [nw, nocc, nbasis], complex
    Gb = walkers.Ghalfb
    pin = slice(0, n_frag)
    ecoul = ecorrcoul_lno_real_rchol_uhf(
        rchola[:, pin], rchola, Ga[:, pin], Ga, Gb[:, pin], Gb
    )
    exx = ecorrxx_lno_real_rchol(rchola[:, pin], rchola, Ga[:, pin], Ga)
    exx += ecorrxx_lno_real_rchol(rcholb[:, pin], rcholb, Gb[:, pin], Gb)
    return (ecoul - exx).sum(axis=1)  # [nw]


class LNOFragmentEnergy(EstimatorBase):
    """Mixed-estimator fragment 2-body energy from the pinned (fragment) orbitals."""

    def __init__(self, n_frag, e_ref):
        super().__init__()
        self.n_frag = n_frag
        self.e_ref = e_ref  # fragment 2-body at the trial (HF) -> subtract for E_corr
        self.scalar_estimator = True
        self._data = {"ENumer": 0.0j, "EDenom": 0.0j, "EFragCorr": 0.0j}
        self._shape = (len(self.names),)
        self._data_index = {k: i for i, k in enumerate(self._data)}
        self.print_to_stdout = True
        self.ascii_filename = None

    def compute_estimator(self, system=None, walkers=None, hamiltonian=None, trial=None):
        trial.calc_greens_function(walkers)
        efrag = _frag_2body(trial, walkers, self.n_frag)
        self._data["ENumer"] = xp.sum(walkers.weight * efrag.real)
        self._data["EDenom"] = xp.sum(walkers.weight)
        return self.data

    def post_reduce_hook(self, data):
        i = self._data_index
        # ponytail: E_corr_frag = <E2_frag>_AFQMC - <E2_frag>_HF. Subtracting the trial
        # reference is the LNO correlation convention; verify against your MP2 numbers.
        data[i["EFragCorr"]] = data[i["ENumer"]] / data[i["EDenom"]] - self.e_ref


def load_fragment(scratch, iF, thr):
    tag = f"{iF}_{thr}"
    with h5py.File(f"{scratch}/Lpq_{tag}.h5", "r") as f:
        L = numpy.array(f["L_pq"])          # arma -> h5py reads transposed
        meta = numpy.array(f["Lpq_meta"]).ravel()
    with h5py.File(f"{scratch}/hpq_{tag}.h5", "r") as f:
        h = numpy.array(f["h_pq"])
    nchol, nmo, nocc, nvir = int(meta[0]), int(meta[1]), int(meta[2]), int(meta[3])
    if L.shape[0] == nchol:                  # orient to (nmo^2, nchol)
        L = L.T.copy()
    if h.shape != (nmo, nmo):
        h = h.T.copy()
    assert L.shape == (nmo * nmo, nchol), L.shape
    return h, L, nmo, nocc, nchol


def build_afqmc(scratch, iF, thr, n_frag, comm, num_walkers=100, num_blocks=100, seed=7):
    h, L, nmo, nocc, nchol = load_fragment(scratch, iF, thr)
    nelec = [nocc, nocc]  # closed-shell LNO reference

    ham = HamGeneric(numpy.array([h, h]), L, 0.0)  # e0 = 0: correlation only

    # Reference determinant = first nocc LNOs (h_pq ordered [occ | vir]).
    psi = numpy.zeros((nmo, 2 * nocc))
    psi[:nocc, :nocc] = numpy.eye(nocc)
    psi[:nocc, nocc:] = numpy.eye(nocc)
    trial = SingleDet(psi, nelec, nmo)
    trial.build()
    trial.half_rotate(ham)

    # HF reference value of the fragment 2-body energy (subtracted to get correlation).
    e_ref = float(_frag_2body(trial, _TrialWalkers(trial), n_frag)[0].real)

    afqmc = AFQMC.build(
        nelec, ham, trial, num_walkers=num_walkers, num_blocks=num_blocks, seed=seed
    )
    return afqmc, LNOFragmentEnergy(n_frag, e_ref)


class _TrialWalkers:
    """Single 'walker' holding the trial Green's function (for the HF reference)."""

    def __init__(self, trial):
        self.Ghalfa = trial.Ghalf[0][None].astype(numpy.complex128)
        self.Ghalfb = trial.Ghalf[1][None].astype(numpy.complex128)
        self.weight = numpy.ones(1)


if __name__ == "__main__":
    # run_lno_afqmc.py <scratch> <iF> <thr> <n_frag> [num_walkers] [num_blocks]
    scratch, iF, thr, n_frag = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    nw = int(sys.argv[5]) if len(sys.argv) > 5 else 100
    nb = int(sys.argv[6]) if len(sys.argv) > 6 else 100
    comm = MPI.COMM_WORLD
    afqmc, lno_est = build_afqmc(scratch, iF, thr, n_frag, comm, num_walkers=nw, num_blocks=nb)
    afqmc.run(additional_estimators={"lno": lno_est})
    afqmc.finalise(verbose=True)
    # NOTE: ETotal is not physical (e0=0, no core term). Read EFragCorr from the
    # "lno" observable; sum over fragments weighted by frag_wts for the cell total.
