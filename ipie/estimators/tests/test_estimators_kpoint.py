import tempfile

import pytest

from ipie.estimators.energy import EnergyEstimator
from ipie.estimators.handler import EstimatorHandler
from ipie.utils.testing import gen_random_test_instances_kpt

# @pytest.mark.unit
def test_energy_estimator():
    nmo = 10
    nocc = 8
    naux = 30
    nk = 8
    nwalker = 10
    system, ham, walker_batch, trial = gen_random_test_instances_kpt(nk, nmo, nocc, naux, nwalker)
    estim = EnergyEstimator(system=system, ham=ham, trial=trial)
    estim.compute_estimator(system, walker_batch, ham, trial)
    assert len(estim.names) == 5
    # assert estim["ENumer"].real == pytest.approx(-754.0373585215561)
    print(estim["ENumer"].real)
    assert estim["ETotal"] == pytest.approx(0.0)
    tmp = estim.data.copy()
    estim.post_reduce_hook(tmp)
    # assert tmp[estim.get_index("ETotal")] == pytest.approx(-75.40373585215562)
    assert estim.print_to_stdout
    assert estim.ascii_filename == None
    assert estim.shape == (5,)
    header = estim.header_to_text
    data_to_text = estim.data_to_text(tmp)
    assert len(data_to_text.split()) == 5

# TODO: Hartree Fock test for local energy of silicon

# @pytest.mark.unit
def test_estimator_handler():
    with tempfile.NamedTemporaryFile() as tmp1, tempfile.NamedTemporaryFile() as tmp2:
        nmo = 10
        nk = 8
        nocc = 8
        naux = 30
        nwalker = 10
        system, ham, walker_batch, trial = gen_random_test_instances_kpt(nk, nmo, nocc, naux, nwalker)
        estim = EnergyEstimator(system=system, ham=ham, trial=trial, filename=tmp1.name)
        estim.print_to_stdout = False
        from ipie.config import MPI

        comm = MPI.COMM_WORLD
        handler = EstimatorHandler(
            comm,
            system,
            ham,
            trial,
            block_size=10,
            observables=("energy",),
            filename=tmp2.name,
        )
        handler["energy1"] = estim
        handler.json_string = ""
        handler.initialize(comm)
        handler.compute_estimators(comm, system, ham, trial, walker_batch)
        handler.compute_estimators(comm, system, ham, trial, walker_batch)


if __name__ == "__main__":
    test_energy_estimator()
    test_estimator_handler()
