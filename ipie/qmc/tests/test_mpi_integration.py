import numpy as np
import json
import os
from mpi4py import MPI
import pytest

from ipie.analysis.extraction import extract_test_data_hdf5
from ipie.qmc.calc import (
        read_input,
        get_driver
        )

try:
    import pytest_mpi
    have_pytest_mpi = True
except ImportError:
    have_pytest_mpi = False


comm = MPI.COMM_WORLD
serial_test = comm.size == 1

_data_dir  = os.path.abspath(os.path.dirname(__file__)) + '/reference_data/'
# glob is a bit dangerous.
# _test_dirs = [d for d in glob.glob(_data_dir+'/*') if os.path.isdir(d)]
_test_dirs = [
        "4x4_hubbard_discrete",
        "ft_4x4_hubbard_discrete",
        "ft_ueg_ecut1.0_rs1.0",
        "ueg_ecut2.5_rs2.0_ne14",
        "h10_cc-pvtz_batched",
        "h10_cc-pvtz_pair_branch",
        "neon_cc-pvdz_rhf",
        "benzene_cc-pvdz_batched", # disabling
        "benzene_cc-pvdz_chunked"
        ]
_tests     = [(_data_dir+d+'/input.json',_data_dir+d+'/reference.json') for d in _test_dirs]

def compare_test_data(ref, test):
    for k, v in ref.items():
        if k == 'sys_info':
            continue
        print(k, np.array(ref[k]) - np.array(test[k]))
        assert np.max(np.abs(np.array(ref[k]) - np.array(test[k]))) < 1e-10

def run_test_system(input_file, benchmark_file):
    comm = MPI.COMM_WORLD
    input_dict = read_input(input_file, comm)
    if input_dict['system'].get('integrals') is not None:
        input_dict['system']['integrals'] = input_file[:-10] + 'afqmc.h5'
        input_dict['trial']['filename'] = input_file[:-10] + 'afqmc.h5'
    elif input_dict['hamiltonian'].get('integrals') is not None:
        input_dict['hamiltonian']['integrals'] = input_file[:-10] + 'afqmc.h5'
        input_dict['trial']['filename'] = input_file[:-10] + 'afqmc.h5'
    afqmc = get_driver(input_dict, comm)
    afqmc.run(comm=comm)
    if comm.rank == 0:
        test_data = extract_test_data_hdf5('estimates.0.h5')
        with open(benchmark_file, 'r') as f:
            ref_data = json.load(f)
        compare_test_data(ref_data, test_data)
    comm.barrier()

@pytest.mark.mpi
@pytest.mark.skipif(serial_test, reason="Test should be run on multiple cores.")
@pytest.mark.skipif(not have_pytest_mpi, reason="Test requires pytest-mpi plugin.")
@pytest.mark.parametrize("input_dir, benchmark_dir", _tests)
def test_system_mpi(input_dir, benchmark_dir):
    run_test_system(input_dir, benchmark_dir)

def teardown_module():
    cwd = os.getcwd()
    files = ['estimates.0.h5']
    for f in files:
        try:
            os.remove(cwd+'/'+f)
        except OSError:
            pass
