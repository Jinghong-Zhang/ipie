#!/bin/bash
#SBATCH -J kptamd_bench_h200
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH --account=joonholee_lab
#SBATCH -o /n/home10/jhzhang/code/ipie_kptafqmc_amd_commit/benchmarks/kptisdf_gpu/results/%x-%j.out

set -x
module load cuda/12.4.1-fasrc01
export PYTHONPATH=/n/home10/jhzhang/code/ipie_kptafqmc_amd_commit:$PYTHONPATH
export IPIE_USE_GPU=1
PY=/n/home10/jhzhang/.conda/envs/ipierun/bin/python
cd /n/home10/jhzhang/code/ipie_kptafqmc_amd_commit/benchmarks/kptisdf_gpu

nvidia-smi --query-gpu=name,memory.total --format=csv

# large isolated benchmarks that need H200 memory/compute
$PY bench_vhs_optimizations.py --mode both --repeats 2 --warmups 1 \
    --case 27,100,13,16
$PY bench_vhs_optimizations.py --mode kernel --repeats 2 --warmups 1 \
    --case 27,100,13,64

# large fake-Hamiltonian AFQMC timing runs, new vs old
$PY fake_kptisdf_afqmc.py --kmesh 3,3,3 --nbasis 60 --nocc 5 --nwalkers 16 \
    --nblocks 3 --nsteps 5 --propagation new
$PY fake_kptisdf_afqmc.py --kmesh 3,3,3 --nbasis 60 --nocc 5 --nwalkers 16 \
    --nblocks 3 --nsteps 5 --propagation old
$PY fake_kptisdf_afqmc.py --kmesh 2,2,2 --nbasis 150 --nocc 14 --nwalkers 32 \
    --nblocks 3 --nsteps 5 --propagation new
$PY fake_kptisdf_afqmc.py --kmesh 2,2,2 --nbasis 150 --nocc 14 --nwalkers 32 \
    --nblocks 3 --nsteps 5 --propagation old
