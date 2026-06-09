#!/bin/bash
#SBATCH -J kptamd_bench_a100
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH --account=joonholee_lab
#SBATCH -o /n/home10/jhzhang/code/ipie_kptafqmc_amd_commit/benchmarks/kptisdf_gpu/results/%x-%j.out

set -x
module load cuda/12.4.1-fasrc01
export PYTHONPATH=/n/home10/jhzhang/code/ipie_kptafqmc_amd_commit:$PYTHONPATH
export IPIE_USE_GPU=1
PY=/n/home10/jhzhang/.conda/envs/ipierun/bin/python
cd /n/home10/jhzhang/code/ipie_kptafqmc_amd_commit/benchmarks/kptisdf_gpu

nvidia-smi --query-gpu=name,memory.total --format=csv

# isolated kernel + taylor benchmarks, medium sizes (A100 80GB)
$PY bench_vhs_optimizations.py --mode both --repeats 3 --warmups 1 \
    --case 8,100,9,32
$PY bench_vhs_optimizations.py --mode both --repeats 2 --warmups 1 \
    --case 8,150,14,32

# fake-Hamiltonian AFQMC timing run (Nisdf = 18 * Nbasis), new vs old
$PY fake_kptisdf_afqmc.py --kmesh 2,2,2 --nbasis 100 --nocc 9 --nwalkers 32 \
    --nblocks 3 --nsteps 5 --propagation new
$PY fake_kptisdf_afqmc.py --kmesh 2,2,2 --nbasis 100 --nocc 9 --nwalkers 32 \
    --nblocks 3 --nsteps 5 --propagation old
