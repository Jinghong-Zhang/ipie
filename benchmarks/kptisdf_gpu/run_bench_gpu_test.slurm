#!/bin/bash
#SBATCH -J kptamd_bench_test
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH --account=joonholee_lab
#SBATCH -o /n/home10/jhzhang/code/ipie_kptafqmc_amd_commit/benchmarks/kptisdf_gpu/results/%x-%j.out

set -x
module load cuda/12.4.1-fasrc01
export PYTHONPATH=/n/home10/jhzhang/code/ipie_kptafqmc_amd_commit:$PYTHONPATH
export IPIE_USE_GPU=1
PY=/n/home10/jhzhang/.conda/envs/ipierun/bin/python
cd /n/home10/jhzhang/code/ipie_kptafqmc_amd_commit/benchmarks/kptisdf_gpu

nvidia-smi --query-gpu=name,memory.total --format=csv

# small isolated kernel benchmark (fits 20 GB MIG slice)
$PY bench_vhs_optimizations.py --mode both --repeats 3 --warmups 1 \
    --case 8,50,7,16

# smoke-test the fake AFQMC end to end at tiny size, both propagation modes
$PY fake_kptisdf_afqmc.py --kmesh 2,2,2 --nbasis 40 --nocc 3 --nwalkers 8 \
    --nblocks 3 --nsteps 5 --propagation new
$PY fake_kptisdf_afqmc.py --kmesh 2,2,2 --nbasis 40 --nocc 3 --nwalkers 8 \
    --nblocks 3 --nsteps 5 --propagation old
