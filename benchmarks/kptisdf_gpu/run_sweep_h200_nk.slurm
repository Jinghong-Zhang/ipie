#!/bin/bash
#SBATCH -J kptamd_sweep_h200_nk
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

# nk-dominated regime up to nk=125 (5x5x5), occ ratios 0.1/0.2/0.4
$PY bench_vhs_optimizations.py --mode both --repeats 2 --warmups 1 \
    --case 125,100,5,4 --case 125,100,10,4 --case 125,100,20,4 \
    --case 64,312,15,4 --case 64,312,31,4 \
    --case 125,160,8,4 --case 125,160,16,4
$PY bench_vhs_optimizations.py --mode both --repeats 1 --warmups 1 \
    --case 64,312,62,4 \
    --case 125,160,32,4
