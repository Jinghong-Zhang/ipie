#!/bin/bash
#SBATCH -J kptamd_sweep_h200_nbsf
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

# Nbsf-dominated regime up to Nbsf=2000 (Nisdf=36000), occ ratios 0.1/0.2/0.4
$PY bench_vhs_optimizations.py --mode both --repeats 2 --warmups 1 \
    --case 8,1000,50,4 --case 8,1000,100,4 \
    --case 27,740,37,4 --case 27,740,74,4 \
    --case 8,2000,100,4
$PY bench_vhs_optimizations.py --mode both --repeats 1 --warmups 1 \
    --case 8,1000,200,4 \
    --case 8,2000,200,4 --case 8,2000,400,4 \
    --case 27,740,148,4
