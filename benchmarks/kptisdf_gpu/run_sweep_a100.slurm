#!/bin/bash
#SBATCH -J kptamd_sweep_a100
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

# size sweep, mid tier; occ_ratio = 2*nocc/nbsf in {0.1, 0.2, 0.4}
$PY bench_vhs_optimizations.py --mode both --repeats 2 --warmups 1 \
    --case 8,250,12,16 --case 8,250,25,16 --case 8,250,50,16 \
    --case 8,500,25,8 --case 8,500,50,8 --case 8,500,100,8 \
    --case 27,250,12,8 --case 27,250,25,8 --case 27,250,50,8 \
    --case 64,150,8,4 --case 64,150,15,4 --case 64,150,30,4
