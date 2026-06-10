#!/bin/bash
#SBATCH -J kptamd_energy_h200
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

# isolated kernel parity + timing
# nbsf=150, nk=8: old heuristic picked low-k here (slow); new FLOP model picks large-k
$PY bench_local_energy.py --kmesh 2,2,2 --nbasis 150 --nocc 14 --nwalkers 32 --repeats 3
# nk=27 case: both pick large-k, measures the hoisted A GEMM
$PY bench_local_energy.py --kmesh 3,3,3 --nbasis 60 --nocc 5 --nwalkers 16 --repeats 3
# large-nbsf, few walkers: new model keeps low-k, measures the restructured low-k path
$PY bench_local_energy.py --kmesh 2,2,2 --nbasis 200 --nocc 18 --nwalkers 8 --repeats 3

# end-to-end fake AFQMC, new vs old energy kernels (propagation fixed to new)
$PY fake_kptisdf_afqmc.py --kmesh 2,2,2 --nbasis 150 --nocc 14 --nwalkers 32 \
    --nblocks 3 --nsteps 5 --propagation new --energy new
$PY fake_kptisdf_afqmc.py --kmesh 2,2,2 --nbasis 150 --nocc 14 --nwalkers 32 \
    --nblocks 3 --nsteps 5 --propagation new --energy old
$PY fake_kptisdf_afqmc.py --kmesh 3,3,3 --nbasis 60 --nocc 5 --nwalkers 16 \
    --nblocks 3 --nsteps 5 --propagation new --energy new
$PY fake_kptisdf_afqmc.py --kmesh 3,3,3 --nbasis 60 --nocc 5 --nwalkers 16 \
    --nblocks 3 --nsteps 5 --propagation new --energy old
