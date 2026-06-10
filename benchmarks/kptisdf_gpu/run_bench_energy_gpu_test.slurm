#!/bin/bash
#SBATCH -J kptamd_energy_gpu_test
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
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

# isolated kernel parity + timing, estimator-dominated case
$PY bench_local_energy.py --kmesh 2,2,2 --nbasis 100 --nocc 9 --nwalkers 32 --repeats 3

# end-to-end fake AFQMC, new vs old energy kernels (propagation fixed to new)
$PY fake_kptisdf_afqmc.py --kmesh 2,2,2 --nbasis 100 --nocc 9 --nwalkers 32 \
    --nblocks 3 --nsteps 5 --propagation new --energy new
$PY fake_kptisdf_afqmc.py --kmesh 2,2,2 --nbasis 100 --nocc 9 --nwalkers 32 \
    --nblocks 3 --nsteps 5 --propagation new --energy old
