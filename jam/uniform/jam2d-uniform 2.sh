#!/bin/bash
#SBATCH --job-name=jam2d-uniform
#SBATCH --partition=gpu
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --mem=8G
#SBATCH --array=0-7

set -euo pipefail

mus=(0.01 0.03 0.05 0.10 0.30 0.50 1.0)
runs_per_mu=9

missed_indices=(0 1 2 5 7 12 15 16)

idx=${missed_indices[$SLURM_ARRAY_TASK_ID]}

mu_idx=$((idx / runs_per_mu))
run_id=$((idx % runs_per_mu + 1))

mu=${mus[$mu_idx]}

module purge
module load miniconda
conda activate jaxdem311

python /home/rg2248/analysis/summer-26/jam/jam2d-uniform.py "$mu" "$run_id"