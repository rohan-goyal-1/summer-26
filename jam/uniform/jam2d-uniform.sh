#!/bin/bash
#SBATCH --job-name=jam2d-uniform
#SBATCH --partition=gpu
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --mem=8G
#SBATCH --array=0-62

mus=(0.01 0.03 0.05 0.10 0.30 0.50 1.0)
runs_per_mu=9

mu_idx=$((SLURM_ARRAY_TASK_ID / runs_per_mu))
run_id=$((SLURM_ARRAY_TASK_ID % runs_per_mu + 1))

mu=${mus[$mu_idx]}

module purge
module load miniconda
conda activate jaxdem311

python /home/rg2248/analysis/summer-26/jam/jam2d-uniform.py "$mu" "$run_id"