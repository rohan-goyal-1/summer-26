#!/bin/bash
#SBATCH --job-name=jam2d-uniform
#SBATCH --partition=gpu
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --mem=8G

set -euo pipefail

mus=(0.01 0.03 0.05 0.10 0.30 0.50 1.0)
runs_per_mu=9

idx=62

mu_idx=$((idx / runs_per_mu))
run_id=$((idx % runs_per_mu + 1))

mu=${mus[$mu_idx]}

module purge
module load miniconda
conda activate jaxdem311

python /home/rg2248/analysis/summer-26/jam/jam2d-uniform.py "$mu" "$run_id"
