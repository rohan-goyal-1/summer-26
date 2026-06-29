#!/bin/bash
#SBATCH --job-name=ga_mix
#SBATCH --partition=gpu_devel
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -euo pipefail

module purge
module load miniconda
conda activate jaxdem311

mu_high=10.0
mus_low=(0.01 0.03 0.05 0.10 0.30 0.50 1.0 3.0 5.0)

for mu_low in "${mus_low[@]}"; do
    echo "[$(date)] Starting mu_low=$mu_low mu_high=$mu_high"

    python /home/rg2248/analysis/summer-26/mixing/ga-mix.py "$mu_low" "$mu_high"

    echo "[$(date)] Finished mu_low=$mu_low mu_high=$mu_high"
done