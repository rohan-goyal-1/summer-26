#!/bin/bash
#SBATCH --job-name=jammed_friction
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

mus=(0.01 0.03 0.05 0.10 0.30 0.50 1.0)

for mu in "${mus[@]}"; do
    echo "[$(date)] Starting mu=$mu"

    python /home/rg2248/project/friction/summer-26/friction/thomson/jammed/jammed_friction.py "$mu"

    echo "[$(date)] Finished mu=$mu"
done
