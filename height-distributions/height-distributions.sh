#!/bin/bash
#SBATCH --job-name=height-distributions-thomson
#SBATCH --partition=gpu_devel
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -euo pipefail

module purge
module load miniconda
conda activate jaxdem311

rads=(0.01 0.03 0.05 0.10 0.30)

for rad in "${rads[@]}"; do
    echo "[$(date)] Starting rad=$rad"

    python /home/rg2248/project/friction/summer-26/height-distributions/thomson-distributions.py "$rad"

    echo "[$(date)] Finished rad=$rad"
done
