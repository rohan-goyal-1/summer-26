#!/bin/bash
#SBATCH --job-name=test-construction
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --mem=8G
#SBATCH --array=0-4

mus=(0.01 0.05 0.10 0.50 1.0)

idx=$((SLURM_ARRAY_TASK_ID))
mu=${mus[$idx]}

module purge
module load miniconda
conda activate jaxdem311

python /home/rg2248/project/friction/summer-26/test-construction/test-construction.py "$mu"
