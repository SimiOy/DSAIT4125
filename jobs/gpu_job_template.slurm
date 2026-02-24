#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=gpu
#SBATCH --account=education-me-courses-ro47020
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/DSAIT4125/jobs/logs/%x.%j.out
#SBATCH --error=/scratch/%u/DSAIT4125/jobs/logs/%x.%j.err

set -euo pipefail

PROJECT_DIR="/scratch/${USER}/DSAIT4125"

module load 2024r1 openmpi py-torch

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/${USER}/conda/envs/surgical-action-recognition

srun python "${PROJECT_DIR}/src/train.py"
