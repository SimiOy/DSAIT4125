#!/bin/bash
#SBATCH --job-name=extract-hdf5-mb140
#SBATCH --partition=staging
#SBATCH --account=tdsei8531
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/projects/0/prjs0951/simi/DSAIT4125/jobs/logs/%x.%j.out
#SBATCH --error=/projects/0/prjs0951/simi/DSAIT4125/jobs/logs/%x.%j.err

set -euo pipefail

PROJECT_DIR="/projects/0/prjs0951/simi/DSAIT4125"
DATA_DIR="/gpfs/work4/0/prjs0951/simi/data/multibypass140"
CONDA_ENV="/projects/0/prjs0951/simi/.conda/envs/surgical-action-recognition"

mkdir -p "${PROJECT_DIR}/jobs/logs"

source "${HOME}/.bashrc"
conda activate "${CONDA_ENV}"

python "${PROJECT_DIR}/scripts/extract_hdf5_multibypass140.py" \
    --data-dir "${DATA_DIR}" \
    --workers 16
