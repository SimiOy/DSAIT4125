#!/bin/bash
#SBATCH --job-name=download-multibypass140
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=/tudelft.net/staff-umbrella/StudentsCVlab/ccharlesworth/jobs/logs/%x.%j.out
#SBATCH --error=/tudelft.net/staff-umbrella/StudentsCVlab/ccharlesworth/jobs/logs/%x.%j.err
# NOTE: add --account and --partition once you know your DAIC allocation
# Run: sacctmgr show user $USER to find your account

set -euo pipefail

UMBRELLA="/tudelft.net/staff-umbrella/StudentsCVlab/ccharlesworth"
PROJECT_DIR="${UMBRELLA}/DSAIT4125"
CONTAINER="${UMBRELLA}/containers/surgical-action-recognition.sif"
DATA_DIR="${UMBRELLA}/data/MultiBypass140"

mkdir -p "${UMBRELLA}/jobs/logs"

srun apptainer exec \
    --bind "${PROJECT_DIR}:/project" \
    --bind "${UMBRELLA}/data:/data" \
    "${CONTAINER}" \
    python /project/scripts/download_multibypass140.py \
        --output-dir /data/MultiBypass140
