#!/bin/bash
#SBATCH --job-name=download-multibypass140
#SBATCH --qos=short
#SBATCH --partition=insy,general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=/tudelft.net/staff-umbrella/StudentsCVlab/ccharlesworth/AutomaticMicrogliaDetection/jobs/logs/%x.%j.out
#SBATCH --error=/tudelft.net/staff-umbrella/StudentsCVlab/ccharlesworth/AutomaticMicrogliaDetection/jobs/logs/%x.%j.err

set -euo pipefail

UMBRELLA="/tudelft.net/staff-umbrella/StudentsCVlab/ccharlesworth"
PROJECT_DIR="${UMBRELLA}/AutomaticMicrogliaDetection"
CONTAINER="${UMBRELLA}/containers/surgical-action-recognition.sif"
DATA_DIR="${UMBRELLA}/data/Microglia"

mkdir -p "${PROJECT_DIR}/jobs/logs"

srun apptainer exec \
    --bind "${PROJECT_DIR}:/project" \
    --bind "${UMBRELLA}/data:/data" \
    "${CONTAINER}" \
    python /project/scripts/download_multibypass140.py \
        --output-dir /data/Microglia
