#!/bin/bash
#SBATCH --job-name=generate-labels-multibypass140
#SBATCH --qos=short
#SBATCH --partition=insy,general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/tudelft.net/staff-umbrella/StudentsCVlab/ccharlesworth/AutomaticMicrogliaDetection/jobs/logs/%x.%j.out
#SBATCH --error=/tudelft.net/staff-umbrella/StudentsCVlab/ccharlesworth/AutomaticMicrogliaDetection/jobs/logs/%x.%j.err

set -euo pipefail

UMBRELLA="/tudelft.net/staff-umbrella/StudentsCVlab/ccharlesworth"
PROJECT_DIR="${UMBRELLA}/AutomaticMicrogliaDetection"
CONTAINER="${UMBRELLA}/containers/surgical-action-recognition.sif"

for CENTER in bern strasbourg; do
    srun apptainer exec \
        --bind "${PROJECT_DIR}:/project" \
        --bind "${UMBRELLA}/data:/data" \
        "${CONTAINER}" \
        python /project/scripts/generate_labels_multibypass140.py \
            --data-dir /data/Microglia \
            --official-labels-root /data/Microglia_labels/labels/ \
            --center ${CENTER} \
            --fold 0
done
