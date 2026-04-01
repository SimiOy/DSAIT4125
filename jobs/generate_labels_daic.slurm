#!/bin/bash
#SBATCH --job-name=generate-labels-multibypass140
#SBATCH --qos=short
#SBATCH --partition=insy,general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/tudelft.net/staff-umbrella/StudentsCVlab/ccharlesworth/jobs/logs/%x.%j.out
#SBATCH --error=/tudelft.net/staff-umbrella/StudentsCVlab/ccharlesworth/jobs/logs/%x.%j.err

set -euo pipefail

UMBRELLA="/tudelft.net/staff-umbrella/StudentsCVlab/ccharlesworth"
PROJECT_DIR="${UMBRELLA}/DSAIT4125"
CONTAINER="${UMBRELLA}/containers/surgical-action-recognition.sif"

# The official labels are inside the downloaded dataset zip.
# After download, the structure should be:
#   /data/MultiBypass140/BernBypass70/labels_by70_splits/...
#   /data/MultiBypass140/StrasBypass70/labels_by70_splits/...
# --official-labels-root should point to the parent of bern/ and strasbourg/
# Adjust if the extracted layout differs.

for CENTER in bern strasbourg; do
    srun apptainer exec \
        --bind "${PROJECT_DIR}:/project" \
        --bind "${UMBRELLA}/data:/data" \
        "${CONTAINER}" \
        python /project/scripts/generate_labels_multibypass140.py \
            --data-dir /data/MultiBypass140 \
            --official-labels-root /data/MultiBypass140 \
            --center ${CENTER} \
            --fold 0
done
