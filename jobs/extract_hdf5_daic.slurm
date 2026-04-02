#!/bin/bash
#SBATCH --job-name=extract-hdf5-multibypass140
#SBATCH --qos=short
#SBATCH --partition=insy,general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/tudelft.net/staff-umbrella/StudentsCVlab/ccharlesworth/DSAIT4125/jobs/logs/%x.%j.out
#SBATCH --error=/tudelft.net/staff-umbrella/StudentsCVlab/ccharlesworth/DSAIT4125/jobs/logs/%x.%j.err

set -euo pipefail

UMBRELLA="/tudelft.net/staff-umbrella/StudentsCVlab/ccharlesworth"
PROJECT_DIR="${UMBRELLA}/DSAIT4125"
CONTAINER="${UMBRELLA}/containers/surgical-action-recognition.sif"

PYPACKAGES="${UMBRELLA}/pypackages"
mkdir -p "${PYPACKAGES}"

# Install cv2 outside the container if not already there (avoids broken Qt bundled in pip wheel)
if [ ! -d "${PYPACKAGES}/cv2" ]; then
    pip install opencv-python-headless==4.8.1.78 --target "${PYPACKAGES}"
fi

srun apptainer exec \
    --bind "${PROJECT_DIR}:/project" \
    --bind "${UMBRELLA}/data:/data" \
    --bind "${PYPACKAGES}:/pypackages" \
    --env PYTHONPATH=/pypackages \
    "${CONTAINER}" \
    python /project/scripts/extract_hdf5_multibypass140.py \
        --data-dir /data/MultiBypass140 \
        --workers 8
