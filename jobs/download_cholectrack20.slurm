#!/bin/bash
#SBATCH --job-name=download_cholectrack20
#SBATCH --partition=compute
#SBATCH --account=education-eemcs-bsc-ti
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/%u/DSAIT4125/jobs/logs/download_cholectrack20.%j.out
#SBATCH --error=/scratch/%u/DSAIT4125/jobs/logs/download_cholectrack20.%j.err

set -euo pipefail

PROJECT_DIR="/scratch/${USER}/DSAIT4125"
TMP_DIR="/tmp/${SLURM_JOBID}/cholectrack20"
SCRATCH_DIR="/scratch/${USER}/datasets"
ZIP_NAME="cholectrack20.zip"

echo "Job ID      : ${SLURM_JOB_ID}"
echo "Node        : $(hostname)"
echo "Project dir : ${PROJECT_DIR}"
echo "Tmp dir     : ${TMP_DIR}"
echo "Scratch dir : ${SCRATCH_DIR}"

module load 2025 openssh git miniconda3

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/${USER}/conda/envs/surgical-action-recognition

# ── Step 1: Download to /tmp (node-local SSD) ─────────────────────────────────
mkdir -p "${TMP_DIR}"
echo "[1/3] Downloading CholecTrack20 to ${TMP_DIR} ..."
srun python "${PROJECT_DIR}/scripts/download_cholectrack20.py" --output-dir "${TMP_DIR}"

# ── Step 2: Zip to a single file (reduces file count on /scratch) ─────────────
# -0 = store only (no compression; video files are already compressed)
echo "[2/3] Zipping dataset ..."
cd "/tmp/${SLURM_JOB_ID}"
zip -0 -r "${ZIP_NAME}" cholectrack20/
echo "Zip size: $(du -sh ${ZIP_NAME} | cut -f1)"

# ── Step 3: rsync zip to /scratch ─────────────────────────────────────────────
echo "[3/3] Copying to ${SCRATCH_DIR} ..."
mkdir -p "${SCRATCH_DIR}"
rsync -ah --progress "${ZIP_NAME}" "${SCRATCH_DIR}/"

echo "Cleaning up /tmp/${SLURM_JOB_ID} ..."
rm -rf "/tmp/${SLURM_JOB_ID}"

echo "Done. Dataset available at: ${SCRATCH_DIR}/${ZIP_NAME}"