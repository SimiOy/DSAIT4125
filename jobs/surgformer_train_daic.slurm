#!/bin/bash
#SBATCH --job-name=surgformer-train-multibypass140
#SBATCH --qos=short
#SBATCH --partition=insy,general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00
#SBATCH --output=/tudelft.net/staff-umbrella/StudentsCVlab/ccharlesworth/DSAIT4125/jobs/logs/%x.%j.out
#SBATCH --error=/tudelft.net/staff-umbrella/StudentsCVlab/ccharlesworth/DSAIT4125/jobs/logs/%x.%j.err

set -euo pipefail

UMBRELLA="/tudelft.net/staff-umbrella/StudentsCVlab/ccharlesworth"
PROJECT_DIR="${UMBRELLA}/DSAIT4125"
CONTAINER="${UMBRELLA}/containers/surgical-action-recognition.sif"
DATA_DIR="${UMBRELLA}/data/MultiBypass140"
PRETRAIN="${UMBRELLA}/pretrain_params/pytorch_model.bin"
OUTPUT_DIR="${UMBRELLA}/results/surgformer_multibypass140"

mkdir -p "${OUTPUT_DIR}"

srun apptainer exec --nv \
    --bind "${PROJECT_DIR}:/project" \
    --bind "${UMBRELLA}/data:/data" \
    --bind "${UMBRELLA}/pretrain_params:/pretrain_params" \
    --bind "${UMBRELLA}/results:/results" \
    "${CONTAINER}" \
    python /project/Surgformer/downstream_phase/run_phase_training.py \
        --model surgformer_HTA \
        --data_set MultiBypass140 \
        --data_strategy offline \
        --output_mode key_frame \
        --cut_black \
        --data_fps 1fps \
        --nb_classes 7 \
        --data_path /data/MultiBypass140 \
        --pretrained_path /pretrain_params/pytorch_model.bin \
        --output_dir /results/surgformer_multibypass140 \
        --batch_size 12 \
        --num_frames 8 \
        --sampling_rate 4 \
        --epochs 50 \
        --no_auto_resume
