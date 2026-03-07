#!/bin/bash
#SBATCH --job-name=surgformer-train
#SBATCH --partition=gpu
#SBATCH --account=education-eemcs-courses-dsait4125
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=2
#SBATCH --mem-per-cpu=5G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/%u/DSAIT4125/jobs/logs/%x.%j.out
#SBATCH --error=/scratch/%u/DSAIT4125/jobs/logs/%x.%j.err

set -euo pipefail

PROJECT_DIR="/scratch/${USER}/DSAIT4125"
SURGFORMER_DIR="${PROJECT_DIR}/Surgformer"
PRETRAIN_PATH="${PROJECT_DIR}/pretrain_params/timeSformer_divST_8x32_224_K400.pyth"   # TimeSformer K400 baseline
DATA_PATH="/scratch/${USER}/data/Cholec80"
NUM_GPUS=2

# Optional settings:
# Model: surgformer_base surgformer_HTA surgformer_HTA_KCA
# Dataset: Cholec80 AutoLaparo

module load 2024r1 openmpi miniconda3 cuda/11.6

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/${USER}/conda/envs/surgical-action-recognition

# provides GLIBCXX_3.4.30+
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

cd "${SURGFORMER_DIR}"

PYTHONUNBUFFERED=1 torchrun --nproc_per_node=${NUM_GPUS} downstream_phase/run_phase_training.py \
    --batch_size 24 \
    --epochs 50 \
    --save_ckpt_freq 10 \
    --model surgformer_HTA_KCA \
    --pretrained_path "${PRETRAIN_PATH}" \
    --mixup 0.8 \
    --cutmix 1.0 \
    --smoothing 0.1 \
    --lr 5e-4 \
    --layer_decay 0.75 \
    --warmup_epochs 5 \
    --data_path "${DATA_PATH}" \
    --eval_data_path "${DATA_PATH}" \
    --nb_classes 7 \
    --data_strategy online \
    --output_mode key_frame \
    --num_frames 16 \
    --sampling_rate 4 \
    --data_set Cholec80 \
    --data_fps 1fps \
    --output_dir "${PROJECT_DIR}/results/Cholec80" \
    --log_dir "${PROJECT_DIR}/results/Cholec80" \
    --num_workers 8 \
    --no_auto_resume
