#!/bin/bash
#SBATCH --job-name=surgformer-ft-mb140-b0-endovit
#SBATCH --partition=gpu_a100
#SBATCH --account=tdsei8531
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --output=/projects/0/prjs0951/simi/DSAIT4125/jobs/logs/%x.%j.out
#SBATCH --error=/projects/0/prjs0951/simi/DSAIT4125/jobs/logs/%x.%j.err

set -euo pipefail

PROJECT_DIR="/projects/0/prjs0951/simi/DSAIT4125"
SURGFORMER_DIR="${PROJECT_DIR}/Surgformer"
CONDA_ENV="/projects/0/prjs0951/simi/.conda/envs/surgical-action-recognition"

ENDOVIT_CKPT="${PROJECT_DIR}/results/weights/endovit_cholec80_surgformer.pth"

DATA_PATH="/gpfs/work4/0/prjs0951/simi/data/multibypass140"
NUM_GPUS=2
TRAIN_FRACTION=1.0

OUTPUT_DIR="/gpfs/work4/0/prjs0951/simi/results/MultiBypass140_Bern_fold0_EndoViT_100pct"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${PROJECT_DIR}/jobs/logs"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

cd "${SURGFORMER_DIR}"

unset SLURM_PROCID SLURM_LOCALID SLURM_NTASKS SLURM_NODELIST

export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_P2P_DISABLE=0

PYTHONUNBUFFERED=1 torchrun --nproc_per_node=${NUM_GPUS} downstream_phase/run_phase_training.py \
    --batch_size 16 \
    --epochs 20 \
    --save_ckpt_freq 2 \
    --model surgformer_HTA_KCA \
    --finetune "${ENDOVIT_CKPT}" \
    --mixup 0.8 \
    --cutmix 1.0 \
    --smoothing 0.1 \
    --lr 1e-4 \
    --layer_decay 0.75 \
    --warmup_epochs 2 \
    --data_path "${DATA_PATH}" \
    --eval_data_path "${DATA_PATH}" \
    --nb_classes 12 \
    --data_strategy online \
    --output_mode key_frame \
    --num_frames 16 \
    --sampling_rate 4 \
    --data_set MultiBypass140 \
    --data_fps 1fps \
    --output_dir "${OUTPUT_DIR}" \
    --log_dir "${OUTPUT_DIR}" \
    --num_workers 8 \
    --clip_grad 1.0 \
    --auto_resume \
    --cut_black \
    --train_fraction ${TRAIN_FRACTION}
