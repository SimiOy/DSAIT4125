#!/bin/bash
#SBATCH --job-name=surgformer-ft-mb140-b0-img
#SBATCH --partition=gpu-a100
#SBATCH --account=education-eemcs-courses-dsait4125
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-task=2
#SBATCH --mem-per-cpu=8000M
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/%u/DSAIT4125/jobs/logs/%x.%j.out
#SBATCH --error=/scratch/%u/DSAIT4125/jobs/logs/%x.%j.err

set -euo pipefail

PROJECT_DIR="/scratch/${USER}/DSAIT4125"
SURGFORMER_DIR="${PROJECT_DIR}/Surgformer"
INIT_CKPT="${PROJECT_DIR}/pretrain_params/imagenet_checkpoint-best.pth"
DATA_PATH="/scratch/${USER}/datasets/MultiBypass140"

NUM_GPUS=2
TRAIN_FRACTION=1.0

module purge
module load 2024r1 miniconda3 cuda/11.6

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/${USER}/conda/envs/surgical-action-recognition

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

cd "${SURGFORMER_DIR}"

unset SLURM_PROCID SLURM_LOCALID SLURM_NTASKS SLURM_NODELIST

export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_P2P_DISABLE=0

PYTHONUNBUFFERED=1 torchrun --nproc_per_node=${NUM_GPUS} downstream_phase/run_phase_training.py \
    --batch_size 24 \
    --epochs 20 \
    --save_ckpt_freq 1 \
    --model surgformer_HTA_KCA \
    --finetune "${INIT_CKPT}" \
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
    --output_dir "${PROJECT_DIR}/results/MultiBypass140_Bern_fold0_ImageNetInit" \
    --log_dir "${PROJECT_DIR}/results/MultiBypass140_Bern_fold0_ImageNetInit" \
    --num_workers 2 \
    --clip_grad 1.0 \
    --auto_resume \
    --cut_black \
    --train_fraction ${TRAIN_FRACTION}
    
