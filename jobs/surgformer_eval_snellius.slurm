#!/bin/bash
#SBATCH --job-name=surgformer-eval-mb140
#SBATCH --partition=gpu_a100
#SBATCH --account=tdsei8531
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/0/prjs0951/simi/DSAIT4125/jobs/logs/%x.%j.out
#SBATCH --error=/projects/0/prjs0951/simi/DSAIT4125/jobs/logs/%x.%j.err

set -euo pipefail

PROJECT_DIR="/projects/0/prjs0951/simi/DSAIT4125"
SURGFORMER_DIR="${PROJECT_DIR}/Surgformer"
CONDA_ENV="/projects/0/prjs0951/simi/.conda/envs/surgical-action-recognition"

DATA_PATH="/gpfs/work4/0/prjs0951/simi/data/multibypass140"

# RUN_DIR must be passed via --export=RUN_DIR=...
if [ -z "${RUN_DIR:-}" ]; then
    echo "ERROR: RUN_DIR is not set. Submit with: sbatch --export=RUN_DIR=<path> ..."
    exit 1
fi

DS_CKPT="${RUN_DIR}/checkpoint-best/mp_rank_00_model_states.pt"
CHECKPOINT_PATH="${RUN_DIR}/checkpoint-best-extracted.pth"

if [ ! -f "${DS_CKPT}" ]; then
    echo "ERROR: deepspeed checkpoint not found at ${DS_CKPT}"
    exit 1
fi

if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "Extracting model weights from deepspeed checkpoint..."
    python3 -c "
import torch
ds = torch.load('${DS_CKPT}', map_location='cpu')
torch.save({'model': ds['module']}, '${CHECKPOINT_PATH}')
print('Saved extracted checkpoint to ${CHECKPOINT_PATH}')
"
fi

module load 2023
module load CUDA/12.1.1

export PATH="${CONDA_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export CUDA_HOME="${CUDA_ROOT}"

cd "${SURGFORMER_DIR}"

unset SLURM_PROCID SLURM_LOCALID SLURM_NTASKS SLURM_NODELIST

export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_P2P_DISABLE=0

MASTER_PORT=$(( 29500 + SLURM_JOB_ID % 1000 ))

PYTHONUNBUFFERED=1 torchrun --nproc_per_node=1 --master_port=${MASTER_PORT} downstream_phase/run_phase_training.py \
    --batch_size 16 \
    --model surgformer_HTA_KCA \
    --data_path "${DATA_PATH}" \
    --eval_data_path "${DATA_PATH}" \
    --nb_classes 12 \
    --data_strategy online \
    --output_mode key_frame \
    --num_frames 16 \
    --sampling_rate 4 \
    --data_set MultiBypass140 \
    --data_fps 1fps \
    --output_dir "${RUN_DIR}" \
    --log_dir "${RUN_DIR}" \
    --num_workers 8 \
    --eval \
    --resume "${CHECKPOINT_PATH}" \
    --no_auto_resume \
    --cut_black