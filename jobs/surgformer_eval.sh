#!/bin/bash
#SBATCH --job-name=surgformer-eval
#SBATCH --partition=gpu-a100
#SBATCH --account=education-eemcs-courses-dsait4125
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=6G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/%u/DSAIT4125/jobs/logs/%x.%j.out
#SBATCH --error=/scratch/%u/DSAIT4125/jobs/logs/%x.%j.err

set -euo pipefail

PROJECT_DIR="/scratch/${USER}/DSAIT4125"
SURGFORMER_DIR="${PROJECT_DIR}/Surgformer"
RUN_DIR="${PROJECT_DIR}/results/Cholec80/surgformer_HTA_KCA_Cholec80_0.0005_0.75_online_key_frame_frame16_Fixed_Stride_4"
CHECKPOINT_PATH="${RUN_DIR}/checkpoint-best.pth"
DATA_PATH="/scratch/${USER}/data/Cholec80"
NUM_GPUS=1

module load 2024r1 openmpi miniconda3 cuda/11.6

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/${USER}/conda/envs/surgical-action-recognition

# provides GLIBCXX_3.4.30+
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

cd "${SURGFORMER_DIR}"

unset SLURM_PROCID SLURM_LOCALID SLURM_NTASKS SLURM_NODELIST

# Speed up NCCL init on single-node multi-GPU (skip InfiniBand/network scanning)
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_P2P_DISABLE=0

PYTHONUNBUFFERED=1 torchrun --nproc_per_node=${NUM_GPUS} downstream_phase/run_phase_training.py \
    --batch_size 32 \
    --model surgformer_HTA_KCA \
    --data_path "${DATA_PATH}" \
    --eval_data_path "${DATA_PATH}" \
    --nb_classes 7 \
    --data_strategy online \
    --output_mode key_frame \
    --num_frames 16 \
    --sampling_rate 4 \
    --data_set Cholec80 \
    --data_fps 1fps \
    --output_dir "${RUN_DIR}" \
    --log_dir "${RUN_DIR}" \
    --num_workers 4 \
    --eval \
    --resume "${CHECKPOINT_PATH}" \
    --no_auto_resume \
    --cut_black