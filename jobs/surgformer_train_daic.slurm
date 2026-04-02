#!/bin/bash
#SBATCH --job-name=surgformer-ft-mb140-b0-img-025
#SBATCH --qos=medium
#SBATCH --partition=insy,general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:2
#SBATCH --mem-per-cpu=6G
#SBATCH --time=24:00:00
#SBATCH --output=/tudelft.net/staff-umbrella/StudentsCVlab/ccharlesworth/AutomaticMicrogliaDetection/jobs/logs/%x.%j.out
#SBATCH --error=/tudelft.net/staff-umbrella/StudentsCVlab/ccharlesworth/AutomaticMicrogliaDetection/jobs/logs/%x.%j.err

set -euo pipefail

UMBRELLA="/tudelft.net/staff-umbrella/StudentsCVlab/ccharlesworth"
PROJECT_DIR="${UMBRELLA}/AutomaticMicrogliaDetection"
CONTAINER="${UMBRELLA}/containers/surgical-action-recognition.sif"

CHOLEC80_CKPT="${UMBRELLA}/AutomaticMicrogliaDetection/results/weights/imagenet_cholec80_surgformer.pth"
TRAIN_FRACTION=0.25

DATA_PATH="${UMBRELLA}/data/Microglia"
OUTPUT_DIR="${UMBRELLA}/results/MultiBypass140_Bern_fold0_ImageNetInit"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${PROJECT_DIR}/jobs/logs"

srun apptainer exec --nv \
    --bind "${PROJECT_DIR}:/project" \
    --bind "${UMBRELLA}/data:/data" \
    --bind "${UMBRELLA}/results:/results" \
    "${CONTAINER}" \
    bash -c "cd /project/Surgformer && torchrun --nproc_per_node=2 \
        downstream_phase/run_phase_training.py \
        --batch_size 16 \
        --epochs 20 \
        --save_ckpt_freq 2 \
        --model surgformer_HTA_KCA \
        --finetune /project/results/weights/imagenet_cholec80_surgformer.pth \
        --mixup 0.8 \
        --cutmix 1.0 \
        --smoothing 0.1 \
        --lr 1e-4 \
        --layer_decay 0.75 \
        --warmup_epochs 2 \
        --data_path /data/Microglia \
        --eval_data_path /data/Microglia \
        --nb_classes 12 \
        --data_strategy online \
        --output_mode key_frame \
        --num_frames 16 \
        --sampling_rate 4 \
        --data_set MultiBypass140 \
        --data_fps 1fps \
        --output_dir /results/MultiBypass140_Bern_fold0_ImageNetInit \
        --log_dir /results/MultiBypass140_Bern_fold0_ImageNetInit \
        --num_workers 4 \
        --clip_grad 1.0 \
        --auto_resume \
        --cut_black \
        --train_fraction ${TRAIN_FRACTION}"