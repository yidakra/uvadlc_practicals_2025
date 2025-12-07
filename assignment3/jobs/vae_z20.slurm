#!/bin/bash
#SBATCH --job-name=vae-z20
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/adeliev/uvadlc_practicals_2025/assignment3/logs/part1/%x-%j.out

if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
fi

module purge
module load 2023

source ~/.bashrc
conda activate dl2025

WORKDIR=/home/adeliev/uvadlc_practicals_2025
cd "${WORKDIR}"

DATA_DIR="${TMPDIR:-/tmp}/${SLURM_JOB_ID:-local}/mnist_data"
mkdir -p "${DATA_DIR}"

python assignment3/part1/train_pl.py \
  --data_dir="${DATA_DIR}" \
  --log_dir=/home/adeliev/uvadlc_practicals_2025/assignment3/part1/logs/z20 \
  --z_dim=20 \
  --epochs=80 \
  --batch_size=128 \
  --num_workers="${SLURM_CPUS_PER_TASK}"

