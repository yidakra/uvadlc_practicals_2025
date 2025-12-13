#!/bin/bash
#SBATCH --job-name=q23-baseline
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/adeliev/uvadlc_practicals_2025/assignment3/logs/part2/%x-%j.out

if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
fi

module purge
module load 2023

source ~/.bashrc
conda activate dl2025

WORKDIR=/home/adeliev/uvadlc_practicals_2025
cd "${WORKDIR}"

CACHE_DIR="${TMPDIR:-/tmp}/${SLURM_JOB_ID:-local}/torch_cache"
mkdir -p "${CACHE_DIR}"
export TORCH_HOME="${CACHE_DIR}"

python assignment3/part2/train.py \
  --train_strats standard \
  --batch_size=128 \
  --valid_ratio=0.5 \
  --num_epochs=30 \
  --epsilon_fgsm=0.031 \
  --alpha_fgsm=0.5 \
  --epsilon_pgd=0.031 \
  --alpha_pgd=0.0078 \
  --num_iter_pgd=10 \
  --save_dir=/home/adeliev/uvadlc_practicals_2025/assignment3/part2/logs/q23_baseline \
  --test_crossover_defense


