#!/bin/bash
#SBATCH --job-name=gpt-part2
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/adeliev/uvadlc_practicals_2025/assignment2/part2/logs/%x-%j.out

module purge
module load 2023

source ~/.bashrc
conda activate dl2025

cd /home/adeliev/uvadlc_practicals_2025/assignment2/part2

python train.py --use_flash_attn --compile --num_epochs 5

