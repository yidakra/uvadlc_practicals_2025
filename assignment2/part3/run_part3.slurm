#!/bin/bash
#SBATCH --job-name=gnn-part3
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/adeliev/uvadlc_practicals_2025/assignment2/part3/logs/%x-%j.out

module purge
module load 2023

source ~/.bashrc
conda activate dl2025

cd /home/adeliev/uvadlc_practicals_2025/assignment2/part3

python public_unittest.py

python train.py --model gcn
python train.py --model matrix-gcn
python train.py --model gat

