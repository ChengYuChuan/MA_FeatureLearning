#!/bin/bash
#
#SBATCH --job-name=GNN_LAP
#SBATCH --output=%j_MLP512_16_3Layers_MSE_R45.txt
#SBATCH --ntasks=1
#SBATCH --time=1-0:00:00
#SBATCH --mem=10000M
#SBATCH --mail-type=ALL

#SBATCH --partition=students
#SBATCH --gres=gpu:mem11g:1
#SBATCH --qos=bigbatch

# JOB STEPS (example: write hostname to output file, and wait 1 minute)

source ~/.bashrc

source activate gcn_env

nvidia-smi
srun hostname
srun python Dynamic_main16.py