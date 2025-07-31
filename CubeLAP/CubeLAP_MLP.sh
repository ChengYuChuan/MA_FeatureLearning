#!/bin/bash
#
#SBATCH --job-name=CLAP_MLP
#SBATCH --output=%j_MLP3_8192_T4_16_3Layers_L1Loss_LR1e-4_Lambda1e-3.txt
#SBATCH --ntasks=1
#SBATCH --time=1-0:00:00
#SBATCH --mem=80000M
#SBATCH --mail-user=yu-chuan.cheng@zo.uni-heidelberg.de
#SBATCH --mail-type=ALL

#SBATCH --partition=students
#SBATCH --gres=gpu:mem11g:1
#SBATCH --qos=bigbatch

# JOB STEPS (example: write hostname to output file, and wait 1 minute)

source ~/.bashrc
cd /home/students/cheng/CubeLAP
source activate NEW_CUNet

nvidia-smi
srun hostname
srun python CubeLAPwMLP_main.py /home/students/cheng/CubeLAP/.env_MLP