#!/bin/bash
#
#SBATCH --job-name=3DAED
#SBATCH --output=%j_CheckPoint_BS2_DoubleConv_8_3Layers_CD_L1Loss_LR2e-4.txt
#SBATCH --ntasks=1
#SBATCH --time=2-0:00:00
#SBATCH --mem=15000M
#SBATCH --mail-type=ALL

#SBATCH --partition=students
#SBATCH --gres=gpu:mem11g:1
#SBATCH --qos=bigbatch

# JOB STEPS (example: write hostname to output file, and wait 1 minute)

source ~/.bashrc
cd /home/students/cheng/3DUnet
source activate MAenv

srun hostname

srun python train.py