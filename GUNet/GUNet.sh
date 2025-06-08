#!/bin/bash
#
#SBATCH --job-name=S4_8
#SBATCH --output=CheckPoint_S4_BS16_8_3Layers_L1Loss_%j.txt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=200gb
#SBATCH --mail-user=yu-chuan.cheng@stud.uni-heidelberg.de
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu-single
#SBATCH --gres=gpu:A100:1

# JOB STEPS (example: write hostname to output file, and wait 1 minute)
nvidia-smi

cd /home/hd/hd_hd/hd_uu312/CUNet
source ~/.bashrc
source activate CUNet

srun hostname
srun python pretrain_encoder_main.py