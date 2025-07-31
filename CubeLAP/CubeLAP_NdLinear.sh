#!/bin/bash
#
#SBATCH --job-name=CLAP_Nd4
#SBATCH --output=Nd3_T4_16_3Layers_L1Loss_LR1e-5_Lambda1e-3_%j.txt
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
srun python CubeLAPwNdLinear.py /home/students/cheng/CubeLAP/.env_NdLinear




