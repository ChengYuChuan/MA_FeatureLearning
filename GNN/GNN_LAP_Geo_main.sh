#!/bin/bash
#
#SBATCH --job-name=GNN_LAP
#SBATCH --output=%j_Geo_R35.txt
#SBATCH --ntasks=1
#SBATCH --time=1-0:00:00
#SBATCH --mem=10000M
#SBATCH --mail-user=yu-chuan.cheng@zo.uni-heidelberg.de
#SBATCH --mail-type=ALL

#SBATCH --partition=students
#SBATCH --gres=gpu:mem11g:1
#SBATCH --qos=bigbatch

# JOB STEPS (example: write hostname to output file, and wait 1 minute)

source ~/.bashrc
cd /home/students/cheng/GNN
source activate gcn_env

nvidia-smi
srun hostname
srun python Geo_main.py