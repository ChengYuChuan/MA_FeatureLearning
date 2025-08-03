#!/bin/bash
#
#SBATCH --job-name=GBuild
#SBATCH --output=%j_Geo_R55.txt
#SBATCH --ntasks=1
#SBATCH --time=1-0:00:00
#SBATCH --mem=80000M
#SBATCH --mail-type=ALL

#SBATCH --partition=students
#SBATCH --gres=gpu:mem11g:1
#SBATCH --qos=bigbatch

# JOB STEPS (example: write hostname to output file, and wait 1 minute)

source ~/.bashrc

source activate gcn_env

nvidia-smi
srun hostname
srun python Graph_building.py