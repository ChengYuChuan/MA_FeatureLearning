#!/bin/bash
#
#SBATCH --job-name=3DAED
#SBATCH --output=CheckPoint_BS2_RBPNI_16_4Layers_CD_L1Loss_LR2e-4_%j.txt
#SBATCH --ntasks=1
#SBATCH --time=2-0:00:00
#SBATCH --mem=15000M
#SBATCH --mail-user=yu-chuan.cheng@zo.uni-heidelberg.de
#SBATCH --mail-type=ALL

#SBATCH --partition=students
#SBATCH --gres=gpu:mem11g:1
#SBATCH --qos=bigbatch

# JOB STEPS (example: write hostname to output file, and wait 1 minute)

#LossType = sys.argv[1] # "SSIMLoss" or "MSELoss"_1e-3 or "L1SSIMLoss" or "L1Loss"_5e-4 or HybridL1MSELoss_3e-4
#DataFolder = sys.argv[2] # "Cubes" or "MaskedCubes"
#PoolType = sys.argv[3] # 'avg' or 'max'
#Learning_Rate = float(sys.argv[4]) # 0.0001

#CHECK The Dir before you submit the sh file!!!!!!!!!!!!!!!!!

source ~/.bashrc
cd /home/students/cheng/3DUnet
source activate MAenv

srun hostname

srun python train.py "L1Loss" "Cubes32" 'max' 0.0002