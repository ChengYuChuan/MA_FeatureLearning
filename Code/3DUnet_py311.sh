#!/bin/bash
#
#SBATCH --job-name=3DAED
#SBATCH --output=Real_BS1_RBPNI_32_4Layers_CD_Cube32_MSELoss_LR2e-4_%j.txt
#SBATCH --ntasks=1
#SBATCH --time=2-0:00:00
#SBATCH --mem=15000M
#SBATCH --mail-user=yu-chuan.cheng@zo.uni-heidelberg.de
#SBATCH --mail-type=ALL
#SBATCH --nodelist=gpu08

#SBATCH --partition=students
#SBATCH --gres=gpu:mem11g:1
#SBATCH --qos=bigbatch

# JOB STEPS (example: write hostname to output file, and wait 1 minute)

#LossType = sys.argv[1] # "SSIMLoss" or "MSELoss"_1e-3 or "L1SSIMLoss" or "L1Loss"_5e-4 or HybridL1MSELoss_3e-4
#Cubesets = sys.argv[2] # "Cubes" or "MaskedCube"
#CubeSize = sys.argv[3] # "24" or "32"
#PoolType = sys.argv[4] # 'avg' or 'max'
#Learning_Rate = float(sys.argv[5]) # 0.0001
#window_size = sys.argv[6] # cube24 should be 5 or 3, cube32 should 7 or 11
#alpha = float(sys.argv[7])
#use_gaussian = sys.argv[8].lower() == "true"

#CHECK The Dir before you submit the sh file!!!!!!!!!!!!!!!!!


source ~/.bashrc
cd /home/students/cheng/3DUnet
source activate MAenv

srun hostname

srun python train.py "MSELoss" "Cubes" 32 'max' 0.0002 5 0.5 false