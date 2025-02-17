#!/bin/sh
#-------slurm option--------#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=30G
#SBATCH --time=12:00:00
#SBATCH --array=1-16

python -u ../func/CLIP_ForwardModel_univariate_reg.py $SLURM_ARRAY_TASK_ID
# python -u ../func/CLIP_ForwardModel_univariate_reg_control.py $SLURM_ARRAY_TASK_ID


