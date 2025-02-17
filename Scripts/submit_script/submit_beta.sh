#!/bin/sh
#-------slurm option--------#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --time=23:00:00
#SBATCH --array=198


python -u ../func/BetasExt_unique.py $SLURM_ARRAY_TASK_ID
#python -u ../func/BetasExt_share.py $SLURM_ARRAY_TASK_ID
#python -u ../func/BetasExt_unique_control.py $SLURM_ARRAY_TASK_ID
#python -u ../func/BetasExt_share_control.py $SLURM_ARRAY_TASK_ID


