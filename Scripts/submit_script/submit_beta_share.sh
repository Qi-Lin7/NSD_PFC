#!/bin/sh
#-------slurm option--------#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --time=23:00:00
#SBATCH --array=11

python -u ../func/BetasExt_share_all.py $SLURM_ARRAY_TASK_ID

