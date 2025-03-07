#!/bin/sh
#-------slurm option--------#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=20G
#SBATCH --time=24:00:00
#SBATCH --array=1

python -u ../func/subsample_vvs_range.py $SLURM_ARRAY_TASK_ID