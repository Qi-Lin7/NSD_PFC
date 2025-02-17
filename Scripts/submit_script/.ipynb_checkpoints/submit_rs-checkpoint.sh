#!/bin/sh
#-------slurm option--------#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=30G
#SBATCH --time=12:00:00
#SBATCH --array=1-78


python -u ../func/preproc_rs.py $SLURM_ARRAY_TASK_ID


