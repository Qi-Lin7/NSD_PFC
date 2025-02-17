#!/bin/sh
#-------slurm option--------#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=20G
#SBATCH --time=12:00:00
#SBATCH --array=1-8

python -u /lustre/home/qilin1/Projects/NSD_GenPFC/Scripts/Manuscript/func/Extract_CLIP.py $SLURM_ARRAY_TASK_ID

