#!/bin/sh
#-------slurm option--------#
#SBATCH --nodes=1
#SBATCH --mem=15G
#SBATCH --time=3:59:00
#SBATCH --array=1

python -u ../func/RidgeR_noCV_CLIP.py $SLURM_ARRAY_TASK_ID
#python -u ../func/RidgeR_CV_CLIP.py $SLURM_ARRAY_TASK_ID
# python -u ../func/RidgeR_CV_CLIP_control.py $SLURM_ARRAY_TASK_ID
# python -u ../func/RidgeR_CV_CLIP_all.py $SLURM_ARRAY_TASK_ID
