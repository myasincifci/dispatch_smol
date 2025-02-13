#!/bin/bash
#SBATCH --job-name=dr-ss
#SBATCH --partition=gpu-2d
#SBATCH --gpus-per-node=80gb:1
#SBATCH --ntasks-per-node=8
#SBATCH --output=logs/job-%j.out
#SBATCH --exclude=head075
#SBATCH --mem=256G 

#SBATCH --array=0-4
declare -a lrs=(1e-5 8.25e-5 1.55e-4 2.275e-4 3e-4)

# 1. copy the squashed dataset to the nodes /tmp 
rsync -ah --progress /home/myasincifci/data/DR.sqfs /tmp

apptainer run --nv -B /tmp/DR.sqfs:/data/DR:image-src=/ /home/myasincifci/containers/main/main.sif \
    python train.py \
        --config-name dr_debug param.lr=${lrs[${SLURM_ARRAY_TASK_ID}]}