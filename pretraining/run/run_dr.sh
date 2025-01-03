#!/bin/bash
#SBATCH --job-name=dr-ss
#SBATCH --partition=gpu-2d
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=8
#SBATCH --output=logs/job-%j.out
#SBATCH --exclude=head075

#SBATCH --array=1-1

# 1. copy the squashed dataset to the nodes /tmp 
rsync -ah --progress /home/myasincifci/data/DR.sqfs /tmp

apptainer run --nv -B /tmp/DR.sqfs:/data/DR:image-src=/ /home/myasincifci/containers/main/main.sif \
    python train.py \
        --config-name dr-yes-no-color