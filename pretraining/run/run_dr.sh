#!/bin/bash
#SBATCH --job-name=dr-ss-cj
#SBATCH --partition=gpu-2d
#SBATCH --gpus-per-node=80gb:1
#SBATCH --ntasks-per-node=8
#SBATCH --output=logs/job-%j.out
#SBATCH --exclude=head075
#SBATCH --mem=256G 

#SBATCH --array=0-4
declare -a seeds=(42 69 666 404 505)

# 1. copy the squashed dataset to the nodes /tmp 
rsync -ah --progress /home/myasincifci/data/DR.sqfs /tmp

apptainer run --nv -B /tmp/DR.sqfs:/data/DR:image-src=/ /home/myasincifci/containers/main/main.sif \
    python train.py \
        --config-name dr_cj_scale_100 seed=${seeds[${SLURM_ARRAY_TASK_ID}]} data.color_aug_fct=1.5