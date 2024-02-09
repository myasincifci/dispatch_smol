#!/bin/bash
#SBATCH --job-name=dispatch-bt
#SBATCH --partition=gpu-2h
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/job-%j.out

# 1. copy the squashed dataset to the nodes /tmp 
rsync -ah --progress /home/myasincifci/data/camelyon17_v1.0.sqfs /tmp/

apptainer run --nv -B /tmp/camelyon17_v1.0.sqfs:/data/camelyon17_v1.0:image-src=/ \
    ../../containers/dispatch-new.sif \
    python \
        train.py \
            --config-name bt_disc_ft_linear_eval
