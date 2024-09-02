#!/bin/bash
#SBATCH --job-name=dispatch-bt
#SBATCH --partition=gpu-5h
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=4
#SBATCH --output=logs/job-%j.out

#SBATCH --array=0-0

# 1. copy the squashed dataset to the nodes /tmp 
rsync -ah --progress /home/myasincifci/data/PACS.sqfs /tmp/

apptainer run --nv -B /tmp/PACS.sqfs:/data/PACS:image-src=/ ../../containers/main/main.sif \
    python train.py \
        --config-name debug mixstyle.p=0.0