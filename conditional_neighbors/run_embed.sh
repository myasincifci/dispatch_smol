#!/bin/bash
#SBATCH --job-name=dispatch-bt
#SBATCH --partition=gpu-5h
#SBATCH --gpus-per-node=p100:1
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/job-%j.out

# 1. copy the squashed dataset to the nodes /tmp 
rsync -ah --progress /home/myasincifci/data/camelyon17_unlabeled_v1.0.sqfs /temp/
rsync -ah --progress /home/myasincifci/data/camelyon17_v1.0.sqfs /temp/

apptainer run --nv -B /temp/camelyon17_unlabeled_v1.0.sqfs:/data/camelyon17_unlabeled_v1.0:image-src=/,/temp/camelyon17_v1.0.sqfs:/data/camelyon17_v1.0:image-src=/ ../../containers/dispatch-new.sif \
    python embed.py