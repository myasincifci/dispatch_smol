#!/bin/bash
#SBATCH --job-name=dp-domainnet
#SBATCH --partition=gpu-9m
#SBATCH --gpus-per-node=1
#SBATCH --constraint="80gb|h100"

#SBATCH --ntasks-per-node=4
#SBATCH --mem=128G 

#SBATCH --output=logs/job-%j.out

#SBATCH --array=1-1

# 1. copy the squashed dataset to the nodes /tmp 
rsync -ah --progress /home/myasincifci/data/domainnet_v1.0.sqfs /temp/

apptainer run --nv -B /temp/domainnet_v1.0.sqfs:/data/domainnet_v1.0:image-src=/ /home/myasincifci/containers/main/main.sif \
    python train.py \
        --config-name domainnet