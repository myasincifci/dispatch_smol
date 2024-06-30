#!/bin/bash
#SBATCH --job-name=dp-domainnet
#SBATCH --partition=gpu-7d
#SBATCH --gpus-per-node=80gb:1

#SBATCH --ntasks-per-node=4
#SBATCH --mem=256G 
#SBATCH --exclude=head024

#SBATCH --output=logs/job-%j.out

#SBATCH --array=1-1

# 1. copy the squashed dataset to the nodes /tmp 
rsync -ah --progress /home/myasincifci/data/domainnet_v1.0.sqfs /tmp/

apptainer run --nv -B /tmp/domainnet_v1.0.sqfs:/data/domainnet_v1.0:image-src=/ /home/myasincifci/containers/main/main.sif \
    python train.py \
        --config-name domainnet