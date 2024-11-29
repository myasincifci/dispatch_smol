#!/bin/bash
#SBATCH --job-name=dp-camelyon
#SBATCH --partition=gpu-7d
#SBATCH --gpus-per-node=40gb:1
#SBATCH --ntasks-per-node=8
#SBATCH --output=logs/job-%j.out

#SBATCH --array=1-1

# 1. copy the squashed dataset to the nodes /tmp 
rsync -ah --progress /home/myasincifci/data/camelyon17_unlabeled_v1.0.sqfs /temp/
rsync -ah --progress /home/myasincifci/data/camelyon17_v1.0.sqfs /temp/

apptainer run --nv -B /temp/camelyon17_unlabeled_v1.0.sqfs:/data/camelyon17_unlabeled_v1.0:image-src=/,/temp/camelyon17_v1.0.sqfs:/data/camelyon17_v1.0:image-src=/ /home/myasincifci/containers/main/main.sif \
    python train.py \
        --cfg-path configs/camelyon/dann_no_col_ms.yaml