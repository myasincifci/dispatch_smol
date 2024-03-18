#!/bin/bash
#SBATCH --job-name=dispatch-bt
#SBATCH --partition=gpu-5h
#SBATCH --gpus-per-node=40gb:1

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G

#SBATCH --output=logs/job-%j.out

# 1. copy the squashed dataset to the nodes /tmp 
rsync -ah --progress /home/myasincifci/data/PACS.sqfs /temp

apptainer run -B \
    /temp/PACS.sqfs:/data/PACS:image-src=/ \
    /home/myasincifci/containers/dispatch.sif \
    python ./train.py --config-name pacs