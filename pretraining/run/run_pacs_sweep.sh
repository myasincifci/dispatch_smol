#!/bin/bash
#SBATCH --job-name=dispatch-bt
#SBATCH --partition=gpu-5h
#SBATCH --gpus-per-node=40gb:1
#SBATCH --ntasks-per-node=4
#SBATCH --output=logs/job-%j.out

#SBATCH --array=0-2
declare -a lrs=(1e-4 5e-4 1e-3)

# 1. copy the squashed dataset to the nodes /tmp 
rsync -ah --progress /home/myasincifci/data/PACS.sqfs /temp/

apptainer run --nv -B /temp/PACS.sqfs:/data/PACS:image-src=/ ../../containers/main/main.sif \
    python train.py \
        --config-name debug param.lr=${lrs[${SLURM_ARRAY_TASK_ID}]}