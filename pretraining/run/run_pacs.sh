#!/bin/bash
#SBATCH --job-name=dispatch-bt
#SBATCH --partition=gpu-2d
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=4
#SBATCH --output=logs/job-%j.out

#SBATCH --output=logs/job-%j.out

#SBATCH --array=1-10

# 1. copy the squashed dataset to the nodes /tmp 
rsync -ah --progress /home/myasincifci/dispatch_smol/data/PACS.hdf5 /tmp

ls /tmp

apptainer run --nv \
    /home/myasincifci/containers/main/main.sif \
    python ./train.py --config-name pacs