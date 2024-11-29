#!/bin/bash
#SBATCH --job-name=dispatch-bt
#SBATCH --partition=gpu-2d
#SBATCH --gpus-per-node=40gb:1
#SBATCH --ntasks-per-node=4
#SBATCH --output=logs/job-%j.out

#SBATCH --output=logs/job-%j.out

#SBATCH --array=1-5

# 1. copy the squashed dataset to the nodes /tmp 
rsync -ah --progress /home/myasincifci/dispatch_smol/data/PACS.hdf5 /tmp

apptainer run --nv -B /tmp/PACS.hdf5:/data/PACS.hdf5 \
    /home/myasincifci/containers/main/main.sif \
    python ./train.py --cfg-path configs/pacs/dann_no_col_ms.yaml