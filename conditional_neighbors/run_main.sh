#!/bin/bash
#SBATCH --job-name=4-2-embed-main
#SBATCH --partition=cpu-2h

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

#SBATCH --output=logs/job-%j.out

rsync -ah --progress /home/myasincifci/dispatch_smol/conditional_neighbors/neighborhood.hdf5 /tmp/

apptainer run ../../containers/main/main.sif \
    python main.py

rsync -ah --progress /tmp/neighborhood_lookup.hdf5 /home/myasincifci/dispatch_smol/conditional_neighbors/