#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=cpu-9m
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/job-%j.out

#SBATCH --array=0-2
declare -a lrs=('1e-4' '5e-4' '1e-3')

echo "Current lr: ${lrs[${SLURM_ARRAY_TASK_ID}]}"