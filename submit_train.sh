#!/bin/bash
#SBATCH --job-name=w2vu_grid
#SBATCH --array=0-15
#SBATCH --time=15:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%a.out
#SBATCH --partition=gpu,gpu-h100,gpu-h100-nvl
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1

TEXT_DATA=$1
ORDER=$2

mkdir -p logs

echo "Launching Array Task $SLURM_ARRAY_TASK_ID on $HOSTNAME"

apptainer exec --nv $EXP/apptainer/unsupgan.sif ./train_gan.sh $TEXT_DATA $ORDER
