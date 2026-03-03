#!/bin/bash
#SBATCH --job-name=tdnn_grid
#SBATCH --array=0-15
#SBATCH --time=15:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%a.out
#SBATCH --partition=gpu,gpu-h100,gpu-h100-nvl
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1

mkdir -p logs

echo "Launching Array Task $SLURM_ARRAY_TASK_ID on $HOSTNAME"

job_count=0

for lr in "[0.001]" "[0.0002]" "[0.0001]" "[0.00001]"; do
  for sigma_min in 0.0 0.05 0.1 0.2 0.3; do

    if [ "$job_count" -eq "$SLURM_ARRAY_TASK_ID" ]; then
      echo "Starting task $SLURM_ARRAY_TASK_ID (lr=$lr, sigma_min=$sigma_min"
      apptainer exec --nv $EXP/apptainer/unsupgan.sif ./train_single.sh $sigma_min $lr
      exit 0
    fi

    job_count=$((job_count + 1))
  done
done
