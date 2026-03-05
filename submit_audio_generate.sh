#!/bin/bash
#SBATCH --job-name=audio_gen
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%a.out

echo "Using CPU"

#SBATCH --partition=gpu,gpu-h100,gpu-h100-nvl
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1

TEST_DATA_NAME=$1
CKPT_NAME=$2

CKPT=$EXP/outputs/cfm_audio_tdnn/${CKPT_NAME}
RESULTS_PATH=$CKPT/audio_gen

mkdir -p logs
mkdir -p $RESULTS_PATH

apptainer exec $EXP/apptainer/unsupgan.sif ./run_audio_generate.sh $TEST_DATA_NAME $CKPT $RESULTS_PATH

