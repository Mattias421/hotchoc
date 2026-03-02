#!/bin/bash
#SBATCH --job-name=w2vu_gen
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%a.out

echo "Using CPU"

#SBATCH --partition=gpu,gpu-h100,gpu-h100-nvl
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1

TEST_DATA_NAME=$1
TRAIN_DATA_NAME=$2
CKPT_NAME=$3

CKPT=$EXP/outputs/w2vu2_${TRAIN_DATA_NAME}/${CKPT_NAME}
RESULTS_PATH=~/w2vu_tools/results/$TRAIN_DATA_NAME/$CKPT_NAME

TEXT_DATA=$DATA/variety-text-corpus/${TRAIN_DATA_NAME}/text/phones

# make sure all gens in a batch use the same dict or this will cause anger
cp $TEXT_DATA/dict.phn.txt $TEST_DATA_NAME/

mkdir -p logs

apptainer exec $EXP/apptainer/unsupgan.sif ./generate.sh $TEST_DATA_NAME $CKPT $RESULTS_PATH $TEXT_DATA

# convert to trn
cd $RESULTS_PATH
awk '{print $0 " (id_" NR ")"}' train_ref.txt > ../ref.trn
awk '{print $0 " (id_" NR ")"}' train.txt > ./hyp.trn
