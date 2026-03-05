#!/bin/bash

sigma_min=$1
lr=$2

PREFIX=cfm_unsup_text

# For wav2vec-U 2.0, use raw audio features
CONFIG_NAME=text
TEXT_DATA=$DATA/variety-text-corpus/LibriLM/text/phones
KENLM_PATH=$TEXT_DATA/lm.phones.filtered.04.bin

PYTHONPATH=$FAIRSEQ_ROOT PREFIX=$PREFIX fairseq-hydra-train \
    --config-dir config/cfm \
    --config-name $CONFIG_NAME \
    task.text_data=${TEXT_DATA} \
    task.kenlm_path=${KENLM_PATH} \
    common.user_dir=$HOME/hotchoc \
    optimization.max_update=100000 \
    dataset.batch_size=64 \
    hydra.run.dir=$EXP/outputs/cfm_text_tdnn/lr_${lr}_smin_${sigma_min} \
    model.sigma_min=$sigma_min \
    optimization.lr="[${lr}]" \

