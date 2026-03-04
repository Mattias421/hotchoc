#!/bin/bash

PREFIX=cfm_unsup_text

# For wav2vec-U 2.0, use raw audio features
CONFIG_NAME=text
TEXT_DATA=$DATA/variety-text-corpus/ImageCaptions/text/phones
KENLM_PATH=$TEXT_DATA/lm.phones.filtered.04.bin

PYTHONPATH=$FAIRSEQ_ROOT PREFIX=$PREFIX fairseq-hydra-train \
    --config-dir config/cfm \
    --config-name $CONFIG_NAME \
    task.text_data=${TEXT_DATA} \
    task.kenlm_path=${KENLM_PATH} \
    common.user_dir=$EXP/hotchoc \
    dataset.batch_size=2 \
    dataset.validate_interval=1 \
    dataset.validate_interval_updates=1 \
    optimization.max_update=100000 \
    hydra.run.dir=$EXP/outputs/cfm_text_tdnn \
    optimization.lr="[0.001]" \

