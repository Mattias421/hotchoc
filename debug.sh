#!/bin/bash

PREFIX=cfm_unsup_audio

# For wav2vec-U 2.0, use raw audio features
CONFIG_NAME=tdnn_flowmatch
TASK_DATA=$DATA/LibriSpeech-10hr-rVAD/features/wav2vec_vox

PYTHONPATH=$FAIRSEQ_ROOT PREFIX=$PREFIX fairseq-hydra-train \
    --config-dir config/cfm \
    --config-name $CONFIG_NAME \
    task.data=${TASK_DATA} \
    common.user_dir=$EXP/hotchoc \
    dataset.batch_size=2 \
    dataset.validate_interval=1 \
    optimization.max_update=100000 \
    hydra.run.dir=$EXP/outputs/cfm_audio_tdnn \

