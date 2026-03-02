#!/bin/bash

PREFIX=cfm_unsup_audio

# For wav2vec-U 2.0, use raw audio features
CONFIG_NAME=tdnn_flowmatch
TASK_DATA=$DATA/LibriSpeech-10hr-rVAD/features/wav2vec_vox


PYTHONPATH=$FAIRSEQ_ROOT PREFIX=$PREFIX fairseq-hydra-train \
    --config-dir config/cfm \
    --config-name $CONFIG_NAME \
    task.data=${TASK_DATA} \
    dataset.batch_size=160 \
    common.user_dir=$HOME/hotchoc \
    optimization.max_update=100000 \
    hydra.run.dir=$EXP/outputs/cfm_audio_tdnn \

