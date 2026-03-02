#!/bin/bash

# Set default SLURM_ARRAY_TASK_ID if not set (for local execution)
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    SLURM_ARRAY_TASK_ID=0
    echo "SLURM_ARRAY_TASK_ID not set, running locally with task ID 0"
fi

PREFIX=w2v_unsup_gan_xp

# For wav2vec-U 2.0, use raw audio features
CONFIG_NAME=w2vu2
TASK_DATA=$DATA/LibriSpeech-10hr-rVAD/features/wav2vec_vox

# Unpaired text input
# TEXT_DATA=$DATA/variety-text-corpus/LibriLM/text/phones
TEXT_DATA_NAME=$1
ORDER=$2
TEXT_DATA=$DATA/variety-text-corpus/${TEXT_DATA_NAME}/text/phones_${ORDER}
KENLM_PATH=$TEXT_DATA/lm.phones.filtered.04.bin

job_count=0

for c_p in 0 3; do
    for g_p in 1 1.5; do
        for s_w in 1.5 2.5; do
            for mmi in 0.3 0.5; do

                if [ "$job_count" -eq "$SLURM_ARRAY_TASK_ID" ]; then
                    echo "Starting Task $SLURM_ARRAY_TASK_ID (cp=$c_p, gp=$g_p, sw=$s_w, mmi=$mmi)"

                    PYTHONPATH=$FAIRSEQ_ROOT PREFIX=$PREFIX fairseq-hydra-train \
                        --config-dir config/gan \
                        --config-name $CONFIG_NAME \
                        task.data=${TASK_DATA} \
                        task.text_data=${TEXT_DATA} \
                        task.kenlm_path=${KENLM_PATH} \
                        common.user_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised \
                        optimization.max_update=100000 \
                        dataset.num_workers=1 \
                        model.code_penalty=$c_p model.gradient_penalty=$g_p \
                        model.smoothness_weight=$s_w model.mmi_weight=$mmi \
                        hydra.run.dir=$EXP/outputs/w2vu2_${TEXT_DATA_NAME}_${ORDER}/cp_${c_p}_gp_${g_p}_sw_${s_w}_mmi_${mmi} \

                    # Exit script successfully once this specific config is submitted/run
                    exit 0
                fi

                # Increment regardless so the next loop maps to the next ID
                job_count=$((job_count + 1))
            done
        done
    done
done
