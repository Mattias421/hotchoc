DATA=$1
CKPT=$2
RESULTS_PATH=$3

PYTHONPATH=$FAIRSEQ_ROOT python ~/exp/hotchoc/audio_generate.py --config-dir config/generate --config-name audio \
fairseq.common.user_dir=${HOME}/exp/hotchoc \
fairseq.task.data=$DATA \
fairseq.common_eval.path=$CKPT/checkpoint_best.pt \
fairseq.dataset.gen_subset=valid results_path=$RESULTS_PATH \
hydra.run.dir=$EXP/outputs/hydra
