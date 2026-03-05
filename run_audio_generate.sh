DATA=$1
CKPT=$2
RESULTS_PATH=$3

cd $FAIRSEQ_ROOT/examples/wav2vec/unsupervised

PYTHONPATH=$FAIRSEQ_ROOT python ~/w2vu_tools/w2vu_generate.py --config-dir config/generate --config-name audio \
fairseq.common.user_dir=${HOME}/hotchoc \
fairseq.task.data=$DATA \
fairseq.common_eval.path=$CKPT/checkpoint_best.pt \
fairseq.dataset.gen_subset=valid results_path=$RESULTS_PATH \
hydra.run.dir=$EXP/outputs/hydra
