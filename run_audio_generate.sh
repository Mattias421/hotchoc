DATA=$1
CKPT=$2
RESULTS_PATH=$3

cd $FAIRSEQ_ROOT/examples/wav2vec/unsupervised

PYTHONPATH=$FAIRSEQ_ROOT python ~/w2vu_tools/w2vu_generate.py --config-dir config/generate --config-name viterbi \
fairseq.common.user_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised \
fairseq.task.data=$DATA \
fairseq.task._name=train_audio \
fairseq.common_eval.path=$CKPT/checkpoint_best.pt \
fairseq.dataset.gen_subset=valid results_path=$RESULTS_PATH \
hydra.run.dir=$EXP/outputs/hydra
