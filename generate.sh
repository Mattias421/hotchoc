DATA=$1
CKPT=$2
RESULTS_PATH=$3
TEXT_DATA=$4

cd $FAIRSEQ_ROOT/examples/wav2vec/unsupervised

PYTHONPATH=$FAIRSEQ_ROOT python ~/w2vu_tools/w2vu_generate.py --config-dir config/generate --config-name viterbi \
fairseq.common.user_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised \
fairseq.task.data=$DATA \
fairseq.common_eval.path=$CKPT/checkpoint_best.pt \
fairseq.dataset.gen_subset=train results_path=$RESULTS_PATH \
decode_stride=2 \
lexicon=$TEXT_DATA/dict.phn.txt \
beam=50 \
hydra.run.dir=$EXP/outputs/hydra
# w2l_decoder=KALDI \
# kaldi_decoder_config.hlg_graph_path=$TEXT_DATA/fst/phn_to_phn_sil/HLG.phn.lm.phones.filtered.06.fst \
# kaldi_decoder_config.output_dict=$TEXT_DATA/fst/phn_to_phn_sil/kaldi_dict.phn.txt \
