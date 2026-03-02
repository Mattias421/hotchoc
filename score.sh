# 1. Build the multi-file hypothesis string
folder=$1
HYP_ARGS=""
for file in $(ls -d $folder/*/); do
    HYP_ARGS="$HYP_ARGS -h ${file}hyp.trn"
done

sclite -r $folder/ref.trn trn $HYP_ARGS -i rm -o sgml stdout > $folder/sclite.sgml
cd $folder
cat sclite.sgml | sc_stats -g range
cd -
