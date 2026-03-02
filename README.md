## Train

## Monitor

Get data

```
grep -oP '"valid_uer": "\K.[0-9.]"' hydra_train.log > valid_uer
```

Plot

```
gnuplot -e "set term dumb; plot 'valid_uer.txt' with lines"
```

## Eval

Generate with

```
ls /mnt/parscratch/users/acq22mc/exp/outputs/w2vu2_NewsCrawl2013/ | xargs -I {} bash -c "sbatch submit_generate.sh /mnt/parscratch/users/acq22mc/data/LibriSpeech-dev-clean-rVAD/features/wav2vec_vox NewsCrawl2013 {}
```

Then score with 

```
source score.sh results/NewsCrawl2013
```
