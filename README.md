# Structured Attention Networks

## Entailment

### Data preprocessing

First run:
```
python preprocess-entail.py --srcfile path-to-sent1-train --targetfile path-to-sent2-train
--labelfile path-to-label-train --srcvalfile path-to-sent1-val --targetvalfile path-to-sent2-val
--labelvalfile path-to-label-val --srctestfile path-to-sent1-test --targettestfile path-to-sent2-test
--labeltestfile path-to-label-test --outputfile data/entail --glove path-to-glove
```

This will create the data hdf5 files. Vocabulary is based on the pretrained Glove embeddings.
sent1 is the premise and sent1 is the hypothesis.

Now run:
```
python get_pretrain_vecs.py --wv_file path-to-glove --outputfile data/glove.hdf5
--dictionary path-to-dict
```
`path-to-dict` is the `*.word.dict` file created from running `preprocess-entail.py`

### Training
Baseline model 
```
th train-entail.lua -parser 0
```
The baseline model essentially replicates the results of Parikh et al. (2016). The only
differences are that we use a hidden layer size of 300 (they use 200), batch size of 32 (they use 4),
and train for 100 epochs (they train for 400 epochs with asynchronous SGD)

Structured attention
```
th train-entail.lua -parser 1 -use_parent 1
```
See `train-entail.lua` (or the paper) for hyperparameters and more training options.
You can add `-gpuid 1` to use the GPU.