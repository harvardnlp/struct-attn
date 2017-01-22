# Structured Attention Networks
Code for the paper Structured Attention Networks.

## Neural Machine Translation

### Data
The Japanese-English data can be downloaded by following the instructions at http://lotus.kuee.kyoto-u.ac.jp/ASPEC

### Preprocessing
To preprocess the data, run
```
python preprocess-nmt.py --srcfile path-to-source-train --targetfile path-to-target-train
--srcvalfile path-to-source-val --targetvalfile path-to-target-val --outputfile data/nmt
```

See the `preprocess-nmt.py` file for other arguments like maximum sequence length, vocabulary size,
batch size, etc.

### Training
Baseline simple (i.e. softmax) attention model
```
th train-nmt.lua -data_file path-to-train -val_data_file path-to-val -attn softmax -savefile nmt-simple
```
Sigmoid attention
```
th train-nmt.lua -data_file path-to-train -val_data_file path-to-val -attn sigmoid -savefile nmt-sigmoid
```
Structured attention (i.e. segmentation attention)
```
th train-nmt.lua -data_file path-to-train -val_data_file path-to-val -attn crf -savefile nmt-struct
```
Here `path-to-train` and `path-to-val` are the `*.hdf5` files from running `preprocess-nmt.py`.
You can add `-gpuid 1` to use the (first) GPU, and change the argument to `-savefile` if you
wish to save to a different path.

### Evaluating
```
th predict-nmt.lua -src_file path-to-source-test -targ_file path-to-target-test
-src_dict path-to-source-dict -targ_dict -path-to-target-dict -output_file pred.txt
```
`-src_dict` and `-targ_dict` are the `*.dict` files created from running `preprocess-nmt.py`.
Argument to `-targ_file` is optional. The code will output predictions to `pred.txt`, and
you can again add `-gpuid 1` to use the GPU.

## Natural Language Inference

### Data
Stanford Natural Language Inference (SNLI) dataset can be downloaded from http://nlp.stanford.edu/projects/snli/

Pre-trained GloVe embeddings can be downloaded from http://nlp.stanford.edu/projects/glove/

### Preprocessing

First run:
```
python preprocess-entail.py --srcfile path-to-sent1-train --targetfile path-to-sent2-train
--labelfile path-to-label-train --srcvalfile path-to-sent1-val --targetvalfile path-to-sent2-val
--labelvalfile path-to-label-val --srctestfile path-to-sent1-test --targettestfile path-to-sent2-test
--labeltestfile path-to-label-test --outputfile data/entail --glove path-to-glove
```

This will create the data hdf5 files. Vocabulary is based on the pretrained Glove embeddings,
with `path-to-glove` being the path to the pretrained Glove word vecs (i.e. the `glove.840B.300d.txt`
file). `sent1` is the premise and `sent1` is the hypothesis.

Now run:
```
python get_pretrain_vecs.py --glove path-to-glove --outputfile data/glove.hdf5
--dictionary path-to-dict
```
`path-to-dict` is the `*.word.dict` file created from running `preprocess-entail.py`

### Training
Baseline model
```
th train.lua -data_file path-to-train -val_data_file path-to-val -test_data_file path-to-test
-pre_word_vecs path-to-word-vecs -savefile entail-simple
```
Here `path-to-word-vecs` is the hdf5 file created from running `get_pretrain_vecs.py`.

The baseline model essentially replicates the results of Parikh et al. (2016). The only
differences are that we use a hidden layer size of 300 (they use 200), batch size of 32 (they use 4),
and train for 100 epochs (they train for 400 epochs with asynchronous SGD)

Structured attention (i.e. syntactic attention)
```
th train-entail.lua -parser 1 -use_parent 1 -data_file path-to-train -val_data_file path-to-val
-test_data_file path-to-test -pre_word_vecs path-to-word-vecs -savefile entail-struct
```

See `train-entail.lua` (or the paper) for hyperparameters and more training options.
You can add `-gpuid 1` to use the (first) GPU, and change the argument to `-savefile` if you
wish to save to a different path.