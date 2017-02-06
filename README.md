# Structured Attention Networks

Code for the paper:

[Structured Attention Networks](https://arxiv.org/pdf/1702.00887)  
Yoon Kim, Carl Denton, Luong Hoang, and Alexander M. Rush   
ICLR 2017

## Dependencies
* Python: `h5py`, `numpy`
* Lua: `nn`, `nngraph`, `cutorch`, `cunn`, `nngraph`

We additionally require a custom `cuda-mod` package which implements some custom
CUDA functions for the linear-chain CRF. This can be installed via

```
git clone https://github.com/harvardnlp/cuda-mod
cd cuda-mod && luarocks install rocks/cuda-mod-1.0-0.rockspec
```

## Models

The structured attention layers described in the paper
can be found under the folder `models/`. Specifically:
* `CRF.lua`: Segmentation attention layer (i.e. linear-chain CRF)
* `EisnerCRF.lua`: Syntactic attention layer (i.e. first-order graph-based dependency parser)

These layers are modular and can be plugged into other deep models.
We use them in place of standard simple (softmax) attention layers
for neural machine translation, natural langage inference, and question answering
(see below).

### Neural Machine Translation

#### Data
The Japanese-English data used for the paper can be downloaded by following the instructions at http://lotus.kuee.kyoto-u.ac.jp/ASPEC

#### Preprocessing
To preprocess the data, run
```
python preprocess-nmt.py --srcfile path-to-source-train --targetfile path-to-target-train
--srcvalfile path-to-source-val --targetvalfile path-to-target-val --outputfile data/nmt
```

See the `preprocess-nmt.py` file for other arguments like maximum sequence length, vocabulary size,
batch size, etc.

#### Training
*Baseline simple (i.e. softmax) attention model*
```
th train-nmt.lua -data_file path-to-train -val_data_file path-to-val -attn softmax -savefile nmt-simple
```
*Sigmoid attention*
```
th train-nmt.lua -data_file path-to-train -val_data_file path-to-val -attn sigmoid -savefile nmt-sigmoid
```
*Structured attention (i.e. segmentation attention)*
```
th train-nmt.lua -data_file path-to-train -val_data_file path-to-val -attn crf -savefile nmt-struct
```
Here `path-to-train` and `path-to-val` are the `*.hdf5` files from running `preprocess-nmt.py`.
You can add `-gpuid 1` to use the (first) GPU, and change the argument to `-savefile` if you
wish to save to a different path.

*Note: structured attention only works with the GPU.*

#### Evaluating
```
th predict-nmt.lua -src_file path-to-source-test -targ_file path-to-target-test
-src_dict path-to-source-dict -targ_dict -path-to-target-dict -output_file pred.txt
```
`-src_dict` and `-targ_dict` are the `*.dict` files created from running `preprocess-nmt.py`.
Argument to `-targ_file` is optional. The code will output predictions to `pred.txt`, and
you can again add `-gpuid 1` to use the GPU.

Evaluation is done with the `multi-bleu.perl` script from [Moses](https://github.com/moses-smt/mosesdecoder).

### Natural Language Inference

#### Data
Stanford Natural Language Inference (SNLI) dataset can be downloaded from http://nlp.stanford.edu/projects/snli/

Pre-trained GloVe embeddings can be downloaded from http://nlp.stanford.edu/projects/glove/

#### Preprocessing

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

#### Training
*Baseline model (i.e. no intra-sentence attention)*
```
th train-entail.lua -attn none -data_file path-to-train -val_data_file path-to-val
-test_data_file path-to-test -pre_word_vecs path-to-word-vecs -savefile entail-baseline
```
*Simple attention (i.e. softmax attention)*
```
th train-entail.lua -attn simple -data_file path-to-train -val_data_file path-to-val
-test_data_file path-to-test -pre_word_vecs path-to-word-vecs -savefile entail-simple
```
*Structured attention (i.e. syntactic attention)*
```
th train-entail.lua -attn struct -data_file path-to-train -val_data_file path-to-val
-test_data_file path-to-test -pre_word_vecs path-to-word-vecs -savefile entail-struct
```
Here `path-to-word-vecs` is the hdf5 file created from running `get_pretrain_vecs.py` and
the `path-to-train` are the `*.hdf5` files created from running `preprocess-entail.py`.
You can add `-gpuid 1` to use the (first) GPU, and change the argument to `-savefile` if you
wish to save to a different path.

The baseline model essentially replicates [A Decomposable Attention Model for Natural Language Inference](https://arxiv.org/abs/1606.01933). Parikh et al. EMNLP 2016.
The differences are that we use a hidden layer size of 300 (they use 200), batch size of 32 (they use 4), and train for 100 epochs (they train for 400 epochs with asynchronous SGD).

See `train-entail.lua` (or the paper) for hyperparameters and more training options.

### Question Answering

#### Data
The bAbI project (bAbI) dataset can be downloaded in all versions from https://research.fb.com/projects/babi/, or a copy of v1.0 from https://github.com/harvardnlp/MemN2N/tree/master/babi_data/en which this code was tested on. The latter is the 1k set where each task includes 1,000 questions.

#### Preprocessing

First run:
```
python preprocess-qa.py -dir input-data-path -vocabsize max-vocabulary-size
```

This will create the data hdf5 files. Vocabulary is based on the input data, and will be written to `word_to_idx.csv`. 

#### Training
For baseline model, see our [MemN2N implementation](https://github.com/harvardnlp/MemN2N).

To train structured attention with a binary-potential CRF, use:
```
th train-qa.lua -datafile data-file.hdf5 -classifier classifier-type
```
Here `data-file.hdf5` is the hdf5 file created from running `preprocess-qa.py` and
the `classifier` is either `binarycrf` or `unarycrf`. You can add `-cuda` to use the (first) GPU, and add `-save -saveminacc number` if you wish to save model (only if the accuracy on test set is at least that specified). To train with Position Encoding or Temporal Encoding (as described in [End-End Memory Networks](https://arxiv.org/pdf/1503.08895v5.pdf) Sukhbaatar et al. NIPS 2015), use `-pe` and `-te` respectively. Note that some default parameters (such as embedding size, max history etc...) are different from those used in the MemN2N paper. In addition, this code implements a 2-step CRF which is tested only on bAbI tasks with 2 supporting facts (however should in theory work for all tasks). 

See `train-qa.lua` (or the paper) for hyperparameters and more training options.

## License
MIT