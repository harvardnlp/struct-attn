import numpy as np
import h5py
import re
import sys
import operator
import argparse

def load_glove_vec(fname, vocab):
    word_vecs = {}
    for line in open(fname, 'r'):
        d = line.split()
        word = d[0]
        vec = np.array(map(float, d[1:]))

        if word in vocab:
            word_vecs[word] = vec
    return word_vecs

def main():
  parser = argparse.ArgumentParser(
      description =__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--dictionary', help="*.dict file", type=str,
                      default='data/entail.word.dict')
  parser.add_argument('--glove', help='pretrained word vectors', type=str, default='')
  parser.add_argument('--outputfile', help="output hdf5 file", type=str,
                      default='data/glove.hdf5')
  
  args = parser.parse_args()
  vocab = open(args.dictionary, "r").read().split("\n")[:-1]
  vocab = map(lambda x: (x.split()[0], int(x.split()[1])), vocab)
  word2idx = {x[0]: x[1] for x in vocab}
  print("vocab size is " + str(len(vocab)))
  w2v_vecs = np.random.normal(size = (len(vocab), 300))
  w2v = load_glove_vec(args.glove, word2idx)
      
  print("num words in pretrained model is " + str(len(w2v)))
  for word, vec in w2v.items():
      w2v_vecs[word2idx[word] - 1 ] = vec
  for i in range(len(w2v_vecs)):
      w2v_vecs[i] = w2v_vecs[i] / np.linalg.norm(w2v_vecs[i])
  with h5py.File(args.outputfile, "w") as f:
    f["word_vecs"] = np.array(w2v_vecs)
    
if __name__ == '__main__':
    main()
