import os
import sys
import argparse
import numpy as np

def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_folder', help="location of folder with the snli files")
    parser.add_argument('--out_folder', help="location of the output folder")
    
    args = parser.parse_args(arguments)

    for split in ["train", "dev", "test"]:
        src_out = open(os.path.join(args.out_folder, "src-"+split+".txt"), "w")
        targ_out = open(os.path.join(args.out_folder, "targ-"+split+".txt"), "w")
        label_out = open(os.path.join(args.out_folder, "label-"+split+".txt"), "w")
        label_set = set(["neutral", "entailment", "contradiction"])

        for line in open(os.path.join(args.data_folder, "snli_1.0_"+split+".txt"),"r"):
            d = line.split("\t")
            label = d[0].strip()
            premise = " ".join(d[1].replace("(", "").replace(")", "").strip().split())
            hypothesis = " ".join(d[2].replace("(", "").replace(")", "").strip().split())
            if label in label_set:
                src_out.write(premise + "\n")
                targ_out.write(hypothesis + "\n")
                label_out.write(label + "\n")

        src_out.close()
        targ_out.close()
        label_out.close()
    
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
