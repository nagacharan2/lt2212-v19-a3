import os, sys
import glob
import argparse
import numpy as np
import pandas as pd
import re
from nltk import word_tokenize

#creates vocabulary
def unique_words(text):
    uniq_words = {}    
    words = re.findall('\w+', text)
    uniq_words = set(words)
#    print(uniq_words)    
    return uniq_words


#removes all the POS tags
def pre_process(file_path):
    text = []
    lines = open(file_path).read()#.split("\n")
    text = re.sub(r'/[^\s]+','',lines)
    return(text)


#implements one hot encoding
def one_hot_encoding(uniquewords):
    one_hot_vectors = {}
    
    for num, word  in enumerate(uniquewords):
        vector = [0] * len(uniquewords)
        vector[num] = 1
        one_hot_vectors[word] = vector
    print(one_hot_vectors)
    return one_hot_vectors

#implement ngram model
    

# gendata.py -- Don't forget to put a reasonable amount code comments
# in so that we better understand what you're doing when we grade!

# add whatever additional imports you may need here. You may not use the
# scikit-learn OneHotEncoder, or any related automatic one-hot encoders.

parser = argparse.ArgumentParser(description="Convert text to features")
parser.add_argument("-N", "--ngram", metavar="N", dest="ngram", type=int, default=3, help="The length of ngram to be considered (default 3).")
parser.add_argument("-S", "--start", metavar="S", dest="startline", type=int,
                    default=0,
                    help="What line of the input data file to start from. Default is 0, the first line.")
parser.add_argument("-E", "--end", metavar="E", dest="endline",
                    type=int, default=None,
                    help="What line of the input data file to end on. Default is None, whatever the last line is.")
parser.add_argument("inputfile", type=str,
                    help="The file name containing the text data.")
parser.add_argument("outputfile", type=str,
                    help="The name of the output file for the feature table.")

args = parser.parse_args()

print("Loading data from file {}.".format(args.inputfile))
text = pre_process(args.inputfile)
uniques = unique_words(text)
one_hot = one_hot_encoding(uniques)
print("Starting from line {}.".format(args.startline))
if args.endline:
    print("Ending at line {}.".format(args.endline))
else:
    print("Ending at last line of file.")

print("Constructing {}-gram model.".format(args.ngram))
print("Writing table to {}.".format(args.outputfile))
    
# THERE ARE SOME CORNER CASES YOU HAVE TO DEAL WITH GIVEN THE INPUT
# PARAMETERS BY ANALYZING THE POSSIBLE ERROR CONDITIONS.
