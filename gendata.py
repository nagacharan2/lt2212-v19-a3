import os, sys
import glob
import argparse
import numpy as np
import pandas as pd
import re
from nltk import word_tokenize
from nltk.util import ngrams
import random

#creates vocabulary
def unique_words(text):
    uniq_words = set()
    for line in text:
        words = re.findall('\w+', line)
        line_words = set(words)
        print(line_words)
        uniq_words.update(line_words)
        print("after update")
        print(uniq_words)
#    print(uniq_words)    
    return uniq_words


#removes all the POS tags
def pre_process(file_path):
    text = []
    lines = open(file_path).read().split("\n")
    for line in lines:
        if len(line) != 0:
            line2 = re.sub(r'\/[^\s]+','',line)
            text.append(line2)
#     print(text)
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

def generate_ngrams(s, n):
    # Convert to lowercases
    s = s.lower()
    
    # Replace all none alphanumeric characters with spaces
    s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)
    
    # Break sentence in the token, remove empty tokens
    tokens = [token for token in s.split(" ") if token != ""]
    
    # Use the zip function to help us generate n-grams
    # Concatentate the tokens into ngrams and return
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]

#implement ngram model
def n_gram_model(text, one_hot_vectors, n=3):
    ngrams_model = []
    print(text)
    # print(one_hot_vectors)
    for line in text:
        n_grams = generate_ngrams(line,n)
        print(n_grams)
        for grams in n_grams:
            print(grams)
            new_gram = grams.split(" ")
            temp = []
            for num,gram in enumerate(new_gram):
                if num == n-1:
                    temp.append(gram)
                else:
                    temp.extend(one_hot_vectors[gram])
            ngrams_model.append(temp) 
    print(ngrams_model)
    return pd.DataFrame(ngrams_model)
    # for gram in n_grams:
    #     print(gram)
        




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
if args.ngram:
    if args.ngram < 2:
        print(" ngrams cannot be less that 2")
        exit(1)

print("Loading data from file {}.".format(args.inputfile))
text = pre_process(args.inputfile)

print("Starting from line {}.".format(args.startline))

if args.endline:
    if args.startline:
        if args.startline > args.endline:
            print(" start line cannot be greater than endline")
            exit(1)
        text = text[args.startline:args.endline]
    else:
        text = text[:args.endline]
    print("Ending at line {}.".format(args.endline))
else:
    if args.startline:
        text = text[args.startline:]
    print("Ending at last line of file.")
print(text)

random.shuffle(text)
lines = round(len(text)/2)
training_data = text[lines:]
testing_data = text[:lines]

train_vocab = unique_words(training_data)
train_one_hot = one_hot_encoding(train_vocab)


test_vocab = unique_words(testing_data)
test_one_hot = one_hot_encoding(test_vocab)


data_train = n_gram_model(training_data, train_one_hot, n=args.ngram)
data_test = n_gram_model(testing_data,test_one_hot,n=args.ngram)

data_train.to_csv('train_'+ args.outputfile+'.csv')
data_test.to_csv('test_'+ args.outputfile+'.csv')
print("Constructing {}-gram model.".format(args.ngram))

print("Writing table to {}.".format(args.outputfile))
    
# THERE ARE SOME CORNER CASES YOU HAVE TO DEAL WITH GIVEN THE INPUT
# PARAMETERS BY ANALYZING THE POSSIBLE ERROR CONDITIONS.
