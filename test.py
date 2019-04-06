import os, sys
import argparse
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# test.py -- Don't forget to put a reasonable amount code comments
# in so that we better understand what you're doing when we grade!

# add whatever additional imports you may need here.

parser = argparse.ArgumentParser(description="Test a maximum entropy model.")
parser.add_argument("-N", "--ngram", metavar="N", dest="ngram", type=int, default=3, help="The length of ngram to be considered (default 3).")
parser.add_argument("datafile", type=str,
                    help="The file name containing the features in the test data.")
parser.add_argument("modelfile", type=str,
                    help="The name of the saved model file.")

args = parser.parse_args()

print("Loading data from file {}.".format(args.datafile))
#read data from file
dataframe = pd.read_csv(args.datafile)

#labels
Y = list(dataframe[dataframe.columns[-1]])

print(dataframe[:-1])
#features
X = dataframe.drop(dataframe.columns[-1], axis =1)

print("Loading model from file {}.".format(args.modelfile))
clf = pickle.load(open(args.modelfile, 'rb'))


print("Testing {}-gram model.".format(args.ngram))

prediction = clf.predict(X)


print("Accuracy is ... {}.".format(accuracy_score(Y, prediction)))
