# Regular Packages
from utils import text_preprocessing, dictionary_mapping, pad, save_word2int
from neural_net import LSTM
from typing import List, Dict, Tuple
from string import punctuation
from nltk import word_tokenize
import pandas as pd
import numpy as np
import json
import tarfile
import requests
import pdb
import re
import os

# Torch Packages
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

def main() -> pd.DataFrame:

    # Import data -> to be substituted for data to be trained with
    train = pd.read_csv(os.path.join(os.getcwd(), '..', 'Data', 'train.csv'))
    test  = pd.read_csv(os.path.join(os.getcwd(), '..', 'Data', 'test.csv'))

    # Preprocess text
    train = text_preprocessing(train, text_column='text')
    test  = text_preprocessing(test,  text_column='text')

    # word2int -> Map Vocabulary to Integers and save to JSON file
    # word4review -> Separate sentences into words for each piece of text
    word2int, word4review = dictionary_mapping(train, test)
    save_word2int(word2int)

    # Map word2int to word4review
    encoded_review = [[word2int[word] for word in review] for review in word4review]

    # Test-Train split
    X_length = len(train)
    X_train , X_test = encoded_review[:X_length], encoded_review[X_length:]
    y_train = [1 if label == 'pos' else 0 for label in train['label'].tolist()]
    y_test  = [1 if label == 'pos' else 0 for label in test['label'].tolist()]

    #Remove outlier reviews: [10 words < review < 500 words]
    outlier = 500
    train_outliers = [True if len(x)<=outlier else False for x in X_train]
    test_outliers  = [True if len(x)<=outlier else False for x in X_test]

    #Train outlier removal
    X_train = [x for x, y in zip(X_train, train_outliers) if y == True]
    y_train = [x for x, y in zip(y_train, train_outliers) if y == True]

    #Test outlier removal
    X_test = [x for x, y in zip(X_test, test_outliers) if y == True]
    y_test = [x for x, y in zip(y_test, test_outliers) if y == True]

    X_train = pad(X_train, outlier)
    X_test  = pad(X_test, outlier)

    return X_train, y_train, X_test, y_test

if __name__=="__main__":
    main()