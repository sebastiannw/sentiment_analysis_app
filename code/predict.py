#Regular Packages
from  utils import one_dim_pad, one_dim_text_processing, load_word2int # pylint: disable=import-error
from  typing import Tuple, List, Dict
from  string import punctuation
from  nltk import word_tokenize
import pandas as pd
import numpy as np
import argparse
import json
import re
import os

#Torch Packages
from   torch.autograd import Variable
from   neural_net import LSTM
import torch
import torch.nn as nn
import torch.nn.functional as F


def main(review: str) -> int:
    
    # Load word dictionary
    word2int = load_word2int(os.path.join(os.getcwd(), '..', 'data', 'word2int'))
    vocabulary = len(word2int) + 1

    # Creating a model instance
    lstm = LSTM(vocabulary, n_output=1, n_embedding=500, n_hidden=1028, n_layers=2)

    # Loading the State Dictionary into the model instance
    PATH = os.path.join(os.getcwd(), '..', 'models', 'Sentiment_LSTM')
    lstm.load_state_dict(torch.load(PATH))

    # Setting the model to predict
    lstm.to('cpu')
    lstm.eval()

    # Preprocessing the review string
    review = one_dim_text_processing(review, word2int)
    review = review[None, :]
    
    hidden = lstm.init_hidden(1)
    sentiment, _ = lstm(review.long(), hidden)

    return sentiment


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Implied Volatility values in a given week.')
    parser.add_argument('--r', type=str, help='Review on which sentiment analysis will be performed.')

    args = parser.parse_args()

    PATH = os.path.join(os.getcwd(), '..', 'models', 'Sentiment_LSTM')

    sentiment = main(args.r)
    print(sentiment)