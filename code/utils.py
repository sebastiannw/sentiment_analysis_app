from typing import List, Dict, Tuple
from string import punctuation
import pandas as pd
import numpy as np
import tarfile
import json
import re
import os
import torch
from torch.autograd import Variable
import torch.nn as nn
from collections import Counter
from nltk import word_tokenize

def save_word2int(word2int):
    filename = os.path.join(os.getcwd(), '..', 'Data', 'word2int_2')
    with open(filename, 'w') as f:
        f.write(json.dumps(word2int))

def load_word2int(filename: str) -> Dict[str, int]:
    with open(filename) as f:
        word2int = json.loads(f.read())
    return word2int

def clean(review:str) -> str:
    regex  = re.compile('<*.?>')
    review = re.sub(regex, ' ', review)
    review = re.sub(' +',  ' ', review)
    return review

def text_preprocessing(data: pd.DataFrame, text_column: str = 'text'):
    data[text_column] = data[text_column].apply(lambda x: clean(x))
    data[text_column] = data[text_column].apply(lambda x: x.lower())
    data[text_column] = data[text_column].str.replace(f'[{punctuation}]', '')
    return data

def dictionary_mapping(train: pd.DataFrame, test: pd.DataFrame):
    full = train['text'].tolist() + test['text'].tolist()
    word4review = list(map(word_tokenize, full))
    words = [word for review in word4review for word in review]
    word_count    = Counter(words)
    frequent_sort = word_count.most_common(len(words))
    word2int = {w: i + 1 for i, (w, _) in enumerate(frequent_sort)}
    word2int['UNKNOWN_TOKEN'] = len(word2int) + 1
    return word2int, word4review
    
def pad(data: List[str], seq_length: int) -> List[str]:
    features = np.zeros((len(data), seq_length))
    
    for i, review in enumerate(data):      
        words = len(review)
        if words < seq_length:
            padding    = list(np.zeros(seq_length-words))
            new_review = padding + review    
        #This step is not necessary with our code, but still nice check    
        elif words >= seq_length:
            new_review = review[0:seq_length]  
            
        features[i,:] = np.array(new_review)
    return features

def one_dim_pad(review: List[str], seq_length:int = 500) -> np.array:
    '''Pads a single string'''
    length = len(review)
    
    if length <= seq_length:
            padding    = list(np.zeros(seq_length-length))
            new_review = padding + review       
    elif length > seq_length:
        new_review = review[0:seq_length]
        
    new_review = np.array(new_review)
    
    return new_review


def one_dim_text_processing(review: str, word_dictionary: Dict[str, int]) -> np.array:
    '''Cleans & Preprocesses a single string for the LSTM'''
    #pdb.set_trace()
    UNK_TKN = len(word_dictionary)
    review = review.lower()
    review = ''.join([ch for ch in review if ch not in punctuation])
    review = word_tokenize(review)
    review = [word_dictionary.get(word, UNK_TKN) for word in review]
    review = one_dim_pad(review, 500)
    return torch.Tensor(review)