""" tools.py
This file provides some helper functions required to read and prepare data
for the model
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import re
import numpy as np
from collections import Counter
from tensorflow.keras.preprocessing.sequence import pad_sequences


SPLIT_PATTERN_WITH_DILIMITER = r'([`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?\n\s])\s*'
SPLIT_PATTERN_NO_DILIMITER   = r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?\n\s]\s*'


def read_data(data_path):
    """ Reads data from an excel file
    Args:
        data_path: Input data path
    Returns:
        qids_raw: Pyhon list of raw qid texts
        conditions_raw: Pyhon list of raw condition texts
        outputs_raw: Pyhon list of raw output texts
    """
    data_set = pd.read_excel(data_path)
    qids_raw       = data_set["QID"].values
    conditions_raw = data_set["CONDITION"].values
    outputs_raw    = data_set["OUTPUT"].values
    return qids_raw, conditions_raw, outputs_raw

def write_data(qids, conditions, outputs, data_path):
    """ Writes data to an excel file
    Args:
        qids: Pyhon list of qid texts
        conditions: Pyhon list of condition texts
        outputs: Pyhon list of output texts
    Return:
        None
    """
    data_set = pd.DataFrame(list(zip(qids, conditions, outputs)),
                            columns=["QID", "CONDITION", "OUTPUT"])
    data_set.to_excel(data_path)


def prepare_data(qids_raw, conditions_raw, outputs_raw):
    """ Prepares data for the model
    Args:
        qids_raw: Pyhon list of raw qid texts
        conditions_raw: Pyhon list of raw condition texts
        outputs_raw: Pyhon list of raw output texts
    Returns:
        qids: Pyhon list of preprocessed qid sequnces
        conditions: Pyhon list of preprocessed condition sequnces
        outputs: Pyhon list of preprocessed output sequnces
        dictionaries_standardization: Pyhton list of dictionaries used for standardizing samples
    """

    qids = []
    conditions = []
    outputs = []
    dictionaries_standardization = []
    for qid_raw, condition_raw, output_raw in zip(qids_raw, conditions_raw, outputs_raw):
        qid, condition, output, dictionary = preprocess_sample(qid_raw, condition_raw, output_raw)
        qids.append(qid)
        conditions.append(condition)
        outputs.append(output)
        dictionaries_standardization.append(dictionary)

    return qids, conditions, outputs, dictionaries_standardization

def preprocess_sample(qid_raw, condition_raw, output_raw):
    """ Preproces a sample to create standarized sequnces
        a. Change qid_raw, condition_raw and output_raw text to lowercas
        b. split qid_raw, condition_raw and output_raw text into tokens (words)
        c. Replace qid_raw tokens with standrized tokens (i.e., <QID0>, <QID1>, ...)
        d. Replace digit tokens with standarized tokens (i.e., <DGT0>, <DGT1>, ...)
        e. Create standardization dictionary for each sample
        f. Add special tokens <BOS> and <EOS> to the begining and end of each sequence
    """
    qid, condition, output = split_to_words(qid_raw, condition_raw, output_raw)
    
    qid, condition, output, dictionary_standardization = standardize_words(qid, condition, output)

    return qid, condition, output, dictionary_standardization

def split_to_words(qid_raw, condition_raw, output_raw):
    """ Splits input raw texts into words (tokens)
    Args:
        qid_raw: raw qid text
        condition_raw: Pyhon list of raw condition texts
        output_raw: raw output texts
    Return:
        qid: Python array of qid words (tokens)
        condition: Python array of condition words (tokens)
        output: Python array of output words (tokens)
    """
    qid       = re.split(SPLIT_PATTERN_NO_DILIMITER, str(qid_raw))
    condition = re.split(SPLIT_PATTERN_NO_DILIMITER, str(condition_raw))
    condition = [cond for cond in condition if cond != " " and cond != ""]
    output    = re.split(SPLIT_PATTERN_WITH_DILIMITER, str(output_raw))

    qid       = [x.lower() for x in qid]
    condition = [x.lower() for x in condition]
    output    = [x.lower() for x in output]
    
    return qid, condition, output

def standardize_words(qid, condition, output):
    """ Standarizes a sample by replacing qids and digits with stanadard words
    Args:
        qid: Python array of qid words (tokens)
        condition: Python array of condition words (tokens)
        output: Python array of output words (tokens)
    Retursn:
        qid: Python array of standarized qid words (tokens)
        condition: Python array of standarized condition words (tokens)
        output: Python array of standarized output words (tokens)
        dictionary_standardization: Pyhton dictionary used for standardizing sample
    """
    dictionary_standardization = {}
    for index, id in enumerate(qid):
        standard_qid = '<QID{}>'.format(index)
        dictionary_standardization[standard_qid] = qid[index]
        qid[index] = standard_qid
    
        for word_index in range(len(condition)):
            if condition[word_index] == id:
                condition[word_index] = standard_qid

        for word_index in range(len(output)):
            if output[word_index] == id:
                output[word_index] = standard_qid

    digit_num = 0
    for word in condition:
        if word.isdigit():
            standard_digit = '<DGT{}>'.format(digit_num)
            digit_num += 1
            dictionary_standardization[standard_digit] = word

            for word_index in range(len(condition)):
                if condition[word_index] == word:
                    condition[word_index] = standard_digit

            for word_index in range(len(output)):
                if output[word_index] == word:
                    output[word_index] = standard_digit

    for word in output:
        if word.isdigit():
            standard_digit = '<DGT{}>'.format(digit_num)
            digit_num += 1
            dictionary_standardization[standard_digit] = word
            for word_index in range(len(output)):
                if output[word_index] == word:
                    output[word_index] = standard_digit
    
    condition   = ['<BOS>']  + condition + ['<EOS>']
    output      = ['<BOS>']  + output  + ['<EOS>']

    return qid, condition, output, dictionary_standardization


def create_vocabulary(word_list, max_vocab_size):
    """ Create Vocabulary dictionary
    Args:
        text(str): inout word list
        max_vocab_size: maximum number of words in the vocabulary
    Returns:
        word2id(dict): word to id mapping
        id2word(dict): id to word mapping
    """
    words = [word for sample in word_list for word in sample]
    freq = Counter(words)
    word2id = {'<PAD>' : 0}
    id2word = {0 : '<PAD>'}

    for word, _ in freq.most_common():
        id = len(word2id)
        if word not in word2id:
            word2id[word] = id
            id2word[id] = word
        if id == max_vocab_size - 1 :
            break

    return word2id, id2word


def replace_using_dict(list, dictionary, drop_unknown=False):
    """ Replaces tokens of the input list using a dictionary
    Args:
        list: a python list of word sequences
        dictionary: a dictionary to convert tokens
        drop_unknown: a flag to specify whether keep or drop tokens not in dictionary
    Returns:
        replaced_list: replaced Pyhthon list of word sequences 

    """
    replaced_list = []
    for line in list:
        if drop_unknown:
            translated_line = [dictionary[word] for word in line if word in dictionary]
        else:
            translated_line = [dictionary[word] if word in dictionary else word for word in line]
        replaced_list.append(translated_line)
    
    return replaced_list

def pad_with_zero(list, max_length, pad_type):
    """ Pad sequnces in the input list with zero
    Args:
        list: a Python list of word sequnces
        max_length: maximum length of each sequence
        pad_type: whether pad begining or end of the sequnces
    Return:
        padded_list: padded list of word sequnces
    """
    padded_list = pad_sequences(list, maxlen=max_length, padding=pad_type, truncating='post')
    return padded_list


def log_to_shell(index, qid_raw, condition_raw, output_raw, decoded_seqeunce):
    """ Prints information to shell
    Args:
        qid_raw: raw qid text
        condition_raw: Pyhon list of raw condition texts
        output_raw: raw output texts
        decoded_seqeunce: decoded output sequnce
    Return:
        None
    """
    print("Sample index",       index)
    print("QID: ",              qid_raw)
    print("CONDITION: ",        condition_raw)
    print("OUTPUT: ",           output_raw,'\n')
    print("Predicted OUTPUT: ", decoded_seqeunce, '\n\n')