
"""
This file provides some helper functions required to read and prepare data
for the model
"""


import pandas as pd
import re
import os
import numpy as np
import pickle
from collections import Counter
from tensorflow.keras.preprocessing.sequence import pad_sequences
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS


SPLIT_PATTERN_WITH_DILIMITER = r'([`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?\n\s])\s*'
SPLIT_PATTERN_NO_DILIMITER   = r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?\n\s]\s*'


def read_data(data_path):
    """
    Reads data from an excel file
    """
    data_set = pd.read_excel(data_path)
    qids_raw       = data_set["QID"].values
    conditions_raw = data_set["CONDITION"].values
    outputs_raw    = data_set["OUTPUT"].values
    return qids_raw, conditions_raw, outputs_raw

def write_data(qids, conditions, outputs, data_path):
    """
    Writes data to excel file
    """
    data_set = pd.DataFrame(list(zip(qids, conditions, outputs)),
                            columns=["QID", "CONDITION", "OUTPUT"])
    data_set.to_excel(data_path)


def prepare_data(qids_raw, conditions_raw, outputs_raw):

    qids = []
    conditions = []
    outputs = []
    dictionaries_lemanization = []
    for qid_raw, condition_raw, output_raw in zip(qids_raw, conditions_raw, outputs_raw):
        qid, condition, output, dictionary = preprocess_sample(qid_raw, condition_raw, output_raw)
        qids.append(qid)
        conditions.append(condition)
        outputs.append(output)
        dictionaries_lemanization.append(dictionary)

    return qids, conditions, outputs, dictionaries_lemanization

def preprocess_sample(qid_raw, condition_raw, output_raw):
    
    qid, condition, output = split_to_words(qid_raw, condition_raw, output_raw)
    
    qid, condition, output, dictionary_lemenization = standardize_words(qid, condition, output)

    return qid, condition, output, dictionary_lemenization

def split_to_words(qid_raw, condition_raw, output_raw):   
    qid       = re.split(SPLIT_PATTERN_NO_DILIMITER, str(qid_raw))
    condition = re.split(SPLIT_PATTERN_NO_DILIMITER, str(condition_raw))
    condition = [cond for cond in condition if cond != " " and cond != ""]
    output    = re.split(SPLIT_PATTERN_WITH_DILIMITER, str(output_raw))

    qid       = [x.lower() for x in qid]
    condition = [x.lower() for x in condition]
    output    = [x.lower() for x in output]
    
    return qid, condition, output

def standardize_words(qid, condition, output):
    dictionary_word_convertion = {}
    for index, id in enumerate(qid):
        standard_qid = '<QID{}>'.format(index)
        dictionary_word_convertion[standard_qid] = qid[index]
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
            dictionary_word_convertion[standard_digit] = word

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
            dictionary_word_convertion[standard_digit] = word
            for word_index in range(len(output)):
                if output[word_index] == word:
                    output[word_index] = standard_digit
    
    condition   = ['<BOS>']  + condition + ['<EOS>']
    output      = ['<BOS>']  + output  + ['<EOS>']

    return qid, condition, output, dictionary_word_convertion


def create_vocabulary(word_list, max_vocab_size):
    """ Create Vocabulary dictionary
    Args:
        text(str): inout text
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
            print(word)
        if id == max_vocab_size - 1 :
            break

    return word2id, id2word


def replace_using_dict(list, dictionary, drop_unknown=False):
    translated_list = []
    for line in list:
        if drop_unknown:
            translated_line = [dictionary[word] for word in line if word in dictionary]
        else:
            translated_line = [dictionary[word] if word in dictionary else word for word in line]
        translated_list.append(translated_line)
    
    return translated_list

def pad_with_zero(list, max_length, pad_type):
    padded_list = pad_sequences(list, maxlen=max_length, padding=pad_type, truncating='post')
    return padded_list





def plot_word_cloud(word_list):
    words = ' '.join(word_list)
    wordcloud = WordCloud(width = 800, height = 800, 
                    background_color ='white',
                    stopwords = None,
                    collocations = False,
                    regexp=None,
                    min_word_length=0,
                    min_font_size = 10).generate(words) 
                         
    #plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    plt.show()


def plot_length_distribution(list, max_length=100, cdf=False):
    length_count = [0 for _ in range(max_length)]
    #item =[list]
    for item in list:
        item_length = len(item)
        if item_length < max_length:
            length_count[item_length] += 1

    length_count = np.array(length_count)
    length_freq = length_count/np.sum(length_count)
    if cdf:
        length_freq = np.cumsum(length_freq)
    plt.plot(length_freq)
    plt.show()


    """
    print(QIDs_process[:2],"\n")
    print(CONDITIONs_process[:2],"\n")
    print(CONDITIONs[:2],"\n")
    print(OUTPUTs_process[:2])
    
    all_word = [word for i in range(len(CONDITIONs_process)) for word in CONDITIONs_process[i]]
    plot_word_cloud(all_word)
    all_word = [word for i in range(len(OUTPUTs_process)) for word in OUTPUTs_process[i]]
    plot_word_cloud(all_word)
    """