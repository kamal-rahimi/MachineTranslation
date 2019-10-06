
""""
pip3 install xlrd
3 install wordcloud


"""


import pandas as pd
import re
import numpy as np
import argparse
import tensorflow as tf

from tensorflow.python import keras

import pickle
from collections import Counter

from matplotlib import pyplot as plt

from wordcloud import WordCloud, STOPWORDS 

from sklearn.model_selection import train_test_split


from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model, Model, Input
from keras.layers import Embedding, LSTM, Dense, Activation, Dropout, TimeDistributed, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras import metrics

import random



SPLIT_PATTERN_WITH_DILIMITER = r'([`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?\n\s])\s*'
SPLIT_PATTERN_NO_DILIMITER   = r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?\n\s]\s*'


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



def read_data(data_path):
    data_set = pd.read_excel("assignment/MT_training_corpus.xlsx")
    return data_set
    #print(data_set["CONDITION"].values)

def preprocess_sample(qid, condition, output):
    
    qid = re.split(SPLIT_PATTERN_NO_DILIMITER, str(qid))
    condition = re.split(SPLIT_PATTERN_WITH_DILIMITER, str(condition))
    condition = [cond for cond in condition if cond != " " and cond != ""]
    output = re.split(SPLIT_PATTERN_WITH_DILIMITER, str(output))

    qid       = [x.lower() for x in qid]
    condition = [x.lower() for x in condition]
    output    = [x.lower() for x in output]

    for index, id in enumerate(qid):
        standard_qid = '<QID{}>'.format(index)
        
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
            for word_index in range(len(condition)):
                if condition[word_index] == word:
                    condition[word_index] = standard_digit

            for word_index in range(len(output)):
                if output[word_index] == word:
                    output[word_index] = standard_digit

    condition   = ['<BOS>']  + condition[:encoder_seq_length-2] + ['<EOS>']
    output      = ['<BOS>']  + output[:decoder_seq_length-2+1]  + ['<EOS>']

    return qid, condition, output

def prepare_data(data_set):
    QIDs = data_set["QID"].values
    CONDITIONs = data_set["CONDITION"].values
    OUTPUTs = data_set["OUTPUT"].values

    QIDs_processed = []
    CONDITIONs_processed = []
    OUTPUTs_processed = []
    for qid, condition, output in zip(QIDs, CONDITIONs, OUTPUTs):
        q, c, o = preprocess_sample(qid, condition, output)
        QIDs_processed.append(q)
        CONDITIONs_processed.append(c)
        OUTPUTs_processed.append(o)

    return CONDITIONs_processed, OUTPUTs_processed

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

def create_vocabulary(word_list, max_vocab_size = 2000):
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
    word2id = {'<BOS>' : 1}
    id2word = {1 : '<BOS>'}
    word2id = {'<EOS>' : 2}
    id2word = {2 : '<EOS>'}
    word2id = {'<UNK>' : 3}
    id2word = {3 : '<UNK>'}

    for word, _ in freq.most_common():
        id = len(word2id)
        if word not in word2id:
            word2id[word] = id
            id2word[id] = word
        if id == max_vocab_size - 1 :
            break

    return word2id, id2word

def replace_using_dict(list, dictionary, unknown):
    translated_list = []
    for line in list:
        translated_line = [dictionary[word] if word in dictionary else unknown for word in line]
        translated_list.append(translated_line)
    
    return translated_list

def pad_front_with_zero(list, max_length=20):
    padded_list = pad_sequences(list, maxlen=max_length, truncating='pre')
    return padded_list



def data_generator(X, y, batch_size):
    """ Creates a datagenrator to feed three images to train embedding model.
     first image is the anchor, second image is related to the anchor and third image is
     unrelated to the anchor image
    Args:
        X: input images
        y: input image lables
        relation_dict: a python dictionary containing relation information
    Returns:
         yields with numpy arrays contining thjree images
    """
    max_length_src = encoder_seq_length
    max_length_tar = decoder_seq_length
    while True:
        #for j in range(0, len(X)-batch_size, batch_size):
        for j in range(random.randint(1,len(X)-batch_size)):
            encoder_input_data  = np.zeros((batch_size, max_length_src),dtype='float32')
            decoder_input_data  = np.zeros((batch_size, max_length_tar),dtype='float32')
            decoder_target_data = np.zeros((batch_size, max_length_tar, num_decoder_tokens),dtype='float32')

            for i, (input_text, target_text) in enumerate(zip(X[j:j+batch_size], y[j:j+batch_size])):
                for t, word in enumerate(input_text):
                    encoder_input_data[i, t] = t # encoder input seq
                for t, word in enumerate(target_text):
                    if t < max_length_tar:
                        decoder_input_data[i, t] = t #word # decoder input seq
                    if t>0:
                        # decoder target sequence (one hot encoded)
                        # does not include the START_ token
                        # Offset by one timestep
                        decoder_target_data[i, t-1, t] = 1
            #print(decoder_input_data[2,:],'\n')
            #print(decoder_target_data[2,:,:],'\n')
            yield([encoder_input_data, decoder_input_data], decoder_target_data)            
 


def create_model(num_encoder_tokens, num_decoder_tokens, latent_dim=40):
    """ Creates longuage model using keras
    Args:
        vocabulary vocab_size
        embedding dimmestion
    Returns:
        model
    """

    # Encoder
    encoder_inputs = Input(shape=(None,))
    
    encoder_embedding_layer = Embedding(num_encoder_tokens, latent_dim, mask_zero = True)   #, mask_zero = True
    encoder_embedding  = encoder_embedding_layer(encoder_inputs)
    
    encoder_lstm_layer = LSTM(latent_dim, return_state=True)
    encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm_layer(encoder_embedding)

    # We keep encoder states and discard encoder ouput.
    encoder_states = [encoder_state_h, encoder_state_c]

    # Decoder
    # We set up the decoder initial sate using encoder states.
    decoder_inputs = Input(shape=(None,))

    decoder_embedding_layer = Embedding(num_decoder_tokens, latent_dim, mask_zero = True) #, mask_zero = True
    decoder_embedding = decoder_embedding_layer(decoder_inputs)

    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

    decoder_dense = Dense(num_decoder_tokens, activation='softmax')

    outputs = TimeDistributed(decoder_dense)(decoder_outputs)
    
    # Define the model
    model = Model([encoder_inputs, decoder_inputs], outputs)
    

    # Next: inference mode (sampling).
    # Here's the drill:
    # 1) encode input and retrieve initial decoder state
    # 2) run one step of decoder with this initial state
    # and a "start of sequence" token as target.
    # Output will be the next target token
    # 3) Repeat with the current target token and current states

    # Define sampling models
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    # Get the embeddings of the decoder sequence
    decoder_embedding2 = decoder_embedding_layer(decoder_inputs)

    decoder_outputs2, state_h, state_c = decoder_lstm(decoder_embedding2, initial_state=decoder_states_inputs)
    decoder_states2 = [state_h, state_c]
    decoder_outputs2 = decoder_dense(decoder_outputs2)
    decoder_model = Model(
                          [decoder_inputs] + decoder_states_inputs,
                          [decoder_outputs2] + decoder_states2)


    
    return model, encoder_model, decoder_model


def train_model(model, X_train, X_valid, y_train, y_valid, epochs=100):
    # Compile the model
    callbacks = [ModelCheckpoint('model.chkpt', save_best_only=True, save_weights_only=False)]
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='Nadam',
                  metrics=['acc'])
    
    batch_size = 200
    
    
    
    train_data_generator = data_generator(X_train, y_train, batch_size)
    valid_data_generator = data_generator(X_valid, y_valid, batch_size)
    
    model.fit_generator(train_data_generator,
                        validation_data=valid_data_generator,
                        epochs=epochs,
                        callbacks=callbacks,
                        verbose=2,
                        steps_per_epoch=len(X_train)/batch_size,
                        validation_steps=len(X_valid)/batch_size)

    return model

def decode_sequence(input_seq, encoder_model, decoder_model, ML_word2id, ML_id2word):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = ML_word2id['<BOS>'] 

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    num_decoded = 0
    k=[]
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        if sampled_token_index in ML_id2word:
            sampled_char = ML_id2word[sampled_token_index]
        else:
            sampled_char = '-'

        k.append(sampled_token_index)
        decoded_sentence += sampled_char
        num_decoded += 1
        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '<EOS>' or num_decoded > 10):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]
    print(k)
    return decoded_sentence



HL_vocab_size = 1000
ML_vocab_size = 100

encoder_seq_length = 20
decoder_seq_length = 10

num_epochs = 30

num_encoder_tokens = HL_vocab_size
num_decoder_tokens = ML_vocab_size


def main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path",   type=str, default="./assignment/MT_training_corpus.xlsx", help="Specify the train data path")
    args = vars(ap.parse_args())
    train_data_path = args["path"]

    data_set = read_data(train_data_path)

    CONDITIONs, OUTPUTs = prepare_data(data_set)

    HL_word2id, HL_id2word = create_vocabulary(CONDITIONs, HL_vocab_size)
    ML_word2id, ML_id2word = create_vocabulary(OUTPUTs, ML_vocab_size)

    CONDITIONs = replace_using_dict(CONDITIONs, HL_word2id, HL_word2id["<UNK>"])
    OUTPUTs    = replace_using_dict(OUTPUTs, ML_word2id, ML_word2id["<UNK>"])

    CONDITIONs = pad_front_with_zero(CONDITIONs, encoder_seq_length)
    OUTPUTs    = pad_front_with_zero(OUTPUTs, decoder_seq_length+1)

    CONDITIONs_train, CONDITIONs_valid, OUTPUTs_train, OUTPUTs_valid = train_test_split(CONDITIONs, OUTPUTs, test_size=.1, random_state=42)

    model, encoder_model, decoder_model = create_model(num_encoder_tokens=HL_vocab_size, num_decoder_tokens=ML_vocab_size)
    model.summary()
 
    model = train_model(model, CONDITIONs_train, CONDITIONs_valid, OUTPUTs_train, OUTPUTs_valid, num_epochs)
    
    for i in range(100):

        #print("input:", [HL_id2word[word] for word in CONDITIONs_train[i] if word !=0 ],'\n')

        print("input:", ''.join([ML_id2word[word] for word in OUTPUTs_train[i] if word !=0 ]),'\n')

        input_seq = CONDITIONs_train[i]
        
        decoded_sentence = decode_sequence(input_seq, encoder_model, decoder_model, ML_word2id, ML_id2word)
        print("predicted:", ''.join(decoded_sentence),'\n')


"""
    model_path = 'model.h5'
    model.save(model_path)
    meta_data_path = 'metadata.pickle'

    with open(meta_data_path,'wb') as f:
        pickle.dump([word2id, id2word], f)

"""



if __name__ == '__main__':
    main()



