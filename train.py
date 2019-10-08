"""" train.py
This file reads and preproces the train dataset and creates (and trains) a seq2seq model
using Recurrent Neurak Networks to predict a target sequnce from an input sequnce.
openpyxl
xlrd
numpy
tensorflow
pandas
sklearn
"""

### Import required packages
import numpy as np
import random
import argparse
import pickle
import os

import tensorflow as tf
from tensorflow.python import keras
from keras.models import Sequential, load_model, Model, Input
from keras.layers import Embedding, LSTM, Dense, Activation, Dropout, TimeDistributed, Bidirectional, Lambda
from keras.callbacks import ModelCheckpoint
from keras import backend as K

## Import helper functions
from sklearn.model_selection import train_test_split

from tools import read_data, prepare_data, create_vocabulary, replace_using_dict, pad_with_zero

## Define default training data path
MT_TRAINING_CORPUS_PATH  = "./data/MT_training_corpus.xlsx"

## Specify path to save model and metadata
MT_SEQ2SEQ_MODEL_PATH    = "./model/mt_seq2seq_model.h5"
MT_MODEL_CHECKPOINT_PATH = "./model/model.chpt"
MT_META_DATA_FILE_PATH   = "./model/metadata.pickle"

## Define model parameter
# Encoder and Decoder maximum vocabulary size
encoder_vocab_size = 150
decoder_vocab_size = 50

# Encoder and Decoder sequnces length
encoder_seq_length = 20
decoder_seq_length = 15

# Number of training epcohs
num_epochs = 30

# Training Batch size
batch_size = 20

# Number of LSTM latend dimention in both Encoder and Decoder
num_latent_dim = 40

# Fraction of data used for validation during training the model
validation_size = 0.1


def data_generator(X, y, batch_size):
    """ Creates a data genrator to feed encoder and decoder input sequnces and decoder
    target sequnce
    Args:
        X: input sequnces
        y: target sequnces
    Returns:
         yields a batch of encoder and decoder input sequnces and decoder target sequnce
    """
    while True:
        for j in range(random.randint(1,len(X)-batch_size)):
            encoder_input_sequnce  = np.zeros((batch_size, encoder_seq_length, encoder_vocab_size), dtype='float32')
            decoder_input_sequnce  = np.zeros((batch_size, decoder_seq_length, decoder_vocab_size), dtype='float32')
            decoder_target_sequnce = np.zeros((batch_size, decoder_seq_length, decoder_vocab_size), dtype='float32')

            for i, (input_seq, target_seq) in enumerate(zip(X[j:j+batch_size], y[j:j+batch_size])):
                for t, word in enumerate(input_seq):
                    encoder_input_sequnce[i, t, word] = 1  # encoder input seq
                for t, word in enumerate(target_seq):
                    if t < decoder_seq_length:
                        decoder_input_sequnce[i, t, word] = 1 # decoder input seq
                    if t>0:
                        decoder_target_sequnce[i, t-1, word] = 1 # decoder target seq

            yield([encoder_input_sequnce, decoder_input_sequnce], decoder_target_sequnce)            
 


def create_seq2seq_model(encoder_vocab_size, decoder_vocab_size, latent_dim):
    """ Creates a seq2seq model using Recurrent Neural Networks(RNN).
    The encoder consists of a left-to-right LSTM layer and outputs states to decoder.
    The decoder is also consists of a left-to-right LSTM layer and outputs a sequence that
    are fed to time distributed fully connected layers with softmax activation to predict 
    target sequence. 
    Args:
        encoder_vocab_size: number of encoder tokens (i.e., encoder vocab size)
        decoder_vocab_size: size of  decoder tokens (i.e., decoder vocab size)
        latent_dim: number of LSTM hidden dimenetions
    Returns:
        model: seq2seq model
    """
    
    ### Encoder
    ## Input layer
    encoder_inputs = Input(shape=(None, encoder_vocab_size), name='encoder_input')
    ## LSTM layer
    encoder = LSTM(latent_dim, return_state=True, name='encoder_lstm')
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We keep encoder states and discard encoder ouput.
    encoder_states = [state_h, state_c]

    ### Decoder
    ## Input layer
    decoder_inputs = Input(shape=(None, decoder_vocab_size), name='decoder_input')
    ## Left to right LSTM layer
    # We set up our decoder to return full output sequences,
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name='decoder_lstm')
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                        initial_state=encoder_states)
    ## Fully connected layer
    decoder_dense = Dense(decoder_vocab_size, activation='softmax', name='decoder_dense')
    decoder_outputs = decoder_dense(decoder_outputs)

    ### Model to jointly train Encoder and Decoder 
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    return model

def create_seq2seq_inference_model(model, latent_dim):
    """ Creates a seq2seq inference model by extracting Encoder and Decoder models
     from the input seq2seq model.
    Args:
        model: a seq2seq model
        laten_dim: number of latent dimention of the seq2seq model
    Returns:
        encoder_model: encoder model of input seq2seq model
        decoder_model: decoder model of input seq2seq model
    """
    ### Inference Model
    # 1. Encode the input sequence using Encoder and return state for decoder input
    # 2. Run one step of decoder with this intial state and "start of sequnce" token
    #  as input. The output will be used as the next decoder input sequnce token
    # 3. This procedure is repteated to predict all output sequnce 
    
    ### Encoder Model
    encoder_inputs = model.input[0] 
    encoder_outputs, state_h_enc, state_c_enc = model.get_layer('encoder_lstm').output   # lstm_1
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = Model(encoder_inputs, encoder_states)
    ### Decoder Model
    ## Decoder State Input
    decoder_inputs = model.input[1]
    decoder_state_input_h = Input(shape=(latent_dim,), name='input_3')
    decoder_state_input_c = Input(shape=(latent_dim,), name='input_4')
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    ## Decoder LSTM layer
    decoder_lstm = model.get_layer('decoder_lstm')
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h_dec, state_c_dec]
    ## Decoder Fully connected layer
    decoder_dense = model.get_layer('decoder_dense')
    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_model = Model([decoder_inputs] + decoder_states_inputs,
                          [decoder_outputs] + decoder_states)

    return encoder_model, decoder_model

def train_seq2seq_model(model, X_train, X_valid, y_train, y_valid, epochs):
    """ Compiles and trains the seq2seq model. The train data is fed to model
    using a generator function
    Args:
        model: seq2seq model
        X_train: train data input sequnce (conditions)
        X_valid: train data input sequnce (conditions)
        y_train: validation target sequnce sequnce (ouputs)
        y_valid: validation target sequnce (ouputs)
        epochs: number of epochs to train model
    Returns:
        model: trained seq2seq model
    """

    # Model is trainined to minimize cross enthrop between true target sequnce
    # and predicted target sequnce
    # Optimizer is set to Nadam and accuracy is used as metric
    model.compile(loss='categorical_crossentropy',
                  optimizer='Nadam',
                  metrics=['acc'])
    
    # Creats data genrators to feed train and validation data
    train_data_generator = data_generator(X_train, y_train, batch_size)
    valid_data_generator = data_generator(X_valid, y_valid, batch_size)
    
    # Define callback fo model checkpoint
    callbacks = [ModelCheckpoint(MT_MODEL_CHECKPOINT_PATH, save_best_only=True, save_weights_only=False)]
    
    # Train the model
    model.fit_generator(train_data_generator,
                        validation_data=valid_data_generator,
                        epochs=epochs,
                        callbacks=callbacks,
                        verbose=2,
                        steps_per_epoch=len(X_train)/batch_size,
                        validation_steps=len(X_valid)/batch_size)

    return model
    

def main():
    """ The main steps to train a seq2seq model:
    1. Read dataset
    2. Preproces each sequnce (create standarized sequnces)
        a. Change QID, CONDITION and OUTPUT text to lowercas
        b. split QID, CONDITION and OUTPUT text into tokens (words)
        c. Replace QID tokens in each sample with standrized tokens (i.e., <QID0>, <QID1>, ...)
        d. Replace digit tokens in each sample with standarized tokens (i.e., <DGT0>, <DGT1>, ...)
        e. Create standardization dictionary for each sample
        f. Add special tokens <BOS> and <EOS> to the begining and end of each sequence
    3. Create dictinries to convert input and target sequnces to an integer id
    4. Replace input and outpu sequnce tokens with an integre id
    5. Pad sequnces with zero to create fixed size input and target sequnces
        a. Input sequnce is pre-padded with zero
        b. Target sequnce is post-padded 
    4. Create a seq2seq model
    4. Train the model
    5. Save the model and model metadata (inclding dictionaries to conver words to id)
    """
    
    # Construct the argument parser and parse the arguments 
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", type=str, default=MT_TRAINING_CORPUS_PATH, help="Specify the train data path")
    args = vars(ap.parse_args())
    train_data_path = args["path"]

    if not os.path.exists(train_data_path):
        print("\n Specified train data path [%s] does not exist\n" % train_data_path)
        return

    # Read dataset from Excel file
    qids_raw, conditions_raw, output_raw = read_data(train_data_path)
    print("\nLoaded train data set from [{}]\n".format(train_data_path))

    # Preprocess the raw input text data
    _, conditions, outputs, dictionaries_lemanization = prepare_data(qids_raw, conditions_raw, output_raw)
    
    # Create dictionaries to convert between word and an integer id
    # for conditions (Human Longuage) and ouputs (Machine longuage)
    condition_word2id, condition_id2word = create_vocabulary(conditions, encoder_vocab_size)
    output_word2id, output_id2word = create_vocabulary(outputs, decoder_vocab_size)
    
    # Replace words of condition and ouput with corresponding id in dictonaries
    conditions = replace_using_dict(conditions, condition_word2id, drop_unknown=True)
    outputs    = replace_using_dict(outputs, output_word2id, drop_unknown=True)

    # Fix all sequnces length to a fixed size with padding
    conditions = pad_with_zero(conditions, encoder_seq_length,'pre')
    outputs    = pad_with_zero(outputs, decoder_seq_length+1,'post')

    # Split train data into train and validation sets
    conditions_train, conditions_valid, outputs_train, outputs_valid = train_test_split(conditions, outputs, test_size=validation_size, random_state=42)

    # Created a seq2seq Recurrent Neural Network model
    model = create_seq2seq_model(encoder_vocab_size, decoder_vocab_size, num_latent_dim)
    model.summary()
    
    # Train the seq2seq model
    model = train_seq2seq_model(model, conditions_train, conditions_valid, outputs_train, outputs_valid, num_epochs)

    # Save model and metadata
    model.save(MT_SEQ2SEQ_MODEL_PATH)
    with open(MT_META_DATA_FILE_PATH,'wb') as f:
        pickle.dump([condition_word2id,condition_id2word, output_word2id, output_id2word], f)
    
    print("\nTrained seq2seq model saved in [{}]\n".format(MT_SEQ2SEQ_MODEL_PATH))
    

if __name__ == '__main__':
    main()



