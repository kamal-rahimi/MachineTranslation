
""""
This file reads and preproces the train dataset and creates (and traind) a seq2seq model
using Recurrent Neurak Networks to predict the output sequnce from input sequnce.
"""

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

from sklearn.model_selection import train_test_split

from tools import read_data, prepare_data, create_vocabulary, replace_using_dict, pad_with_zero


MT_TRAINING_CORPUS_PATH = "./data/MT_training_corpus.xlsx"

MT_ENCODER_MODEL_PATH    = "./model/encoder.h5"
MT_DECODER_MODEL_PATH    = "./model/decoder.h5"
MT_MODEL_CHECKPOINT_PATH ="./model/decoder.chpt"

MT_META_DATA_FILE_PATH   = "./model/metadata.pickle"

encoder_vocab_size = 50
decoder_vocab_size = 30

encoder_seq_length = 10
decoder_seq_length = 7

num_epochs = 30

batch_size = 5
num_latent_dim = 10

validation_size =0.1


def data_generator(X, y, batch_size):
    """ Creates a datagenrator to feed encoder and decoder input sequnces and decoder
    target sequnce
    Args:
        X: input sequnces
        y: target sequnces
    Returns:
         yields a batch of encoder and decoder input sequnces and decoder target sequnce
    """
    while True:
        for j in range(random.randint(1,len(X)-batch_size)):
            encoder_input_sequnce  = np.zeros((batch_size, encoder_seq_length), dtype='float32')
            decoder_input_sequnce  = np.zeros((batch_size, decoder_seq_length), dtype='float32')
            decoder_target_sequnce = np.zeros((batch_size, decoder_seq_length, decoder_vocab_size), dtype='float32')

            for i, (input_seq, target_seq) in enumerate(zip(X[j:j+batch_size], y[j:j+batch_size])):
                for t, word in enumerate(input_seq):
                    encoder_input_sequnce[i, t] = word  # encoder input seq
                for t, word in enumerate(target_seq):
                    if t < decoder_seq_length:
                        decoder_input_sequnce[i, t] = word # decoder input seq
                    if t>0:
                        # decoder target sequence (one hot encoded)
                        decoder_target_sequnce[i, t-1, word] = 1

            yield([encoder_input_sequnce, decoder_input_sequnce], decoder_target_sequnce)            
 


def create_seq2seq_model(encoder_vocab_size, decoder_vocab_size, latent_dim):
    """ Creates seq2seq model using Recurrent Neural Networks(RNN).
    The encoder consists of an Embedding layer followed by a bidiectional
    LSTM layer to create bidiectional encoding of encoder sequnce.
    The encoder outputs states to decoder.
    The decoder is also consists of an Embedding layer followed by a
    left-to-right LSTM layer. The decoder LSTM  layer outputs a sequence that
    are fed to  fully connected layers with softmax activation to predict 
    output sequence. 
    Args:
        encoder_vocab_size: number of encoder tokens (i.e., encoder vocab size)
        decoder_vocab_size: size of  decoder tokens (i.e., decoder vocab size)
        latent_dim: number of LSTM hidden dimenetions
    Returns:
        model: seq2seq model
    """
    ### Encoder
    ## Input layer
    encoder_inputs = Input(shape=(None,))
    ## Embedding Layer
    encoder_embedding_layer = Embedding(encoder_vocab_size, latent_dim, mask_zero = True)   #, mask_zero = True
    encoder_embedding  = encoder_embedding_layer(encoder_inputs)
    ## Bidirectional LSTM layer
    encoder_lstm_layer = Bidirectional(LSTM(latent_dim, return_state=True))
    encoder_outputs, encoder_state_h_f, encoder_state_c_f, encoder_state_h_b, encoder_state_c_b = encoder_lstm_layer(encoder_embedding)
    # We keep encoder states and discard encoder ouput.
    # Lambda layer to add state outputs of both diections
    encoder_state_h = Lambda(lambda a: a[0] + a[1])([encoder_state_h_f, encoder_state_h_b])
    encoder_state_c = Lambda(lambda a: a[0] + a[1])([encoder_state_c_f, encoder_state_c_b])
    encoder_states = [encoder_state_h, encoder_state_c]

    ### Decoder
    ## Input layer
    decoder_inputs = Input(shape=(None,))
    ## Embedding Layer
    #  We set up the decoder initial sate using encoder states.
    decoder_embedding_layer = Embedding(decoder_vocab_size, latent_dim, mask_zero = True) #, mask_zero = True
    decoder_embedding = decoder_embedding_layer(decoder_inputs)
    ## Left to right LSTM layer
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    ## Fully connected layer
    decoder_dense = Dense(decoder_vocab_size)
    outputs_logits = decoder_dense (decoder_outputs)
    outputs = decoder_dense(decoder_outputs)
    
    ### Model to jointly train Encoder and Decoder 
    model = Model([encoder_inputs, decoder_inputs], outputs)
    

    ### Inference Model
    # 1. Encode the input sequence using Encoder and return state for decoder input
    # 2. Run one step of decoder with this intial state and "start of sequnce" token
    #  as input. The output will be used as the next decoder input sequnce token
    # 3. This procedure is repteated to predict all output sequnce 
    
    ### Encoder Model
    encoder_model = Model(encoder_inputs, encoder_states)

    ### Decoder Model
    ## Decoder State Input
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    ## Decoder Embedding layer
    decoder_embedding2 = decoder_embedding_layer(decoder_inputs)
    ## Decoder LSTM layer
    decoder_outputs2, state_h2, state_c2 = decoder_lstm(decoder_embedding2, initial_state=decoder_states_inputs)
    decoder_states2 = [state_h2, state_c2]
    ## Decoder Fully connected layer
    decoder_outputs2 = decoder_dense(decoder_outputs2)
    ## Decoder Softmax layer
    decoder_outputs2 = Activation('softmax')(decoder_outputs2)
    
    decoder_model = Model(
                          [decoder_inputs] + decoder_states_inputs,
                          [decoder_outputs2] + decoder_states2)

    return model, encoder_model, decoder_model

def train_seq2seq_model(model, X_train, X_valid, y_train, y_valid, epochs):
    """ Compiles and trains the seq2seq model. Train data is fed to model 
    using a generator function
    Args:
        model: number of encoder tokens (i.e., encoder vocab size)
        X_train: train data input sequnce (conditions)
        X_valid: train data output sequnce (conditions)
        y_train: validation input data sequnce (ouputs)
        y_valid: validation ouput data sequnce (ouputs)
        epochs: number of epochs to train model
    Returns:
        model: trained seq2seq model
    """
    model.compile(loss=tf.losses.softmax_cross_entropy,
                  optimizer='Nadam',
                  #metrics=['acc']
                  )
    
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

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path",   type=str, default=MT_TRAINING_CORPUS_PATH, help="Specify the train data path")
    args = vars(ap.parse_args())
    train_data_path = args["path"]

    if not os.path.exists(train_data_path):
        print("\n Specified train data path [%s] does not exist\n" % train_data_path)
        return

    # Read dataset from Excel file
    dataset = read_data(train_data_path)

    # Preprocess the raw input text data
    conditions, outputs, dictionaries_lemanization = prepare_data(dataset)
    
    # Create dictionaries to convert between word and an integer id
    # for conditions (Human Longuage) and ouputs
    condition_word2id, condition_id2word = create_vocabulary(conditions, encoder_vocab_size)
    output_word2id, output_id2word = create_vocabulary(outputs, decoder_vocab_size)
    
    # Replace words of condition and ouput with corresponding id in condi
    conditions = replace_using_dict(conditions, condition_word2id)
    outputs    = replace_using_dict(outputs, output_word2id)

    # Fix all sequnces length to a fixed size with padding
    conditions = pad_with_zero(conditions, encoder_seq_length,'pre')
    outputs    = pad_with_zero(outputs, decoder_seq_length+1,'post')

    # Split train data into train and validation sets
    conditions_train, conditions_valid, outputs_train, outputs_valid = train_test_split(conditions, outputs, test_size=validation_size, random_state=42)

    # Created a seq2seq Recurrent Neural Network model
    model, encoder_model, decoder_model = create_seq2seq_model(encoder_vocab_size, decoder_vocab_size, num_latent_dim)
    model.summary()
    
    # Train the seq2seq model
    model = train_seq2seq_model(model, conditions_train, conditions_valid, outputs_train, outputs_valid, num_epochs)
    
    # Save model and metadata
    encoder_model.save(MT_ENCODER_MODEL_PATH)
    decoder_model.save(MT_DECODER_MODEL_PATH)
    with open(MT_META_DATA_FILE_PATH,'wb') as f:
        pickle.dump([condition_word2id, condition_id2word, output_word2id, output_id2word], f)


if __name__ == '__main__':
    main()



