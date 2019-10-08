
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

MT_SEQ2SEQ_MODEL_PATH    = "./model/mt_seq2seq_model.h5"
MT_MODEL_CHECKPOINT_PATH ="./model/model.chpt"

MT_META_DATA_FILE_PATH   = "./model/metadata.pickle"

encoder_vocab_size = 150
decoder_vocab_size = 60

encoder_seq_length = 15
decoder_seq_length = 12

num_epochs = 30

batch_size = 20
num_latent_dim = 20

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
                        # decoder target sequence (one hot encoded)
                        decoder_target_sequnce[i, t-1, word] = 1

            yield([encoder_input_sequnce, decoder_input_sequnce], decoder_target_sequnce)            
 


def create_seq2seq_model(encoder_vocab_size, decoder_vocab_size, latent_dim):
    """ Creates seq2seq model using Recurrent Neural Networks(RNN).
    The encoder consists of an LSTM layer to create encoding of encoder sequnce.
    The encoder outputs states to decoder.
    The decoder is also consists of a left-to-right LSTM layer. The decoder LSTM  layer outputs a sequence that
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
    """ Creates a seq2seq inference model by xxtracting Encoder and Decoder model
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
    encoder_inputs = model.input[0]   # input_1
    encoder_outputs, state_h_enc, state_c_enc = model.get_layer('encoder_lstm').output   # lstm_1
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = Model(encoder_inputs, encoder_states)
    ### Decoder Model
    ## Decoder State Input
    decoder_inputs = model.input[1]  # input_2
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
    model.compile(loss='categorical_crossentropy',
                  optimizer='Nadam',
                  metrics=['acc']
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


beams = 2

def decode_sequence(input_seq, encoder_model, decoder_model, word2id, id2word):
    # Encode the input as state vectors.
    encoder_input = np.zeros((1, len(input_seq), encoder_vocab_size))
    for t, word_id in enumerate(input_seq):
        encoder_input[0, t, word_id] = 1

    states_value = encoder_model.predict([encoder_input])
    # Generate empty target sequence of length 1.
    decoder_input = np.zeros((1, 1, decoder_vocab_size))
    # Populate the first character of target sequence with the start character.
    decoder_input[0, 0, word2id['<BOS>'] ] = 1 
    seq_length = 0
    sampled_seq, sampled_seq_prob, sampled_seq_length = decode_sequence_beam(decoder_model, decoder_input, states_value, word2id, seq_length)
    
    sampled_seq_decode = [id2word[seq] for seq in reversed(sampled_seq) if seq in id2word and seq !=0 ]
    return sampled_seq_decode, sampled_seq_prob

def decode_sequence_beam(decoder_model, decoder_input, states_value, word2id, seq_length):
    
    output_tokens, h, c = decoder_model.predict([decoder_input] + states_value)
    states_value = [h, c]
    
    seq_length += 1
    # Sample a token
    beam_top_token_indecies = np.argsort(output_tokens[0, -1, :])[-beams:]
    sampled_seq_list = []
    sampled_seq_prob_list = []
    sampled_seq_length_list = []
    for beam in range(beams):
        sampled_token_index = beam_top_token_indecies[beam]
        sampled_token_prob  = output_tokens[0, -1, sampled_token_index]
        if sampled_token_index == word2id['<EOS>'] or seq_length == decoder_seq_length:
            return [sampled_token_index,0], sampled_token_prob, 0.00000001
        else:
            # Update the target sequence (of length 1).
            decoder_input = np.zeros((1, 1, decoder_vocab_size))
            decoder_input[0, 0, sampled_token_index] = 1
            # Update states
            sampled_seq, sampled_seq_prob, sampled_seq_length = decode_sequence_beam(decoder_model, decoder_input, states_value, word2id, seq_length)
            sampled_seq.append(sampled_token_index)
            sampled_seq_prob *= sampled_token_prob
            
            sampled_seq_list.append(sampled_seq)
            sampled_seq_prob_list.append(sampled_seq_prob)
            sampled_seq_length_list.append(sampled_seq_length)
    
    weighted_prob = np.log(np.array(sampled_seq_prob_list))/np.array(sampled_seq_length_list)
    
    best_beam = np.argmax(weighted_prob)
    
    return sampled_seq_list[best_beam], sampled_seq_prob_list[best_beam], sampled_seq_length_list[best_beam]+1



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
    model = create_seq2seq_model(encoder_vocab_size, decoder_vocab_size, num_latent_dim)
    model.summary()
    
    # Train the seq2seq model
    model = train_seq2seq_model(model, conditions_train, conditions_valid, outputs_train, outputs_valid, num_epochs)
    
    encoder_model, decoder_model = create_seq2seq_inference_model(model, num_latent_dim)
    for i in range(10):

        #print("input:", [condition_id2word[word] for word in CONDITIONs_train[i] if word !=0 ],'\n')

        print("input:", ''.join([output_id2word[word] for word in outputs_train[i] if word !=0 ]),'\n')

        input_seq = conditions_train[i]
        print("condition:", ''.join([condition_id2word[word] for word in conditions_train[i] if word !=0 ]),'\n')
        
        decoded_sentence, decoded_senetnce_prob = decode_sequence(input_seq, encoder_model, decoder_model, output_word2id, output_id2word)
        print("predicted:", ''.join(decoded_sentence),'\n')
        #print("sequence: ",decoded_sentence, '\n')
        print("probabaility: ",decoded_senetnce_prob, '\n')



    # Save model and metadata
    model.save(MT_SEQ2SEQ_MODEL_PATH)
    with open(MT_META_DATA_FILE_PATH,'wb') as f:
        pickle.dump([condition_word2id,condition_id2word,
                     output_word2id, output_id2word,
                     encoder_vocab_size, decoder_vocab_size,
                     num_latent_dim], f)


if __name__ == '__main__':
    main()



