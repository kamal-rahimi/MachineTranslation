
""""
pip3 install xlrd
3 install wordcloud

"""

import numpy as np
import random
import argparse

import tensorflow as tf
from tensorflow.python import keras
from keras.models import Sequential, load_model, Model, Input
from keras.layers import Embedding, LSTM, Dense, Activation, Dropout, TimeDistributed, Bidirectional, Lambda
from keras.callbacks import ModelCheckpoint
from keras import backend as K

from sklearn.model_selection import train_test_split


from tools import read_data, prepare_data, create_vocabulary, replace_using_dict, pad_with_zero


MT_TRAINING_CORPUS_PATH = "./assignment/MT_training_corpus.xlsx"


encoder_vocab_size = 50
decoder_vocab_size = 30

encoder_seq_length = 10
decoder_seq_length = 7

num_epochs = 30

num_encoder_tokens = encoder_vocab_size
num_decoder_tokens = decoder_vocab_size

batch_size = 5
num_latent_dim = 10

beams = 3

validation_size =0.1



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
        for j in range(random.randint(1,len(X)-batch_size)):
            encoder_input_data  = np.zeros((batch_size, max_length_src), dtype='float32')
            decoder_input_data  = np.zeros((batch_size, max_length_tar), dtype='float32')
            decoder_target_data = np.zeros((batch_size, max_length_tar, num_decoder_tokens), dtype='float32')

            for i, (input_text, target_text) in enumerate(zip(X[j:j+batch_size], y[j:j+batch_size])):
                for t, word in enumerate(input_text):
                    encoder_input_data[i, t] = word  # encoder input seq
                for t, word in enumerate(target_text):
                    if t < max_length_tar:
                        decoder_input_data[i, t] = word # decoder input seq
                    if t>0:
                        # decoder target sequence (one hot encoded)
                        decoder_target_data[i, t-1, word] = 1

            yield([encoder_input_data, decoder_input_data], decoder_target_data)            
 


def create_seq2seq_model(num_encoder_tokens, num_decoder_tokens, latent_dim):
    """ Creates seq2seq model using Recurrent Neural Networks(RNN).
    The encoder consists of an Embedding layer followed by a bidiectional
    LSTM layer to create bidiectional encoding of encoder sequnce.
    The encoder outputs states to decoder.
    The decoder is also consists of an Embedding layer followed by a
    left-to-right LSTM layer. The decoder LSTM  layer outputs a sequence that
    are fed to  fully connected layers with softmax activation to predict 
    output sequence. 
    Args:
        num_encoder_tokens: number of encoder tokens (i.e., encoder vocab size)
        num_decoder_tokens: size of  decoder tokens (i.e., decoder vocab size)
        latent_dim: number of LSTM hidden dimenetions
    Returns:
        model: seq2seq model
    """
    ### Encoder
    ## Input layer
    encoder_inputs = Input(shape=(None,))
    ## Embedding Layer
    encoder_embedding_layer = Embedding(num_encoder_tokens, latent_dim, mask_zero = True)   #, mask_zero = True
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
    decoder_embedding_layer = Embedding(num_decoder_tokens, latent_dim, mask_zero = True) #, mask_zero = True
    decoder_embedding = decoder_embedding_layer(decoder_inputs)
    ## Left to right LSTM layer
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    ## Fully connected layer
    decoder_dense = Dense(num_decoder_tokens)
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

def train_seq2seq_model(model, X_train, X_valid, y_train, y_valid, epochs=100):
    
    model.compile(loss=tf.losses.softmax_cross_entropy,
                  optimizer='Nadam',
                  #metrics=['acc']
                  )
        
    train_data_generator = data_generator(X_train, y_train, batch_size)
    valid_data_generator = data_generator(X_valid, y_valid, batch_size)
    
    callbacks = [ModelCheckpoint('model.chkpt', save_best_only=True, save_weights_only=False)]
    model.fit_generator(train_data_generator,
                        validation_data=valid_data_generator,
                        epochs=epochs,
                        callbacks=callbacks,
                        verbose=2,
                        steps_per_epoch=len(X_train)/batch_size,
                        validation_steps=len(X_valid)/batch_size)

    return model




def decode_sequence(input_seq, encoder_model, decoder_model, word2id, id2word):
    # Encode the input as state vectors.
    states_value = encoder_model.predict([input_seq])
    # Generate empty target sequence of length 1.
    decoder_input = np.zeros((1, 1))
    # Populate the first character of target sequence with the start character.
    decoder_input[0, 0] = word2id['<BOS>']
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
            decoder_input[0, 0] = sampled_token_index
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

    # Read Dataset from Excel file
    dataset = read_data(train_data_path)

    conditions, outputs, dictionaries_lemanization = prepare_data(dataset)

    HL_word2id, HL_id2word = create_vocabulary(conditions, encoder_vocab_size)
    ML_word2id, ML_id2word = create_vocabulary(outputs, decoder_vocab_size)

    conditions = replace_using_dict(conditions, HL_word2id)
    outputs    = replace_using_dict(outputs, ML_word2id)

    conditions = pad_with_zero(conditions, encoder_seq_length,'pre')
    outputs    = pad_with_zero(outputs, decoder_seq_length+1,'post')

    conditions_train, conditions_valid, outputs_train, outputs_valid = train_test_split(conditions, outputs, test_size=validation_size, random_state=42)

    model, encoder_model, decoder_model = create_seq2seq_model(encoder_vocab_size, decoder_vocab_size, num_latent_dim)
    model.summary()
 
    model = train_seq2seq_model(model, conditions_train, conditions_valid, outputs_train, outputs_valid, num_epochs)
    
    for i in range(100):

        #print("input:", [HL_id2word[word] for word in CONDITIONs_train[i] if word !=0 ],'\n')

        print("input:", ''.join([ML_id2word[word] for word in outputs_train[i] if word !=0 ]),'\n')

        input_seq = conditions_train[i]
        
        decoded_sentence, decoded_senetnce_prob = decode_sequence(input_seq, encoder_model, decoder_model, ML_word2id, ML_id2word)
        print("predicted:", ''.join(decoded_sentence),'\n')
        #print("sequence: ",decoded_sentence, '\n')
        print("probabaility: ",decoded_senetnce_prob, '\n')


"""
    model_path = 'model.h5'
    model.save(model_path)
    meta_data_path = 'metadata.pickle'

    with open(meta_data_path,'wb') as f:
        pickle.dump([HL_word2id, HL_id2word, ML_word2id, ML_id2word], f)

"""



if __name__ == '__main__':
    main()



