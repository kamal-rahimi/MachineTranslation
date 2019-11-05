""" predict.py
This file reads and preproces the test dataset. Loades a trained seq2seq model
and predict the iput for each sample in test dataset. It writes prediction results
to a file and print to shell
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


### Import required packages
import numpy as np
import pickle
import os

from tensorflow.python import keras
from keras.models import load_model

MT_TEST_CORPUS_PATH                 = "./data/MT_test.xlsx"
MT_TEST_CORPUS_PATH_WITH_PREDCITION = "./data/MT_test_with_predcitions.xlsx"

## Import helper functions and constants
from tools import read_data, prepare_data, replace_using_dict, pad_with_zero, write_data, log_to_shell
from train import MT_SEQ2SEQ_MODEL_PATH, MT_META_DATA_FILE_PATH
from train import create_seq2seq_inference_model
from train import encoder_seq_length, decoder_seq_length, encoder_vocab_size, decoder_vocab_size, num_latent_dim

## Specify prediction paramets
# Beam serahc paramets to predict the most likely target sequence
beam_search_max_branch = 3 # Maximum number of branch at each time step for beam search
beam_search_max_depth = 4  # Maimum sequnce step to branch in beam search

def decode_sequence(input_seq, encoder_model, decoder_model, word2id, id2word):
    """ Decodes an input sequnce uing the enoder and decoder model of trained seq2seq model
    Beam serahc algorithm is used to find a decoded sequnce with highed liklihood.
    Args:
        input_seq: Input sequnce
        encoder_model: Enoder model of the seq2seq model (Keras)
        decoder_model: Decoder model of the seq2seq model (keras)
        word2id: Python dictionary to conver word to id
        id2word: Python dictionary to conver id to word
    Returns:
        decoded_seq: Decoded sequence predicted by the model
        decoded_seq_prob: The linkleihood of the predicted sequnce by the model
    """
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
    decoded_seq, decoded_seq_prob, _ = decode_sequence_beam(decoder_model, decoder_input, states_value, word2id, seq_length)
    
    return decoded_seq, decoded_seq_prob

def decode_sequence_beam(decoder_model, decoder_input, states_value, word2id, seq_length):
    """ This function decodes a sequnce using beam search. That is in each step of
    decoding, search space tree is branched based on the number of specified number_search_branch
    parameter for maximum depth of beam_search_max_depth
    The beam search algorithm is implemented using a recursive call of this function itself.
    Args:
        decoder_model: Decoder model of the seq2seq model (keras)
        decoder_input: The input to decoder in each step
        states_value: The previous state values input
        word2id: Python dictionary to conver word to id
        seq_length: Current Sequence length from the begining of sequnce (used to control beam search depth)
    Returns:
        sampled_seq: Sampled sequence upto this step (from end to this step, reursive function call)
        sampled_seq_prob: The linklihood of sampled sequnce upto this step (from end to this step)
        sampled_seq_length:The sampled Sequnce length to the the end of sequnce (from end to this step)
    """
    ## Get probabilitis of next word in the sequnce and state values
    output_tokens, h, c = decoder_model.predict([decoder_input] + states_value)
    
    ## Update states
    states_value = [h, c]
    
    ## Increment sequence length
    seq_length += 1
    
    ## Choose number of branches to split tree for beam search
    # To avoid too many searches will branch up to beam_search_max_depth sequnce length
    if seq_length < beam_search_max_depth:
        number_search_branch = beam_search_max_branch
    else:
        number_search_branch = 1
    
    ## Choose tokens with highest probabities
    beam_top_token_indecies = np.argsort(output_tokens[0, -1, :])[-number_search_branch:]
    
    sampled_seq_list = []        # List of sampled sequnce from end to this step
    sampled_seq_prob_list = []   # List of liklihood for th sampled sequnce from end to this step
    sampled_seq_length_list = [] # List of lenght for sampled sequnce from end to this step
    ## Split the search space for sequnce to differenr barnches
    for beam in range(number_search_branch):
        sampled_token_index = beam_top_token_indecies[beam]
        sampled_token_prob  = output_tokens[0, -1, sampled_token_index]
        if sampled_token_index == word2id['<EOS>'] or seq_length == decoder_seq_length:
            return [sampled_token_index], sampled_token_prob, 0.00000001 # smalle number to avoid divde by zero
        else:
            ## Update the target sequence (of length 1).
            decoder_input = np.zeros((1, 1, decoder_vocab_size))
            decoder_input[0, 0, sampled_token_index] = 1
            
            ## recusrive call to decode_sequence_beam function itself to find
            ## best sequnce from this point to the end
            sampled_seq, sampled_seq_prob, sampled_seq_length = decode_sequence_beam(decoder_model, decoder_input, states_value, word2id, seq_length)
            
            ## Save the sampled sequnce (This a sampled sequnce from end to this point)
            sampled_seq.append(sampled_token_index)

            ## calculate the Sequnce probabity from end to this step
            sampled_seq_prob *= sampled_token_prob
            
            ## Append the sampled sequnce to a list
            sampled_seq_list.append(sampled_seq)
            ## Append the sampled sequnce probability to a list
            sampled_seq_prob_list.append(sampled_seq_prob)
            ## Append the sampled sequnce length to a list
            sampled_seq_length_list.append(sampled_seq_length)
    
    ## Claculate weighted probabity of list sequnces (bracnh beams) from end pof sequnce to this step
    # The sequnce probabities are ajusted for lenght, thus model will not prefer shorter length
    # sequnces. This is required because longer sequnces are generally have lower probability 
    weighted_prob = np.log(np.array(sampled_seq_prob_list))/np.array(sampled_seq_length_list)
    
    ## Choose a sequnce from beam branch with the highest probability 
    best_beam = np.argmax(weighted_prob)
    
    return sampled_seq_list[best_beam], sampled_seq_prob_list[best_beam], sampled_seq_length_list[best_beam]+1


def main():
    """ The main steps to predict an output sequnce using the seq2seq model:
    1. Read test dataset
    2. Preproces each sequnce (create standarized sequnces)
        a. Change QID and CONDITION text to lowercase
        b. split QID and CONDITION text into tokens (words)
        c. Replace QID tokens in each sample with standrized tokens (i.e., <QID0>, <QID1>, ...)
        d. Replace digit tokens in each sample with standarized tokens (i.e., <DGT0>, <DGT1>, ...)
        e. Create standardization dictionary for each sample
        f. Add special tokens <BOS> and <EOS> to the begining and end of each sequence
    3. Replace condition sequnce tokens with an integre id usng the encoder_word2id dictionary
    4. Pad condition sequnce with zero to create a fixed size input sequnce
        a. Input sequnce is pre-padded with zero
    5. Extract Encoder and Decoder parts of saved seq2seq model
    6. Use a beam search algorithm to predict the output sequnce
    7. Reverse predicted output sequnce to words using the decoder_id2word dictionary 
    8. Revrese Digit and QID standardization from the predicted output
    9. Save the precited outputs to a file
    """

    # Test data path
    test_data_path = MT_TEST_CORPUS_PATH

    # Output data path
    test_data_output_path = MT_TEST_CORPUS_PATH_WITH_PREDCITION

    # Make sure an Encoder model exists
    if not os.path.exists(MT_SEQ2SEQ_MODEL_PATH):
        print("\n The seq2seq model [%s] does not exist\n" % MT_SEQ2SEQ_MODEL_PATH)
        return

    # Load model and metadata
    model = load_model(MT_SEQ2SEQ_MODEL_PATH)

    with open(MT_META_DATA_FILE_PATH,'rb') as f:
        [condition_word2id, condition_id2word, output_word2id, output_id2word] = pickle.load(f)

    print("\nLoaded a trained seq2seq model from [{}]\n".format(MT_SEQ2SEQ_MODEL_PATH))

    encoder_model, decoder_model = create_seq2seq_inference_model(model, num_latent_dim)
    
    #test_data_path = MT_TRAINING_CORPUS_PATH
    
    #Read dataset from Excel file
    qids_raw, conditions_raw, output_raw = read_data(test_data_path)
    print("\nLoaded test dataset from [{}]\n".format(test_data_path))

    # Preprocess the raw input text data
    _, conditions, _, dictionaries_lemanization = prepare_data(qids_raw, conditions_raw, output_raw)
    
    # Replace words of qid, condition and ouput with corresponding id in dictonaries
    conditions = replace_using_dict(conditions, condition_word2id, drop_unknown=True)

    # Fix all sequnces length to a fixed size with padding
    conditions = pad_with_zero(conditions, encoder_seq_length,'pre')

    outputs_predcited = [None for _ in conditions]
    for sample_index, condition in enumerate(conditions):

        input_seq = condition
        decoded_seqeunce, _ = decode_sequence(input_seq, encoder_model, decoder_model, output_word2id, output_id2word)
        
        decoded_seqeunce = replace_using_dict([decoded_seqeunce], output_id2word)
        decoded_seqeunce = replace_using_dict(decoded_seqeunce, dictionaries_lemanization[sample_index])

        decoded_seqeunce = [seq for seq in decoded_seqeunce[0] if seq != '<PAD>' and seq != '<EOS>'\
                                                                and '<QID' not in seq and '<DGT' not in seq]
        decoded_seqeunce = reversed(decoded_seqeunce)
        decoded_seqeunce = ''.join(decoded_seqeunce)

        outputs_predcited[sample_index] = decoded_seqeunce

        if sample_index % 10 == 0:
            log_to_shell(sample_index, qids_raw[sample_index],
                           conditions_raw[sample_index], output_raw[sample_index],
                           decoded_seqeunce )
        
    write_data(qids_raw, conditions_raw, outputs_predcited, test_data_output_path)
    print("\nSaved predictions to [{}]\n".format(test_data_output_path))

if __name__ == '__main__':
    main()

