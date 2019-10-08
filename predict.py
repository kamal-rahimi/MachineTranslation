"""
"""

import numpy as np
import argparse
import pickle
import os

from tensorflow.python import keras
from keras.models import load_model

MT_TEST_CORPUS_PATH                 = "./data/MT_test_submission.xlsx"
MT_TEST_CORPUS_PATH_WITH_PREDCITION = "./data/MT_test_submission_with_predcitions.xlsx"

MT_SEQ2SEQ_MODEL_PATH    = "./model/mt_seq2seq_model.h5"
MT_MODEL_CHECKPOINT_PATH ="./model/model.chpt"

MT_TRAINING_CORPUS_PATH = "./data/MT_training_corpus.xlsx"
MT_META_DATA_FILE_PATH   = "./model/metadata.pickle"

from tools import read_data, prepare_data, replace_using_dict, pad_with_zero, write_data

from train2 import create_seq2seq_inference_model
from train2 import encoder_seq_length, decoder_seq_length, encoder_vocab_size, decoder_vocab_size, num_latent_dim

beam_search_max_branch = 3
beam_search_max_depth = 4

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
    
    return sampled_seq, sampled_seq_prob

def decode_sequence_beam(decoder_model, decoder_input, states_value, word2id, seq_length):
    
    output_tokens, h, c = decoder_model.predict([decoder_input] + states_value)
    states_value = [h, c]
    
    seq_length += 1
    # Sample a token
    if seq_length < beam_search_max_depth:
        number_search_branch = beam_search_max_branch
    else:
        number_search_branch = 1
    
    beam_top_token_indecies = np.argsort(output_tokens[0, -1, :])[-number_search_branch:]
    sampled_seq_list = []
    sampled_seq_prob_list = []
    sampled_seq_length_list = []
    for beam in range(number_search_branch):
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
    ap.add_argument("-p",  "--path",     type=str, default=MT_TEST_CORPUS_PATH, help="Specify test data path")
    ap.add_argument("-o",  "--output",   type=str, default= MT_TEST_CORPUS_PATH_WITH_PREDCITION, help="Specify output data path")
    MT_TEST_CORPUS_PATH_WITH_PREDCITION
    args = vars(ap.parse_args())
    test_data_path = args["path"]
    test_data_output_path = args["output"]

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
    _, conditions, outputs, dictionaries_lemanization = prepare_data(qids_raw, conditions_raw, output_raw)
    
    # Replace words of qid, condition and ouput with corresponding id in dictonaries
    conditions = replace_using_dict(conditions, condition_word2id, drop_unknown=True)
    outputs    = replace_using_dict(outputs, output_word2id, drop_unknown=True)

    # Fix all sequnces length to a fixed size with padding
    conditions = pad_with_zero(conditions, encoder_seq_length,'pre')
    outputs    = pad_with_zero(outputs, decoder_seq_length+1,'post')

    outputs_predcited = [None for _ in outputs]
    for sample_index, condition in enumerate(conditions):

        input_seq = condition
        decoded_seqeunce, decoded_sequence_prob = decode_sequence(input_seq, encoder_model, decoder_model, output_word2id, output_id2word)
        
        decoded_seqeunce = replace_using_dict([decoded_seqeunce], output_id2word)
        decoded_seqeunce = replace_using_dict(decoded_seqeunce, dictionaries_lemanization[sample_index])

        decoded_seqeunce = [seq for seq in decoded_seqeunce[0] if seq != '<PAD>' and seq != '<EOS>']
        decoded_seqeunce = reversed(decoded_seqeunce)
        decoded_seqeunce = ''.join(decoded_seqeunce)

        outputs_predcited[sample_index] =  decoded_seqeunce
        if sample_index % 10 == 0:
            print("Sample index",       sample_index)
            print("QID: ",              qids_raw[sample_index])
            print("CONDITION: ",        conditions_raw[sample_index])
            print("OUTPUT: ",           output_raw[sample_index],'\n')
            print("Predicted OUTPUT: ", decoded_seqeunce, '\n\n')
    
    
    write_data(qids_raw, conditions_raw, outputs_predcited, test_data_output_path)
    print("\nSaved predictions to [{}]\n".format(test_data_output_path))

if __name__ == '__main__':
    main()

