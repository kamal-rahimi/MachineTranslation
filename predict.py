"""
"""

import numpy as np
import argparse
import pickle
import os

from tensorflow.python import keras
from keras.models import load_model

MT_TEST_CORPUS_PATH = "./data/MT_test_submission.xlsx"

MT_ENCODER_MODEL_PATH    = "./model/encoder.h5"
MT_DECODER_MODEL_PATH    = "./model/decoder.h5"
MT_MODEL_CHECKPOINT_PATH ="./model/decoder.chpt"

MT_META_DATA_FILE_PATH   = "./model/metadata.pickle"


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
    ap.add_argument("-p", "--path",   type=str, default=MT_TEST_CORPUS_PATH, help="Specify test data path")
    args = vars(ap.parse_args())
    test_data_path = args["path"]

    # make test data path exists
    if not os.path.exists(test_data_path):
        print("\n Specified test data path [%s] does not exist\n" % test_data_path)
        return
    # Make sure an Encoder model exists
    if not os.path.exists(MT_ENCODER_MODEL_PATH):
        print("\n Trained Encoder model [%s] does not exist\n" % MT_ENCODER_MODEL_PATH)
        return
    # Make sure a Decoder model exists
    if not os.path.exists(MT_DECODER_MODEL_PATH):
        print("\n Trained decoder model [%s] does not exist\n" % MT_DECODER_MODEL_PATH)
        return
    # Make sure model metadata (word to id conversion dictionaries) exist
    if not os.path.exists(MT_META_DATA_FILE_PATH):
        print("\n Trained model metadata [%s] does not exist\n" % MT_META_DATA_FILE_PATH)
        return

    # Load model and metadata
    encoder_model = load_model(MT_ENCODER_MODEL_PATH)
    decoder_model = load_model(MT_DECODER_MODEL_PATH)
    with open(MT_META_DATA_FILE_PATH,'rb') as f:
        [condition_word2id,condition_id2word,
         output_word2id, output_id2word,
         encoder_vocab_size, decoder_vocab_size,
         num_latent_dim] = pickle.load(f)


if __name__ == '__main__':
    main()

