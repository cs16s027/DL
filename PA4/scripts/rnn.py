import numpy as np
import time

import helper

import tensorflow as tf
from tensorflow.python.layers.core import Dense

class RNN:
    def __init__(self, source_word_to_int, target_word_to_int, enc_embedding_size, dec_embedding_size,
                 enc_rnn_size, dec_rnn_size, num_layers, lr, batch_size):
        self.source_word_to_int = source_word_to_int
        self.source_vocab_size = len(self.source_word_to_int)
        self.target_word_to_int = target_word_to_int
        self.target_vocab_size = len(self.target_word_to_int)
        self.enc_embedding_size = enc_embedding_size
        self.dec_embedding_size = dec_embedding_size
        self.enc_rnn_size = enc_rnn_size
        self.dec_rnn_size = dec_rnn_size
        self.num_layers = num_layers
        self.lr = lr
        self.batch_size = batch_size

    def get_model_inputs(self):
        input_data = tf.placeholder(tf.int32, [None, None], name='input')
        targets = tf.placeholder(tf.int32, [None, None], name='targets')
        lr = tf.placeholder(tf.float32, name='learning_rate')

        target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length')
        max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_len')
        source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')

        return input_data, targets, lr, target_sequence_length, max_target_sequence_length, source_sequence_length

    def encoding_layer(self, input_data, source_sequence_length):

        # Encoder embedding
        enc_embed_input = tf.contrib.layers.embed_sequence(input_data, self.source_vocab_size, self.enc_embedding_size)

        ###########
        # Encoder cell
        encoder_cell = tf.contrib.rnn.LSTMCell(self.enc_rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2), state_is_tuple = True)
        # Biderectional LSTM layer
        ((encoder_fw_output, encoder_bw_output),
        (encoder_fw_state, encoder_bw_state)) = tf.nn.bidirectional_dynamic_rnn(
                                                encoder_cell,
                                                encoder_cell,
                                                enc_embed_input,
                                                sequence_length = source_sequence_length,
                                                dtype = tf.float32
        )

        # Encoder output
        enc_output = tf.concat((encoder_fw_output, encoder_bw_output), 2)
        # Encoder state
        encoder_state_c = tf.concat((encoder_fw_state.c, encoder_bw_state.c), 1)
        encoder_state_h = tf.concat((encoder_fw_state.h, encoder_bw_state.h), 1)
        enc_state = tf.contrib.rnn.LSTMStateTuple(
                        c = encoder_state_c,
                        h = encoder_state_h,
        )
        return enc_output, tuple([enc_state])

    # Process the input we'll feed to the decoder
    def process_decoder_input(self, target_data):
        '''Remove the last word id from each batch and concat the <GO> to the begining of each batch'''
        ending = tf.strided_slice(target_data, [0, 0], [self.batch_size, -1], [1, 1])
        dec_input = tf.concat([tf.fill([self.batch_size, 1], self.target_word_to_int['<go>']), ending], 1)
        return dec_input

    def decoding_layer(self, target_sequence_length, max_target_sequence_length, enc_state, dec_input):
        # 1. Decoder Embedding
        dec_embeddings = tf.Variable(tf.random_uniform([self.target_vocab_size, self.dec_embedding_size]))
        dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)

        # 2. Construct the decoder cell
        def make_cell(rnn_size):
            dec_cell = tf.contrib.rnn.LSTMCell(rnn_size,
                                               initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            return dec_cell

        dec_cell = tf.contrib.rnn.MultiRNNCell([make_cell(self.dec_rnn_size) for _ in range(self.num_layers)])

        # 3. Dense layer to translate the decoder's output at each time
        # step into a choice from the target vocabulary
        output_layer = Dense(self.target_vocab_size,
                             kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))

        # 4. Set up a training decoder and an inference decoder
        # Training Decoder
        with tf.variable_scope("decode"):

            # Helper for the training process. Used by BasicDecoder to read inputs.
            training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
                                                                sequence_length=target_sequence_length,
                                                                time_major=False)

            # Basic decoder
            training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                               training_helper,
                                                               enc_state,
                                                               output_layer)

            # Perform dynamic decoding using the decoder
            training_decoder_output = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                           impute_finished=True,
                                                                           maximum_iterations=max_target_sequence_length)[0]
        # 5. Inference Decoder
        # Reuses the same parameters trained by the training process
        with tf.variable_scope("decode", reuse=True):
            start_tokens = tf.tile(tf.constant([self.target_word_to_int['<go>']], dtype=tf.int32), [self.batch_size], name='start_tokens')

            # Helper for the inference process.
            inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings,
                                                                    start_tokens,
                                                                    self.target_word_to_int['<eos>'])

            # Basic decoder
            inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                            inference_helper,
                                                            enc_state,
                                                            output_layer)

            # Perform dynamic decoding using the decoder
            inference_decoder_output = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                                impute_finished=True,
                                                                maximum_iterations=max_target_sequence_length)[0]



        return training_decoder_output, inference_decoder_output

    def seq2seq_model(self, input_data, targets, target_sequence_length,
                      max_target_sequence_length, source_sequence_length):

        # Pass the input data through the encoder. We'll ignore the encoder output, but use the state
        _, enc_state = self.encoding_layer(input_data, source_sequence_length)

        # Prepare the target sequences we'll feed to the decoder in training mode
        dec_input = self.process_decoder_input(targets)

        # Pass encoder state and decoder inputs to the decoders
        training_decoder_output, inference_decoder_output = self.decoding_layer(target_sequence_length,
                                                                           max_target_sequence_length,
                                                                           enc_state,
                                                                           dec_input)

        return training_decoder_output, inference_decoder_output
