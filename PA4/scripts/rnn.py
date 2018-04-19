import numpy as np
import tensorflow as tf

# HYPERPARAMS
encoder_vocab_size = 255 + 1 # tokens + (padding?)
encoder_embedding_size = 128
encoder_hidden_size = 256

tf.reset_default_graph()
with tf.Session() as session:

    encoder_input = tf.placeholder(
                        name = 'encoder_input',
                        shape = (None, None),
                        dtype = tf.int32
    )
    encoder_input_length = tf.placeholder(
                            name = 'encoder_inputs_length',
                            shape = (None, ),
                            dtype = tf.int32
    )
    decoder_target = tf.placeholder(
                      name = 'decoder_target',
                      shape = (None, None),
                      dtype = tf.int32
    )

    # Encoder embedding

    encoder_embedding = tf.get_variable(
                        name = 'encoder_embedding',
                        shape = [encoder_vocab_size, encoder_embedding_size],
                        dtype = tf.float32,
                        initializer = tf.random_normal_initializer()
    )

    encoder_embedding_input = tf.nn.embedding_lookup(
                              encoder_embedding,
                              encoder_input
    )

    # Encoder cell
    encoder_cell = tf.nn.rnn_cell.LSTMCell(encoder_hidden_size)
    # Biderectional LSTM layer
    ((encoder_fw_output, encoder_bw_output),
    (encoder_fw_state, encoder_bw_state)) = tf.nn.bidirectional_dynamic_rnn(
                                            cell_fw = encoder_cell,
                                            cell_bw = encoder_cell,
                                            inputs = encoder_embedding_input,
                                            sequence_length = encoder_input_length,
                                            time_major = True,
                                            dtype = tf.float32
    )

    # Encoder output
    encoder_output = tf.concat((encoder_fw_output, encoder_bw_output), 2)
    # Encoder state
    encoder_state_c = tf.concat((encoder_fw_state.c, encoder_bw_state.c), 1)
    encoder_state_h = tf.concat((encoder_fw_state.h, encoder_bw_state.h), 1)
    encoder_state = tf.nn.rnn_cell.LSTMStateTuple(
                    c = encoder_state_c,
                    h = encoder_state_h
    )

    session.run(tf.global_variables_initializer())
