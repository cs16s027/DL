import numpy as np
import tensorflow as tf

# HYPERPARAMS
encoder_vocab_size = 100
encoder_embedding_size = 128
encoder_hidden_size = 256

decoder_vocab_size = 100
decoder_embedding_size = 128
decoder_hidden_size = 512

learning_rate = 0.001
batch_size = 1
max_gradient_norm = 5.0

tf.reset_default_graph()
with tf.Session() as session:

    # Encoder placeholders
    encoder_input = tf.placeholder(
                        name = 'encoder_input',
                        shape = (None, None),
                        dtype = tf.int32
    )
    encoder_input_length = tf.placeholder(
                           name = 'encoder_input_length',
                           shape = (None, ),
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


    # Decoder placeholders

    decoder_input = tf.placeholder(
                    name = 'decoder_input',
                    shape = (None, None),
                    dtype = tf.int32
    )
    decoder_input_length = tf.placeholder(
                           name = 'decoder_input_length',
                           shape = (None, ),
                           dtype = tf.int32
    )
    decoder_output = tf.placeholder(
                    name = 'decoder_output',
                    shape = (None, None),
                    dtype = tf.int32
    )
    decoder_mask = tf.placeholder(
                    name = 'decoder_mask',
                    shape = (None, None),
                    dtype = tf.float32
    )

    # Decoder embedding

    decoder_embedding = tf.get_variable(
                        name = 'decoder_embedding',
                        shape = [decoder_vocab_size, decoder_embedding_size],
                        dtype = tf.float32,
                        initializer = tf.random_normal_initializer()
    )

    decoder_embedding_input = tf.nn.embedding_lookup(
                              decoder_embedding,
                              decoder_input
    )

    #### Decoder ####
    # Decoder cell
    decoder_cell = tf.nn.rnn_cell.LSTMCell(decoder_hidden_size)
    # FC layer for decoder vocab
    projection_layer = tf.layers.Dense(
                   units = decoder_vocab_size,
                   kernel_initializer = tf.random_normal_initializer()
    )
    # Helper
    helper = tf.contrib.seq2seq.TrainingHelper(
             inputs = decoder_embedding_input,
             sequence_length = decoder_input_length,
             time_major = True)
    # Basic decoder object
    decoder = tf.contrib.seq2seq.BasicDecoder(
              decoder_cell,
              helper,
              encoder_state,
              output_layer = projection_layer
    )
    # Output of decoder
    output = tf.contrib.seq2seq.dynamic_decode(
                 decoder,
                 output_time_major = True,
    )[0]
    logits = output.rnn_output

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
               labels = decoder_output,
               logits = logits)
    train_loss = (tf.reduce_sum(cross_entropy * decoder_mask) /
    batch_size)

    # Calculate and clip gradients
    params = tf.trainable_variables()
    gradients = tf.gradients(train_loss, params)
    clipped_gradients, _ = tf.clip_by_global_norm(
                           gradients, max_gradient_norm
    )

    # Optimization
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(
                  zip(clipped_gradients, params)
    )

    session.run(tf.global_variables_initializer())


    _, loss = session.run(
              [train_op, train_loss],
              {
                 encoder_input : np.int32([[1, 2, 3, 1, 0], [4, 1, 4, 0, 0]]).reshape((5, 2)),
                 encoder_input_length : np.int32([4, 1]).reshape((2, )),
                 decoder_input : np.int32([[2, 9, 5, 4, 4, 3, 0, 0], [2, 44, 2, 4, 0, 0, 0, 0]]).reshape((8, 2)),
                 decoder_input_length : np.int32([6, 3]).reshape((2, )),
                 decoder_mask : np.float32([[1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0]]).reshape((8, 2)),
                 decoder_output : np.int32([[9, 5, 4, 4, 3, 1, 0, 0], [44, 2, 4, 1, 0, 0, 0, 0]]).reshape((8, 2))
              }
    )
