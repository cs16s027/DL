import numpy as np
import time

import helper

import tensorflow as tf
from tensorflow.python.layers.core import Dense

source_path = 'data/train/train.combined'
target_path = 'data/train/summaries.txt'

source_sentences = helper.load_data(source_path)
target_sentences = helper.load_data(target_path)

# Build int2word and word2int dicts
twodigits = [str(word) for word in range(-99, 100)]
fourdigits = [str(word) for word in np.arange(-7000, 7000, 100)]
source_int_to_word, source_word_to_int = helper.extract_vocab(source_sentences, twodigits)
target_int_to_word, target_word_to_int = helper.extract_vocab(target_sentences, ['6am', '12pm', '12am'] + twodigits + fourdigits)

# Convert words to ids
source_word_ids = [[source_word_to_int.get(word.lower(), source_word_to_int['<unk>']) for word in line.split()] for line in source_sentences]
target_word_ids = [[target_word_to_int.get(word.lower(), target_word_to_int['<unk>']) for word in line.split()] + [target_word_to_int['<eos>']] for line in target_sentences]

'''
print("Example source sequence")
print(source_word_ids[:3])
print("\n")
print("Example target sequence")
print(target_word_ids[:3])
'''
# Number of Epochs
epochs = 1
# Batch Size
batch_size = 16
# RNN Size
enc_rnn_size = 256
dec_rnn_size = 512
# Number of Layers
num_layers = 1
# Embedding Size
encoding_embedding_size = 256
decoding_embedding_size = 256
# Learning Rate
learning_rate = 0.001

def get_model_inputs():
    input_data = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    lr = tf.placeholder(tf.float32, name='learning_rate')

    target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length')
    max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_len')
    source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')

    return input_data, targets, lr, target_sequence_length, max_target_sequence_length, source_sequence_length


def encoding_layer(input_data, rnn_size, num_layers,
                   source_sequence_length, source_vocab_size,
                   encoding_embedding_size):


    # Encoder embedding
    enc_embed_input = tf.contrib.layers.embed_sequence(input_data, source_vocab_size, encoding_embedding_size)

    '''
    # RNN cell
    def make_cell(rnn_size):
        enc_cell = tf.contrib.rnn.LSTMCell(rnn_size,
                                           initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return enc_cell

    enc_cell = tf.contrib.rnn.MultiRNNCell([make_cell(rnn_size) for _ in range(num_layers)])

    enc_output, enc_state = tf.nn.dynamic_rnn(enc_cell, enc_embed_input, sequence_length=source_sequence_length, dtype=tf.float32)
    '''
    #'''
    ###########
    # Encoder cell
    encoder_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2), state_is_tuple = True)
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

    ##############
    #'''
    return enc_output, tuple([enc_state])

# Process the input we'll feed to the decoder
def process_decoder_input(target_data, vocab_to_int, batch_size):
    '''Remove the last word id from each batch and concat the <GO> to the begining of each batch'''
    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    dec_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<go>']), ending], 1)

    return dec_input

def decoding_layer(target_word_to_int, decoding_embedding_size, num_layers, rnn_size,
                   target_sequence_length, max_target_sequence_length, enc_state, dec_input):
    # 1. Decoder Embedding
    target_vocab_size = len(target_word_to_int)
    dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)

    # 2. Construct the decoder cell
    def make_cell(rnn_size):
        dec_cell = tf.contrib.rnn.LSTMCell(rnn_size,
                                           initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return dec_cell

    dec_cell = tf.contrib.rnn.MultiRNNCell([make_cell(rnn_size) for _ in range(num_layers)])

    # 3. Dense layer to translate the decoder's output at each time
    # step into a choice from the target vocabulary
    output_layer = Dense(target_vocab_size,
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
        start_tokens = tf.tile(tf.constant([target_word_to_int['<go>']], dtype=tf.int32), [batch_size], name='start_tokens')

        # Helper for the inference process.
        inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings,
                                                                start_tokens,
                                                                target_word_to_int['<eos>'])

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

def seq2seq_model(input_data, targets, lr, target_sequence_length,
                  max_target_sequence_length, source_sequence_length,
                  source_vocab_size, target_vocab_size,
                  enc_embedding_size, dec_embedding_size,
                  enc_rnn_size, dec_rnn_size, num_layers):

    # Pass the input data through the encoder. We'll ignore the encoder output, but use the state
    _, enc_state = encoding_layer(input_data,
                                  enc_rnn_size,
                                  num_layers,
                                  source_sequence_length,
                                  source_vocab_size,
                                  encoding_embedding_size)

    print (enc_state)

    # Prepare the target sequences we'll feed to the decoder in training mode
    dec_input = process_decoder_input(targets, target_word_to_int, batch_size)

    # Pass encoder state and decoder inputs to the decoders
    training_decoder_output, inference_decoder_output = decoding_layer(target_word_to_int,
                                                                       decoding_embedding_size,
                                                                       num_layers,
                                                                       dec_rnn_size,
                                                                       target_sequence_length,
                                                                       max_target_sequence_length,
                                                                       enc_state,
                                                                       dec_input)

    return training_decoder_output, inference_decoder_output

# Build the graph
train_graph = tf.Graph()
# Set the graph to default to ensure that it is ready for training
with train_graph.as_default():

    # Load the model inputs
    input_data, targets, lr, target_sequence_length, max_target_sequence_length, source_sequence_length = get_model_inputs()

    # Create the training and inference logits
    training_decoder_output, inference_decoder_output = seq2seq_model(input_data,
                                                                      targets,
                                                                      lr,
                                                                      target_sequence_length,
                                                                      max_target_sequence_length,
                                                                      source_sequence_length,
                                                                      len(source_word_to_int),
                                                                      len(target_word_to_int),
                                                                      encoding_embedding_size,
                                                                      decoding_embedding_size,
                                                                      enc_rnn_size,
                                                                      dec_rnn_size,
                                                                      num_layers)

    # Create tensors for the training logits and inference logits
    training_logits = tf.identity(training_decoder_output.rnn_output, 'logits')
    inference_logits = tf.identity(inference_decoder_output.sample_id, name='predictions')

    # Create the weights for sequence_loss
    masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')

    with tf.name_scope("optimization"):

        # Loss function
        cost = tf.contrib.seq2seq.sequence_loss(
            training_logits,
            targets,
            masks)

        # Optimizer
        optimizer = tf.train.AdamOptimizer(lr)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)

def pad_sentence_batch(sentence_batch, pad_int):
    """Pad sentences with <pad> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]

def get_batches(targets, sources, batch_size, source_pad_int, target_pad_int):
    """Batch targets, sources, and the lengths of their sentences together"""
    for batch_i in range(0, len(sources)//batch_size):
        start_i = batch_i * batch_size
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))

        # Need the lengths for the _lengths parameters
        pad_targets_lengths = []
        for target in pad_targets_batch:
            pad_targets_lengths.append(len(target))

        pad_source_lengths = []
        for source in pad_sources_batch:
            pad_source_lengths.append(len(source))

        yield pad_targets_batch, pad_sources_batch, pad_targets_lengths, pad_source_lengths

# Split data to training and validation sets
train_source = source_word_ids[batch_size:]
train_target = target_word_ids[batch_size:]
valid_source = source_word_ids[:batch_size]
valid_target = target_word_ids[:batch_size]
(valid_targets_batch, valid_sources_batch, valid_targets_lengths, valid_sources_lengths) = next(get_batches(valid_target, valid_source, batch_size,
                           source_word_to_int['<pad>'],
                           target_word_to_int['<pad>']))

display_step = 20 # Check training loss after every 20 batches

checkpoint = "./models/best.ckpt"
with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(1, epochs+1):
        for batch_i, (targets_batch, sources_batch, targets_lengths, sources_lengths) in enumerate(
                get_batches(train_target, train_source, batch_size,
                           source_word_to_int['<pad>'],
                           target_word_to_int['<pad>'])):

            # Training step
            _, loss = sess.run(
                [train_op, cost],
                {input_data: sources_batch,
                 targets: targets_batch,
                 lr: learning_rate,
                 target_sequence_length: targets_lengths,
                 source_sequence_length: sources_lengths})

            # Debug message updating us on the status of the training
            if batch_i % display_step == 0 and batch_i > 0:

                # Calculate validation cost
                validation_loss = sess.run(
                [cost],
                {input_data: valid_sources_batch,
                 targets: valid_targets_batch,
                 lr: learning_rate,
                 target_sequence_length: valid_targets_lengths,
                 source_sequence_length: valid_sources_lengths})

                print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}  - Validation loss: {:>6.3f}'
                      .format(epoch_i,
                              epochs,
                              batch_i,
                              len(train_source) // batch_size,
                              loss,
                              validation_loss[0]))




    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, checkpoint)
    print('Model Trained and Saved')

def source_to_seq(text):
    '''Prepare the text for the model'''
    sequence_length = 40
    return [source_word_to_int.get(word, source_word_to_int['<unk>']) for word in text]+ [source_word_to_int['<pad>']]*(sequence_length-len(text))


input_sentence = 'temperature time 6-21 min 33 mean 40 max 45 windChill time 6-21 min 28 mean 35 max 41 windSpeed time 6-21 min 3 mean 6 max 9 mode-bucket-0-20-2 0-10 windDir time 6-21 mode W gust time 6-21 min 0 mean 0 max 0 skyCover time 6-21 mode-bucket-0-100-4 0-25 skyCover time 6-9 mode-bucket-0-100-4 0-25 skyCover time 6-13 mode-bucket-0-100-4 0-25 skyCover time 9-21 mode-bucket-0-100-4 0-25 skyCover time 13-21 mode-bucket-0-100-4 0-25 precipPotential time 6-21 min 1 mean 2 max 7'.split()
# Sunny , with a high near 46 . West wind between 6 and 9 mph .
text = source_to_seq(input_sentence)

checkpoint = "./models/best.ckpt"

loaded_graph = tf.Graph()

with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(checkpoint + '.meta')
    loader.restore(sess, checkpoint)

    input_data = loaded_graph.get_tensor_by_name('input:0')
    logits = loaded_graph.get_tensor_by_name('predictions:0')
    source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_length:0')
    target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')

    #Multiply by batch_size to match the model's input parameters
    answer_logits = sess.run(logits, {input_data: [text]*batch_size,
                                      target_sequence_length: [len(text)]*batch_size,
                                      source_sequence_length: [len(text)]*batch_size})[0]


pad = source_word_to_int["<pad>"]

print('Original Text:', input_sentence)

print('\nSource')
print('  Word Ids:    {}'.format([i for i in text]))
print('  Input Words: {}'.format(" ".join([source_int_to_word[i] for i in text])))

print('\nTarget')
print('  Word Ids:       {}'.format([i for i in answer_logits if i != pad]))
print('  Response Words: {}'.format(" ".join([target_int_to_word[i] for i in answer_logits if i != pad])))
