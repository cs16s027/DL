import numpy as np
import time

import helper
from rnn import RNN

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
enc_embedding_size = 256
dec_embedding_size = 256
# Learning Rate
learning_rate = 0.001

# Build the graph
train_graph = tf.Graph()
# Set the graph to default to ensure that it is ready for training
with train_graph.as_default():

    rnn = RNN(source_word_to_int, target_word_to_int, enc_embedding_size, dec_embedding_size,
                 enc_rnn_size, dec_rnn_size, num_layers, learning_rate, batch_size)

    # Load the model inputs
    input_data, targets, lr, target_sequence_length, max_target_sequence_length, source_sequence_length = rnn.get_model_inputs()

    # Create the training and inference logits
    training_decoder_output, inference_decoder_output = rnn.seq2seq_model(input_data,
                                                                      targets,
                                                                      target_sequence_length,
                                                                      max_target_sequence_length,
                                                                      source_sequence_length
                                                                      )

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


# Split data to training and validation sets
train_source = source_word_ids[batch_size:]
train_target = target_word_ids[batch_size:]
valid_source = source_word_ids[:batch_size]
valid_target = target_word_ids[:batch_size]
(valid_targets_batch, valid_sources_batch, valid_targets_lengths, valid_sources_lengths) = next(helper.get_batches(valid_target, valid_source, batch_size,
                           source_word_to_int['<pad>'],
                           target_word_to_int['<pad>']))

display_step = 20 # Check training loss after every 20 batches

checkpoint = "./models/best.ckpt"
with tf.Session(graph = train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(1, epochs+1):
        for batch_i, (targets_batch, sources_batch, targets_lengths, sources_lengths) in enumerate(
                helper.get_batches(train_target, train_source, batch_size,
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

with tf.Session(graph = loaded_graph) as sess:
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
