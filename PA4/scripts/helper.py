import os
import numpy as np

def load_data(path):
    input_file = os.path.join(path)
    with open(input_file, "r") as f:
        data = [line.strip() for line in f.readlines()]
    return data

def extract_vocab(data, extras):

    token_dict = {}
    for line in data:
        tokens = line.strip().split()
        for token in tokens:
            token = token.lower()
            if token in extras:
                continue
            if token not in token_dict:
                token_dict[token] = 0
            token_dict[token] += 1

    freq = 1
    words = []
    for token in token_dict.keys():
        if token_dict[token] >= 1:
            words.append(token)
    for extra in extras:
        words.append(token)

    special_words = ['<pad>', '<unk>', '<go>',  '<eos>']
    set_words = set(words)
    int_to_vocab = {word_i: word for word_i, word in enumerate(special_words + list(set_words))}

    vocab_to_int = {word: word_i for word_i, word in int_to_vocab.items()}

    return int_to_vocab, vocab_to_int

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
