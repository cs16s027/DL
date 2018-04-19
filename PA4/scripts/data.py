import numpy as np
import h5py

def encodeData(corpus, vocab):
    with open(corpus, 'r') as corpus_file:
        lines = [line.strip() for line in corpus_file.readlines()]
    with open(vocab, 'r') as vocab_file:
        vocab = [line.strip() for line in vocab_file.readlines()]
        vocab_dict = {}
        for index, token in enumerate(vocab):
            vocab_dict[token] = index + 2 # PAD: 0, EOS : 1

    MAX_LENGTH = 150
    num_lines = len(lines)

    encoded_lines = np.zeros((num_lines, MAX_LENGTH), dtype = np.int16)
    for index, line in enumerate(lines):
        tokens = [token.lower() for token in line.split()]
        seq = np.int16([vocab_dict[token] for token in tokens] + [1])
        encoded_lines[index, : seq.shape[0]] = seq

    return encoded_lines

if __name__ == '__main__':

    with h5py.File('data/train.h5', 'w') as f:
        encoded_lines = encodeData('data/train/train.combined', 'data/vocab/encoder.txt')
        f.create_dataset(data = encoded_lines, shape = encoded_lines.shape, dtype = encoded_lines.dtype, name = 'X')
        print 'Encoder data is of shape {}'.format(encoded_lines.shape)

        encoded_lines = encodeData('data/train/summaries.txt', 'data/vocab/decoder.txt')
        f.create_dataset(data = encoded_lines, shape = encoded_lines.shape, dtype = encoded_lines.dtype, name = 'Y')
        print 'Decoder data is of shape {}'.format(encoded_lines.shape)
