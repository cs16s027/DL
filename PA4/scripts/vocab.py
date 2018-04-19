import numpy as np

def getVocabFile(corpus, vocab_path, extras):
    lines = [line.strip() for line in open(corpus, 'r').readlines()]

    token_dict = {}
    sizes = []

    for line in lines:
        tokens = line.split()
        for token in tokens:
            token = token.lower()
            if token in extras:
                continue
            if token not in token_dict:
                token_dict[token] = 0
            token_dict[token] += 1
        sizes.append(len(tokens))

    print 'Miminum sequence length = {}, Maximum sequence length = {}'.format(min(sizes), max(sizes))
    print 'Number of tokens = {}'.format(len(token_dict.keys()))

    freq = 1
    vocab_size = 0
    with open(vocab_path, 'w') as vocab_file:
        for token in token_dict.keys():
            if token_dict[token] >= 1:
                vocab_file.write('{}\n'.format(token))
                vocab_size += 1
        for extra in extras:
            vocab_file.write('{}\n'.format(extra))
            vocab_size += 1

    return vocab_size

if __name__ == '__main__':
    # TODO : remove repetitions
    twodigits = [str(word) for word in range(-99, 100)]
    fourdigits = [str(word) for word in np.arange(-7000, 7000, 100)]
    vocab_size = getVocabFile('data/train/train.combined', 'data/vocab/encoder.txt', twodigits)
    print 'Vocabulary for endocder : {} tokens'.format(vocab_size)
    # Decoder-additions: [6am, numbers from -100-100]
    vocab_size = getVocabFile('data/train/summaries.txt', 'data/vocab/decoder.txt',\
                              ['6am', '12pm', '12am'] + twodigits + fourdigits)
    print 'Vocabulary for decoder : {} tokens'.format(vocab_size)
