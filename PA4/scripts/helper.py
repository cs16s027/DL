import os

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


def pad_id_sequences(source_ids, source_vocab_to_int, target_ids, target_vocab_to_int, sequence_length):
    new_source_ids = [list(reversed(sentence + [source_vocab_to_int['<pad>']] * (sequence_length - len(sentence)))) \
                      for sentence in source_ids]
    new_target_ids = [sentence + [target_vocab_to_int['<pad>']] * (sequence_length - len(sentence)) \
                      for sentence in target_ids]

    return new_source_ids, new_target_ids


def batch_data(source, target, batch_size):
    """
    Batch source and target together
    """
    for batch_i in range(0, len(source)//batch_size):
        start_i = batch_i * batch_size
        source_batch = source[start_i:start_i + batch_size]
        target_batch = target[start_i:start_i + batch_size]
        yield source_batch, target_batch
