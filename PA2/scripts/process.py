import sys

def processData(lines):
    print 'Number of paragraphs = {}'.format(len(lines))
    sentence_count, word_count = 0, 0
    for index, line in enumerate(lines):
        sentences = line.split('.')
        for sentence in sentences:
            words = sentence.split(' ')
            word_count += len(words)
        sentence_count += len(sentences)
    print 'Number of sentences = {}'.format(sentence_count)
    print 'Number of words = {}'.format(word_count)

def cleanData():
    pass

if __name__ == '__main__':
    _, fname = sys.argv
    lines = [line.strip() for line in open(fname, 'r').readlines()]
    processData(lines)
