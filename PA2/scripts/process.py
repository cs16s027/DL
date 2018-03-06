import re
import sys

stopwords = [line.strip() for line in open('data/stopwords.txt', 'r').readlines()]
def processData(lines):
    data = []
    regex = re.compile('[\[\]\{\}\(\)",:;<>\/_a-zA-Z0-9!@#$%\^&\*\+=\'-]*')
    for index, line in enumerate(lines):
        if line.strip() == False:
            continue
        line = line.replace('?', '.')
        line = regex.sub('', line)
        if line.strip() == False:
            continue
        sentences = line.strip().split('.')
        for sentence in sentences:
            processed_sentence = []
            words = sentence.split(' ')
            if len(words) <= 1:
                continue
            for word in words:
                if word == '' or word in stopwords:
                    continue
                processed_sentence.append(word)
            processed_sentence = ' '.join(processed_sentence)
            data.append(processed_sentence)
    return data

def getStatistics(data):
    print 'Number of lines = {}'.format(len(data))
    word_count = 0
    for line in data:
        word_count += len(line.split(' '))
    print 'Number of words = {}'.format(word_count)

if __name__ == '__main__':
    _, fname, dest = sys.argv
    lines = [line.strip() for line in open(fname, 'r').readlines()]
    data = processData(lines)
    getStatistics(data)
    with open(dest, 'w') as f:
        for line in data:
            f.write(line + '\n')

