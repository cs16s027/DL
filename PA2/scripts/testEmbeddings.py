import gensim
import sys

_, testfile, modelname = sys.argv
resultsfile = '{}.txt'.format(modelname)
model = gensim.models.Word2Vec.load(modelname)
lines = [line.strip().split(' ') for line in open(testfile, 'r').readlines()]
with open(resultsfile, 'w') as f:
    for line in lines:
        a, aa, b, bb = line[:4]
        query = ' : '.join([a, aa]) + ' :: ' + ' : '.join([b, bb])
        try:
            options = model.wv.most_similar(positive = [aa, b], negative = [a])
            answer = ','.join([option[0] for option in options[:5]])
        except KeyError:
            answer = 'OoV'
        string = 'Query = {}, Answer = {}'.format(query, answer)
        f.write(string + '\n')

