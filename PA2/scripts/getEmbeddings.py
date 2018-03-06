import gensim
import sys

class Sentences(object):
	def __init__(self, filename):
		self.filename = filename
 
	def __iter__(self):
		for line in open(self.filename, 'r'):
			yield line.split()

_, corpusname = sys.argv
sentences = Sentences(corpusname)
 
size = 200
sg = 0
hs = 1
window = 5
modelname = 'skipgram' if sg == 1 else 'cbow'
hsname = 'hierarchical' if hs == 1 else 'negative'

savename = '{}.{}.{}.{}.{}.bin'.format(corpusname.split('.txt')[0].split('/')[-1], modelname, size, window, hsname)

model = gensim.models.Word2Vec(sentences, min_count = 5, size = size, workers = 4, sg = sg, window = window, hs = 1)
print savename
model.save('models/{}'.format(savename))

