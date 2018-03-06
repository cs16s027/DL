time ./scripts/codes/word2vec/word2vec -train data/processed/master.txt -output models/skipgram_200_8_master.bin -cbow 0 -size 200 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 20 -binary 1 -iter 15
echo 'Done-4'
