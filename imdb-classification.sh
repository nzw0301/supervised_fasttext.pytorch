#!/usr/bin/env bash

cd data/imdb
sh imdb.sh
perl ../wikifil.pl < train-data.txt > train-data.txt.normalized
perl ../wikifil.pl < test-data.txt  > test-data.txt.normalized
paste train-label train-data.txt.normalized | sed -e 's/^/__label__/g' > train.tsv
paste test-label test-data.txt.normalized | sed -e 's/^/__label__/g' > test.tsv
rm test-data.txt.normalized test-label test-data.txt train-data.txt train-data.txt.normalized train-label
cd ../../

python data/train_valid_shuffle.py  --input-file data/imdb/train.tsv --val 0.1

# unigram
python -m supervised_fasttext.main

# bi-gram
python -m supervised_fasttext.main parameters.ngram=2 parameters.word_n_gram_min_count=1

# comparison to the original implementation
git clone git@github.com:facebookresearch/fastText.git
cd fastText
make

# unigram
./fasttext supervised -input ../data/imdb/train.0.tsv -output model-imdb -dim 10 -epoch 10 -minCount 5

./fasttext test model-imdb.bin ../data/imdb/train.0.val.tsv
# N	2500
# P@1	0.872
# R@1	0.872
./fasttext test model-imdb.bin ../data/imdb/test.tsv
# N	25000
# P@1	0.883
# R@1	0.883

# bigram
./fasttext supervised -input ../data/imdb/train.0.tsv -output model-imdb-bigram -dim 10 -epoch 10 -minCount 5 -wordNgrams 2
./fasttext test model-imdb-bigram.bin ../data/imdb/train.0.val.tsv
# N	2500
# P@1	0.878
# R@1	0.878
./fasttext test model-imdb-bigram.bin ../data/imdb/test.tsv
#N	25000
# P@1	0.89
# R@1	0.89
