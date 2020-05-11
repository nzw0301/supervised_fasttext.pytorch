## Requirements

- `pytorch` >= 1.4.0
- `hydra`: 0.11.3
- `gensim`: 3.8.0
- `numpy`: 1.18.1
- `Optuna`: [Master branch](https://github.com/optuna/optuna/tree/2f896b2b51a99a6ebe7206310c5b9d3cb88fc782)
- [`tokenizer`](https://github.com/nzw0301/tokenizer)

## Usage

```bash
# install dependencies
$ pip install git+https://github.com/nzw0301/tokenizer
$ conda install -y numpy gensim
$ pip install hydra-core

$ python -m supervised_fasttext.main --cfg job
dataset:
  path: data/imdb/
  test_fname: test.tsv
  train_fname: train.0.tsv
  val_fname: train.0.val.tsv
optuna:
  lr_max: 0.5
  lr_min: 1.0e-05
  n_jobs: 1
  num_trials: 30
parameters:
  dim: 10
  epochs: 10
  freeze: 1
  gpu_id: 0
  initialize_oov: mean
  label_separator: "\t"
  logging_file: result.json
  lr_update_rate: 100
  metric: acc
  min_count: 5
  ngram: 1
  patience: 5
  pooling: mean
  pre_trained: null
  replace_OOV: 0
  seed: 0
  word_n_gram_min_count: 10
```

Please check [`conf/config.yaml`](./conf/config.yaml) for the details.

## Demo: Binary classification on IMDb dataset

```bash
cd data/imdb
sh imdb.sh
perl ../wikifil.pl < train-data.txt > train-data.txt.normalized
perl ../wikifil.pl < test-data.txt  > test-data.txt.normalized
paste train-label train-data.txt.normalized | sed -e 's/^/__label__/g' > train.tsv
paste test-label test-data.txt.normalized | sed -e 's/^/__label__/g' > test.tsv
rm test-data.txt.normalized test-label test-data.txt train-data.txt train-data.txt.normalized train-label
cd ../../

# train val split
python data/train_valid_shuffle.py  --input-file data/imdb/train.tsv

# uni-gram
# learning rate is optimised by using optuna.
python -m supervised_fasttext.main parameters.min_count=10

# bi-gram
python -m supervised_fasttext.main parameters.ngram=2 parameters.word_n_gram_min_count=5 parameters.min_count=10

```
