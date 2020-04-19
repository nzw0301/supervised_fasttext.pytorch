## Requirements

- `pytorch`: 1.4.0
- `hydra`: 0.11.3
- `gensim`: 3.8.0
- `numpy`: 1.18.1
- [`tokenizer`](https://github.com/nzw0301/tokenizer)

## Usage

```bash
# install dependencies
$ pip install git+https://github.com/nzw0301/tokenizer

$ python -m supervised_fasttext.main --cfg job
dataset:
  path: ../../../data/imdb/
  test_fname: test.tsv
  train_fname: train.0.tsv
  val_fname: train.0.val.tsv
parameters:
  dim: 10
  epochs: 10
  freeze: 1
  gpu_id: 0
  initialize_oov: mean
  label_separator: "\t"
  logging_file: result.json
  lr: 0.1
  lr_update_rate: 100
  metric: loss
  min_count: 5
  ngram: 1
  patience: 5
  pooling: mean
  pre_trained: null
  replace_OOV: 0
  seed: 7
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
python -m supervised_fasttext.main
Use cpu
[2020-04-19 18:40:55,368][__main__][INFO] - Use cpu

#training_data: 22500, #val_data: 2500, #test_data: 25000
[2020-04-19 18:40:55,368][__main__][INFO] - #training_data: 22500, #val_data: 2500, #test_data: 25000

In training data, the size of word vocab: 27353 ngram vocab: 0, total: 27353
[2020-04-19 18:40:55,369][__main__][INFO] - In training data, the size of word vocab: 27353 ngram vocab: 0, total: 27353

Progress: 0.1000000 Avg. train loss: 0.6202, train acc: 62.5%, Avg. val loss: 0.4859, val acc: 78.2%[2020-04-19 18:42:34,869][__main__][INFO] -
Progress: 0.2000000 Avg. train loss: 0.3983, train acc: 82.6%, Avg. val loss: 0.3964, val acc: 83.2%[2020-04-19 18:44:06,242][__main__][INFO] -
Progress: 0.3000000 Avg. train loss: 0.3265, train acc: 86.4%, Avg. val loss: 0.3570, val acc: 85.2%[2020-04-19 18:45:37,699][__main__][INFO] -
Progress: 0.4000000 Avg. train loss: 0.2899, train acc: 88.1%, Avg. val loss: 0.3442, val acc: 85.2%[2020-04-19 18:47:36,103][__main__][INFO] -
Progress: 0.5000000 Avg. train loss: 0.2640, train acc: 89.3%, Avg. val loss: 0.3329, val acc: 86.2%[2020-04-19 18:49:17,778][__main__][INFO] -
Progress: 0.6000000 Avg. train loss: 0.2444, train acc: 90.4%, Avg. val loss: 0.3243, val acc: 86.8%[2020-04-19 18:51:47,613][__main__][INFO] -
Progress: 0.7000000 Avg. train loss: 0.2269, train acc: 91.2%, Avg. val loss: 0.3276, val acc: 86.7%[2020-04-19 18:53:29,558][__main__][INFO] -
Progress: 0.8000000 Avg. train loss: 0.2134, train acc: 92.0%, Avg. val loss: 0.3181, val acc: 87.2%[2020-04-19 18:55:17,562][__main__][INFO] -
Progress: 0.9000000 Avg. train loss: 0.2035, train acc: 92.5%, Avg. val loss: 0.3154, val acc: 87.2%[2020-04-19 18:57:02,068][__main__][INFO] -
Progress: 1.0000000 Avg. train loss: 0.1957, train acc: 93.0%, Avg. val loss: 0.3146, val acc: 87.3%[2020-04-19 18:59:25,374][__main__][INFO] -

Test loss: 0.2883, test acc.: 88.2%[2020-04-19 19:00:04,749][__main__][INFO] -
Test loss: 0.2883, test acc.: 88.2%

# bi-gram
python -m supervised_fasttext.main parameters.ngram=2 parameters.word_n_gram_min_count=1
Use cpu
[2020-04-19 18:58:49,210][__main__][INFO] - Use cpu

#training_data: 22500, #val_data: 2500, #test_data: 25000
[2020-04-19 18:58:49,210][__main__][INFO] - #training_data: 22500, #val_data: 2500, #test_data: 25000

In training data, the size of word vocab: 27353 ngram vocab: 1152383, total: 1179736
[2020-04-19 18:58:49,210][__main__][INFO] - In training data, the size of word vocab: 27353 ngram vocab: 1152383, total: 1179736

Progress: 0.1000000 Avg. train loss: 0.6908, train acc: 53.3%, Avg. val loss: 0.6528, val acc: 57.8%[2020-04-19 19:00:28,954][__main__][INFO] -
Progress: 0.2000000 Avg. train loss: 0.5034, train acc: 76.4%, Avg. val loss: 0.4482, val acc: 79.2%[2020-04-19 19:01:47,814][__main__][INFO] -
Progress: 0.3000000 Avg. train loss: 0.3641, train acc: 85.3%, Avg. val loss: 0.3734, val acc: 83.9%[2020-04-19 19:04:13,907][__main__][INFO] -
Progress: 0.4000000 Avg. train loss: 0.2969, train acc: 88.3%, Avg. val loss: 0.3437, val acc: 86.1%[2020-04-19 19:07:42,645][__main__][INFO] -
Progress: 0.5000000 Avg. train loss: 0.2538, train acc: 90.6%, Avg. val loss: 0.3274, val acc: 86.8%[2020-04-19 19:09:16,552][__main__][INFO] -
Progress: 0.6000000 Avg. train loss: 0.2215, train acc: 92.1%, Avg. val loss: 0.3227, val acc: 86.8%[2020-04-19 19:13:07,828][__main__][INFO] -
Progress: 0.7000000 Avg. train loss: 0.1989, train acc: 93.2%, Avg. val loss: 0.3132, val acc: 87.4%[2020-04-19 19:14:19,116][__main__][INFO] -
Progress: 0.8000000 Avg. train loss: 0.1819, train acc: 94.2%, Avg. val loss: 0.3112, val acc: 87.3%[2020-04-19 19:15:25,216][__main__][INFO] -
Progress: 0.9000000 Avg. train loss: 0.1701, train acc: 94.7%, Avg. val loss: 0.3099, val acc: 87.6%[2020-04-19 19:16:41,786][__main__][INFO] -
Progress: 1.0000000 Avg. train loss: 0.1630, train acc: 95.0%, Avg. val loss: 0.3090, val acc: 87.4%[2020-04-19 19:17:37,623][__main__][INFO] -

Avg. test loss: 0.2760, test acc.: 88.9%[2020-04-19 19:17:50,701][__main__][INFO] -
Avg. test loss: 0.2760, test acc.: 88.9%
```
