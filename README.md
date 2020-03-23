## Requirements

- `pytorch`: 1.4.0
- `hydra`: 0.11.3
- `torchtext`: 0.5.0
- `gensim`: 3.8.0
- `numpy`: 1.18.1

## Usage

```bash
$ python -m supervised_fasttext.main --cfg job
dataset:
  input_test_fname: test.tsv
  input_train_fname: train.tsv
  path: ../../../data/
parameters:
  dim: 10
  epochs: 10
  gpu_id: 0
  logging_file: result.json
  lr: 0.1
  lr_update_rate: 100
  metric: loss
  min_count: 5
  patience: 5
  pre_trained: null
  seed: 7
  val_ratio: 0.1
```

Please check [`conf/config.yaml`](./conf/config.yaml) for the details.

## Demo: Binary classification on IMDb dataset

```bash
cd data/imdb
sh imdb.sh
perl ../wikifil.pl < train-data.txt > train-data.txt.normalized
perl ../wikifil.pl < test-data.txt  > test-data.txt.normalized
paste train-label train-data.txt.normalized > train.tsv
paste test-label test-data.txt.normalized > test.tsv
rm test-data.txt.normalized test-label test-data.txt train-data.txt train-data.txt.normalized train-label
cd ../../

python -m supervised_fasttext.main

Use cpu
[2020-03-21 00:18:31,001][__main__][INFO] - Use cpu

#training_data: 22500, #val_data: 2500, #test_data: 25000
[2020-03-21 00:18:31,003][__main__][INFO] - #training_data: 22500, #val_data: 2500, #test_data: 25000

the size of vocab in training data: 70158
[2020-03-21 00:18:31,019][__main__][INFO] - the size of vocab in training data: 70158

Progress: 0.1000000 Avg. train loss: 0.6123, train acc: 63.5%, Avg. val loss: 0.4622, val acc: 80.5%[2020-03-21 00:19:45,289][__main__][INFO] -
Progress: 0.2000000 Avg. train loss: 0.3970, train acc: 82.8%, Avg. val loss: 0.3808, val acc: 83.8%[2020-03-21 00:21:19,497][__main__][INFO] -
Progress: 0.3000000 Avg. train loss: 0.3262, train acc: 86.2%, Avg. val loss: 0.3708, val acc: 84.5%[2020-03-21 00:22:45,684][__main__][INFO] -
Progress: 0.4000000 Avg. train loss: 0.2894, train acc: 88.2%, Avg. val loss: 0.3243, val acc: 87.8%[2020-03-21 00:24:08,320][__main__][INFO] -
Progress: 0.5000000 Avg. train loss: 0.2621, train acc: 89.4%, Avg. val loss: 0.3143, val acc: 87.2%[2020-03-21 00:25:31,675][__main__][INFO] -
Progress: 0.6000000 Avg. train loss: 0.2433, train acc: 90.4%, Avg. val loss: 0.3111, val acc: 88.2%[2020-03-21 00:26:44,189][__main__][INFO] -
Progress: 0.7000000 Avg. train loss: 0.2270, train acc: 91.3%, Avg. val loss: 0.3177, val acc: 87.2%[2020-03-21 00:28:03,670][__main__][INFO] -
Progress: 0.7999987 Avg. train loss: 0.2122, train acc: 92.1%, Avg. val loss: 0.3129, val acc: 87.3%[2020-03-21 00:29:33,457][__main__][INFO] -
Progress: 0.9000000 Avg. train loss: 0.2019, train acc: 92.6%, Avg. val loss: 0.3037, val acc: 88.7%[2020-03-21 00:30:57,533][__main__][INFO] -
Progress: 1.0000000 Avg. train loss: 0.1936, train acc: 93.0%, Avg. val loss: 0.2990, val acc: 88.6%[2020-03-21 00:32:29,672][__main__][INFO] -

Avg. test loss: 0.2882, test acc.: 88.3%[2020-03-21 00:33:04,539][__main__][INFO] -
Avg. test loss: 0.2882, test acc.: 88.3%
```
