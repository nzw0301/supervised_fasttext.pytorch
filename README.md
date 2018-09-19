# Usage

```
python -m supervised_fasttext.main -h
usage: main.py [-h] [--dim D] [--epochs N] [--lr LR] [--lr-update-rate ulr]
               [--no-cuda] [--gpu-id G] [--path PATH] [--train TRAIN]
               [--test TEST] [--seed S] [--val V] [--pre-trained PRE_TRAINED]
               [--logging-file LOGGING_FILE] [--patience PATIENCE]
               [--metric METRIC]

PyTorch supervised fastText example

optional arguments:
  -h, --help            show this help message and exit
  --dim D               number of hidden units (default: 10)
  --epochs N            number of epochs to train (default: 10)
  --lr LR               learning rate (default: 0.1)
  --lr-update-rate ulr  change learning rate schedule (default: 100)
  --no-cuda             disables CUDA training
  --gpu-id G            id of used GPU (default: 0)
  --path PATH           path to the data files (default: ./)
  --train TRAIN         file name of training data (default: train.tsv)
  --test TEST           file name of test data (default: test.tsv)
  --seed S              random seed (default: 7)
  --val V               ratio of validation data (default: 0.1)
  --pre-trained PRE_TRAINED
                        path to word vectors formatted by word2vec's text
                        (default: ``)
  --logging-file LOGGING_FILE
                        path to logging json file (default: `result.json`)
  --patience PATIENCE   the number of epochs for earlystopping (default: 5)
  --metric METRIC       metric name to be monitored by earlystopping. [loss,
                        acc] (default: loss)
```
