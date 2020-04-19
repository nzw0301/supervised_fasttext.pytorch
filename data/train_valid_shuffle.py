import argparse

import numpy as np
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser(description='Splits dataset into training and validation sets')
parser.add_argument('-sep', type=str, default='\t', help='separator between label and sentence')
parser.add_argument('--input-file', type=str, help='input training file.')
parser.add_argument('--seed', type=int, default=0, help='random seed of train shuffle')
parser.add_argument('--val', type=float, default=0.1, help='random seed of train shuffle')

args = parser.parse_args()
seed = args.seed
rnd = np.random.RandomState(seed)
separator = args.sep

x, y = [], []

input_fname = args.input_file
with open(input_fname) as f:
    for l in f:
        label, sentence = l.strip().split(sep=separator)
        y.append(label)
        x.append(sentence)

x = np.array(x)
y = np.array(y)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=args.val, random_state=rnd, shuffle=True, stratify=y)

splits = input_fname.split('.')
fname, file_extension = '.'.join(splits[:-1]) , splits[-1]

train_fname = '{}.{}.{}'.format(fname, seed, file_extension)
with open(train_fname, 'w') as f:
    for sentence, label, in zip(x_train, y_train):
        line = label + separator + sentence + '\n'
        f.write(line)

val_fname = '{}.{}.val.{}'.format(fname, seed, file_extension)
with open(val_fname, 'w') as f:
    for sentence, label, in zip(x_val, y_val):
        line = label + separator + sentence + '\n'
        f.write(line)
