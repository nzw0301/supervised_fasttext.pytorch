import argparse
import random
import numpy as np
import json
from collections import Counter, OrderedDict
from gensim.models import KeyedVectors

import torch
import torch.nn.functional as F
from torch import optim
from torchtext.data import TabularDataset, Iterator, Field, LabelField, Dataset
from torchtext.vocab import Vocab

from .model import SupervisedFastText
from .utils import EarlyStopping


def build_vocab_with_word2vec(field_instance, w2v_word_set, *args, **kwargs):
    """
    Delete words that only appear in training data.
    This function is used for fixed embeddings.

    :param field_instance: `Filed` instance which has `vocab` such that it processes train data.
    :param w2v_word_set: words set of pre-trained embeddings
    :param args: same to `Filed.build_vocab`
    :param kwargs: same to `Filed.build_vocab`
    :return: `Vocab`
    """
    counter = Counter()
    sources = []
    for arg in args:
        if isinstance(arg, Dataset):
            sources += [getattr(arg, name) for name, field in
                        arg.fields.items() if field is field_instance]
        else:
            sources.append(arg)
    for data in sources:
        for x in data:
            counter.update([w for w in x if w in w2v_word_set])

    specials = list(OrderedDict.fromkeys(
        tok for tok in [field_instance.unk_token, field_instance.pad_token, field_instance.init_token,
                        field_instance.eos_token]
        if tok is not None))
    return Vocab(counter, specials=specials, **kwargs)


def test(model, device, test_iter, divide_by_num_data=True):
    model.eval()
    loss = 0.
    correct = 0
    N = len(test_iter)
    test_iter.init_epoch()

    with torch.no_grad():
        for batch in test_iter:
            data, target = batch.text, batch.label
            data, target = data.to(device), target.to(device)  # TODO: `.to(device)` can be removed
            output = model(data)
            loss += F.nll_loss(output, target).item()
            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    if divide_by_num_data:
        return loss / N, correct / N
    else:
        return loss, correct


def main():
    parser = argparse.ArgumentParser(description='PyTorch supervised fastText example')
    parser.add_argument('--dim', type=int, default=10, metavar='D',
                        help='number of hidden units (default: 10)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--lr-update-rate', type=int, default=100, metavar='ulr',
                        help='change learning rate schedule (default: 100)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--gpu-id', type=int, default=0, metavar='G',
                        help='id of used GPU (default: 0)')
    parser.add_argument('--path', type=str, default='./',
                        help='path to the data files (default: ./)')
    parser.add_argument('--train', type=str, default='train.tsv',
                        help='file name of training data (default: train.tsv)')
    parser.add_argument('--test', type=str, default='test.tsv',
                        help='file name of test data (default: test.tsv)')
    parser.add_argument('--seed', type=int, default=7, metavar='S',
                        help='random seed (default: 7)')
    parser.add_argument('--val', type=float, default=0.1, metavar='V',
                        help='ratio of validation data (default: 0.1)')
    parser.add_argument('--pre-trained', type=str, default='',
                        help='path to word vectors formatted by word2vec\'s text (default: `''`)')
    parser.add_argument('--logging-file', type=str, default='result.json',
                        help='path to logging json file (default: `result.json`)')
    parser.add_argument('--patience', type=int, default=5,
                        help='the number of epochs for earlystopping (default: 5')
    parser.add_argument('--metric', type=str, default='loss',
                        help='metric name to be monitored by earlystopping. [loss, acc] (default: loss')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device('cuda:{}'.format(args.gpu_id) if use_cuda else 'cpu')

    path = args.path
    TEXT = Field(pad_token=None, unk_token=None, batch_first=True)  # do not use padding and <unk>
    LABEL = LabelField()

    train_data, test_data = TabularDataset.splits(
        path=path, train=args.train,
        test=args.test, format='tsv',
        fields=[('label', LABEL), ('text', TEXT)])

    train_data, val_data = train_data.split(split_ratio=(1.-args.val), random_state=random.getstate())

    # load embeddings
    if args.pre_trained:
        print('Loading pre-trained word embeddings {}'.format(args.pre_trained))
        pre_trained_w2v = KeyedVectors.load_word2vec_format(fname=args.pre_trained)
        w2v_v = set(pre_trained_w2v.vocab.keys())
        TEXT.vocab = build_vocab_with_word2vec(TEXT, w2v_v, train_data)
    else:
        TEXT.build_vocab(train_data)

    LABEL.build_vocab(train_data)

    if device.type == 'cpu':
        iterator_device = -1
    else:
        iterator_device = device

    train_iter = Iterator(dataset=train_data, batch_size=1, device=iterator_device, train=True, shuffle=True,
                          repeat=False, sort=False)
    val_iter = Iterator(dataset=val_data, batch_size=1, device=iterator_device, train=False, shuffle=False,
                        repeat=False, sort=False)
    test_iter = Iterator(dataset=test_data, batch_size=1, device=iterator_device, train=False, shuffle=False,
                         repeat=False, sort=False)

    if args.metric == 'loss':
        mode = 'min'
    elif args.metric == 'acc':
        mode = 'max'
    else:
        ValueError('Invalid monitored metric error for `EarlyStopping`.')
    early_stopping = EarlyStopping(mode=mode, patience=args.patience)
    
    epochs = args.epochs
    pre_trained_word_vectors = None
    dim = args.dim
    if args.pre_trained:
        pre_trained_word_vectors = np.zeros((len(TEXT.vocab), pre_trained_w2v.vector_size), dtype=np.float32)
        for id, word in enumerate(TEXT.vocab.itos):
            pre_trained_word_vectors[id] = pre_trained_w2v.get_vector(word)
        pre_trained_word_vectors = torch.from_numpy(pre_trained_word_vectors)
        dim = pre_trained_w2v.vector_size

    print('Use {}'.format(device))
    print('#training_data: {}, #val_data: {}, #test_data: {}'.format(len(train_iter), len(val_iter), len(test_iter)),
          end=', ')
    print('the size of vocab in training data: {}'.format(len(TEXT.vocab)))

    model = SupervisedFastText(V=len(TEXT.vocab), num_classes=len(LABEL.vocab), embedding_dim=dim,
                               pre_trained_emb=pre_trained_word_vectors,
                               freeze=True).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    # parameters for update learning rate
    num_tokens = 0
    train_iter.init_epoch()
    for batch in train_iter:
        num_tokens += batch.text.shape[1]

    learning_rate_schedule = args.lr_update_rate
    total_num_processed_tokens_in_training = epochs * num_tokens
    num_processed_tokens = 0
    local_processed_tokens = 0
    N = len(train_iter)
    learning_history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'test_loss': 0.,
        'test_acc': 0.
    }

    test_loss_list = []
    test_acc_list = []

    is_stopped = False  # earlystopping flag
    for epoch in range(1, epochs + 1):
        # begin training phase
        train_iter.init_epoch()
        sum_loss = 0.
        model.train()

        for batch_idx, batch in enumerate(train_iter):
            data, target = batch.text, batch.label
            data, target = data.to(device), target.to(device)  # TODO: `.to(device)` can be removed
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()

            # update learning rate
            local_processed_tokens += data.shape[1]
            if local_processed_tokens > learning_rate_schedule:
                num_processed_tokens += local_processed_tokens
                local_processed_tokens = 0
                progress = num_processed_tokens / total_num_processed_tokens_in_training
                optimizer.param_groups[0]['lr'] = args.lr * (1. - progress)
        train_loss = sum_loss / N
        # end training phase

        # validation
        val_loss, val_acc = test(model, device, val_iter)

        progress = num_processed_tokens / total_num_processed_tokens_in_training  # approximated progress
        print('Progress: {:.7f} Avg. train loss: {:.4f}, Avg. val loss: {:.4f}, avl acc: {:.1f}%'.format(
            progress, train_loss, val_loss, val_acc*100
        ))

        # test
        test_loss, test_acc = test(model, device, test_iter)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)

        # save this epoch
        learning_history['train_loss'].append(train_loss)
        learning_history['val_loss'].append(val_loss)
        learning_history['val_acc'].append(val_acc)

        # check earlystopping
        if args.metric == 'loss':
            is_stopped = early_stopping.is_stopped(val_loss)
        else:
            is_stopped = early_stopping.is_stopped(val_acc)

        if is_stopped:
            print('Earlystop!')
            break

    if is_stopped:
        best_epoch_index = epoch - args.patience - 1
    else:
        best_epoch_index = -1

    # store test evaluation result
    test_loss = test_loss_list[best_epoch_index]
    test_acc = test_acc_list[best_epoch_index]
    learning_history['test_loss'] = test_loss
    learning_history['test_acc'] = test_acc

    # logging_file
    with open(args.logging_file, 'w') as log_file:
        json.dump(learning_history, log_file)

    print('Ave. test loss: {:.4f}, test acc.: {:.1f}%'.format(
        test_loss,
        test_acc*100
    ))


if __name__ == '__main__':
    main()
