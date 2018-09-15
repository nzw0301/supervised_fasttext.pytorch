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
    test_loss = 0.
    correct = 0
    N = len(test_iter)
    test_iter.init_epoch()

    with torch.no_grad():
        for batch in test_iter:
            data, target = batch.text, batch.label
            data, target = data.to(device), target.to(device)  # TODO: `.to(device)` can be removed
            output = model(data)
            test_loss += F.nll_loss(output, target).item()
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    if divide_by_num_data:
        return test_loss / N, correct / N
    else:
        return test_loss, correct


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
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--val', type=float, default=0.1, metavar='V',
                        help='ratio of validation data (default: 0.1)')
    parser.add_argument('--pre-trained', type=str, default='',
                        help='path to word vectors formatted by word2vec\'s text (default: `''`)')
    parser.add_argument('--logging-file', type=str, default='result.json',
                        help='path to logging json file (default: `result.json`)')

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

    print('#train_data: {}, #val_data: {}, #test_data: {}'.format(len(train_iter), len(val_iter), len(test_iter)))

    epochs = args.epochs
    pre_trained_word_vectors = None
    dim = args.dim
    if args.pre_trained:
        pre_trained_word_vectors = np.zeros((len(TEXT.vocab), pre_trained_w2v.vector_size), dtype=np.float32)
        for id, word in enumerate(TEXT.vocab.itos):
            pre_trained_word_vectors[id] = pre_trained_w2v.get_vector(word)
        pre_trained_word_vectors = torch.from_numpy(pre_trained_word_vectors)
        dim = pre_trained_w2v.vector_size

    model = SupervisedFastText(V=len(TEXT.vocab), num_classes=len(LABEL.vocab), embedding_dim=dim,
                               pre_trained_emb=pre_trained_word_vectors,
                               freeze=True).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    # parameters for update learning rate
    num_tokens = 0
    train_iter.init_epoch()
    for i, batch in enumerate(train_iter):
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
        'test_loss': [],
        'test_acc': [],
    }

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
        print('Progress: {:.7f} Average train loss: {:.4f}, Average val loss: {:.4f}, Accuracy: {:.1f}%'.format(
            progress, train_loss, val_loss, val_acc*100
        ))

        # test
        test_loss, test_acc = test(model, device, test_iter)

        # save this epoch
        learning_history['train_loss'].append(train_loss)
        learning_history['val_loss'].append(val_loss)
        learning_history['val_acc'].append(val_acc)
        learning_history['test_loss'].append(test_loss)
        learning_history['test_acc'].append(test_acc)

    print('Average test loss: {:.4f}, Accuracy: {:.1f}%'.format(
        learning_history['test_loss'][-1],
        learning_history['test_acc'][-1]*100
    ))

    # logging_file
    with open(args.logging_file, 'w') as log_file:
        json.dump(learning_history, log_file)


if __name__ == '__main__':
    main()
