import random
import json
import logging
from collections import Counter, OrderedDict

from gensim.models import KeyedVectors
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torchtext.data import TabularDataset, Iterator, Field, LabelField, Dataset
from torchtext.vocab import Vocab

from .model import SupervisedFastText
from .utils import EarlyStopping


def clean(example, vocab):
    """
    TODO: write something here
    :param example:
    :param vocab:
    :return:
    """
    return [word for word in example if word in vocab]


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
    """
    TODO:
    :param model:
    :param device:
    :param test_iter:
    :param divide_by_num_data:
    :return:
    """
    model.eval()
    loss = 0.
    correct = 0
    N = len(test_iter)
    test_iter.init_epoch()

    with torch.no_grad():
        for batch in test_iter:
            data, target = batch.text, batch.label
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += F.nll_loss(output, target).item()
            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    if divide_by_num_data:
        return loss / N, correct / N
    else:
        return loss, correct


def check_conf(cfg):
    """
    Validate the hydra'S config parameters
    :param cfg:
    :return: None
    """
    assert 0 < cfg['parameters']['dim']
    assert 0 < cfg['parameters']['min_count']
    assert 0 < cfg['parameters']['epochs']
    assert 0. < cfg['parameters']['lr']
    assert 0. < cfg['parameters']['lr_update_rate']
    assert 0. < cfg['parameters']['val_ratio'] < 1.
    assert 0 < cfg['parameters']['patience']
    assert cfg['parameters']['metric'] in ['loss', 'acc']


@hydra.main(config_path='../conf/config.yaml')
def main(cfg):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.terminator = ''
    logger.addHandler(stream_handler)

    check_conf(cfg)

    use_cuda = torch.cuda.is_available()

    torch.manual_seed(cfg['parameters']['seed'])
    random.seed(cfg['parameters']['seed'])


    device = torch.device('cuda:{}'.format(cfg['parameters']['gpu_id']) if use_cuda else 'cpu')

    TEXT = Field(pad_token=None, unk_token=None, batch_first=True)  # do not use padding and <unk>
    LABEL = LabelField()

    train_data, test_data = TabularDataset.splits(
        path=cfg['dataset']['path'],
        train=cfg['dataset']['input_train_fname'],
        test=cfg['dataset']['input_test_fname'],
        format='tsv',
        fields=[('label', LABEL), ('text', TEXT)]
    )

    train_data, val_data = train_data.split(
        split_ratio=(1. - cfg['parameters']['val_ratio']),
        random_state=random.getstate()
    )

    # load embeddings
    # TODO: should be removed non-appearing words in `train_data`?
    pretrained_path = cfg['parameters']['pre_trained']
    if pretrained_path:
        logger.info('Loading pre-trained word embeddings {}\n'.format(pretrained_path))
        pre_trained_w2v = KeyedVectors.load_word2vec_format(fname=pretrained_path)
        w2v_v = set(pre_trained_w2v.vocab.keys())
        TEXT.vocab = build_vocab_with_word2vec(TEXT, w2v_v, train_data)
    else:
        TEXT.build_vocab(train_data)

    LABEL.build_vocab(train_data)

    # Delete unknown words
    # https://github.com/pytorch/text/issues/355#issuecomment-422047412
    for data in [train_data, val_data, test_data]:
        for i in range(len(data)):
            data[i].text = clean(data[i].text, TEXT.vocab.stoi)

    train_iter = Iterator(
        dataset=train_data, batch_size=1, device=device, train=True, shuffle=True, repeat=False, sort=False
    )
    val_iter = Iterator(
        dataset=val_data, batch_size=1, device=device, train=False, shuffle=False, repeat=False, sort=False
    )
    test_iter = Iterator(
        dataset=test_data, batch_size=1, device=device, train=False, shuffle=False, repeat=False, sort=False
    )

    metric = cfg['parameters']['metric']
    if metric == 'loss':
        mode = 'min'
    else:
        mode = 'max'

    early_stopping = EarlyStopping(mode=mode, patience=cfg['parameters']['patience'])

    epochs = cfg['parameters']['epochs']
    pre_trained_word_vectors = None
    dim = cfg['parameters']['dim']
    if pretrained_path:
        pre_trained_word_vectors = np.zeros((len(TEXT.vocab), pre_trained_w2v.vector_size), dtype=np.float32)
        for i, word in enumerate(TEXT.vocab.itos):
            pre_trained_word_vectors[i] = pre_trained_w2v.get_vector(word)
        pre_trained_word_vectors = torch.from_numpy(pre_trained_word_vectors)
        dim = pre_trained_w2v.vector_size

    logger.info('Use {}\n'.format(device))
    logger.info('#training_data: {}, #val_data: {}, #test_data: {}\n'.format(
        len(train_iter), len(val_iter), len(test_iter))
    )
    logger.info('the size of vocab in training data: {}\n'.format(len(TEXT.vocab)))

    model = SupervisedFastText(
        V=len(TEXT.vocab),
        num_classes=len(LABEL.vocab),
        embedding_dim=dim,
        pre_trained_emb=pre_trained_word_vectors,
        freeze=True
    ).to(device)

    optimizer = optim.SGD(model.parameters(), lr=cfg['parameters']['lr'])

    # parameters for update learning rate
    num_tokens = 0
    train_iter.init_epoch()
    for batch in train_iter:
        num_tokens += batch.text.shape[1]

    learning_rate_schedule = cfg['parameters']['lr_update_rate']
    total_num_processed_tokens_in_training = epochs * num_tokens
    num_processed_tokens = 0
    local_processed_tokens = 0
    N = len(train_iter)

    learning_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'test_loss': 0.,
        'test_acc': 0.
    }

    test_loss_list = []
    test_acc_list = []

    is_stopped = False  # flag of early stopping
    for epoch in range(1, epochs + 1):
        # begin training phase
        train_iter.init_epoch()
        sum_loss = 0.
        correct = 0
        model.train()

        for batch_idx, batch in enumerate(train_iter):
            data, target = batch.text, batch.label
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            sum_loss += loss.item()

            # update learning rate
            local_processed_tokens += data.shape[1]
            if local_processed_tokens > learning_rate_schedule:
                num_processed_tokens += local_processed_tokens
                local_processed_tokens = 0
                progress = num_processed_tokens / total_num_processed_tokens_in_training
                optimizer.param_groups[0]['lr'] = cfg['parameters']['lr'] * (1. - progress)
        train_loss = sum_loss / N
        train_acc = correct / N
        # end training phase

        # validation
        val_loss, val_acc = test(model, device, val_iter)

        progress = num_processed_tokens / total_num_processed_tokens_in_training  # approximated progress
        logger.info(
            '\rProgress: {:.7f} Avg. train loss: {:.4f}, train acc: {:.1f}%, '
            'Avg. val loss: {:.4f}, val acc: {:.1f}%'.format(
                progress, train_loss, train_acc * 100, val_loss, val_acc * 100
            )
        )

        # test
        test_loss, test_acc = test(model, device, test_iter)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)

        # save this epoch
        learning_history['train_loss'].append(train_loss)
        learning_history['train_acc'].append(train_acc)
        learning_history['val_loss'].append(val_loss)
        learning_history['val_acc'].append(val_acc)

        # check early stopping
        if metric == 'loss':
            is_stopped = early_stopping.is_stopped(val_loss)
        else:
            is_stopped = early_stopping.is_stopped(val_acc)

        if is_stopped:
            logger.info('Early stop!')
            break

    if is_stopped:
        best_epoch_index = epoch - cfg['parameters']['patience'] - 1
    else:
        best_epoch_index = -1

    # store test evaluation result
    test_loss = test_loss_list[best_epoch_index]
    test_acc = test_acc_list[best_epoch_index]
    learning_history['test_loss'] = test_loss
    learning_history['test_acc'] = test_acc

    # logging_file
    with open(cfg['parameters']['logging_file'], 'w') as log_file:
        json.dump(learning_history, log_file)

    logger.info('\nAvg. test loss: {:.4f}, test acc.: {:.1f}%'.format(
        test_loss,
        test_acc * 100
    ))


if __name__ == '__main__':
    main()
