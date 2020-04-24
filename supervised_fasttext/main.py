import json
import logging
import os
import random

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from hydra import utils
from gensim.models import KeyedVectors
from torch import optim
from torch.utils.data.dataloader import DataLoader

from supervised_fasttext.dataset import SentenceDataset
from supervised_fasttext.dictionary import SupervisedDictionary
from supervised_fasttext.model import SupervisedFastText
from supervised_fasttext.utils import EarlyStopping

_valid_initialised_methods = ['uniform', 'mean']


def initialise_word_embeddigns_from_pretrained_embeddings(
        embeddings: KeyedVectors, dictionary: SupervisedDictionary,
        OOV_initialised_method: str, rnd: np.random.RandomState
):
    assert OOV_initialised_method in _valid_initialised_methods

    shape = (dictionary.size_word_vocab, embeddings.vector_size)

    if OOV_initialised_method == 'uniform':
        upper = 1. / embeddings.vector_size
        pretrained_word_vectors = rnd.rand(-upper, upper, size=shape).astype(dtype=np.float32)
    else:
        global_mean_vector = np.mean(embeddings.vectors, axis=0)
        pretrained_word_vectors = np.tile(global_mean_vector, (dictionary.size_word_vocab, 1))

    for word_id, word in enumerate(dictionary.word_vocab.id2word):
        if word == dictionary.replace_word:
            continue
        pretrained_word_vectors[word_id] = embeddings.get_vector(word)

    return torch.from_numpy(pretrained_word_vectors)


def evaluation(model, device, test_iter, divide_by_num_data=True):
    """
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

    with torch.no_grad():
        for sentence, label, _ in test_iter:
            sentence, label = sentence.to(device), label.to(device)
            output = model(sentence)
            loss += F.nll_loss(output, label).item()
            pred = output.argmax(1, keepdim=False)
            correct += pred.eq(label).sum().item()

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
    assert 0 < cfg['parameters']['patience']
    assert cfg['parameters']['metric'] in ['loss', 'acc']
    assert cfg['parameters']['initialize_oov'] in _valid_initialised_methods


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
    rnd = np.random.RandomState(cfg['parameters']['seed'])
    OOV_initialized_method = cfg['parameters']['initialize_oov']
    is_freeze = cfg['parameters']['freeze'] > 0
    is_replaced_OOV = cfg['parameters']['replace_OOV'] > 0

    device = torch.device('cuda:{}'.format(cfg['parameters']['gpu_id']) if use_cuda else 'cpu')

    working_dir = utils.get_original_cwd() + '/'

    # load embeddings
    pretrained_path = cfg['parameters']['pre_trained']
    pretrained_vocab = {}
    if pretrained_path:
        pretrained_path = working_dir + cfg['parameters']['pre_trained']
        logger.info('Loading pre-trained word embeddings {}\n'.format(pretrained_path))
        pretrained_w2v = KeyedVectors.load_word2vec_format(fname=pretrained_path)
        pretrained_vocab = set(pretrained_w2v.vocab.keys())
        assert cfg['parameters']['ngram'] == 1

    dictionary = SupervisedDictionary(
        replace_OOV_word=is_replaced_OOV,
        min_count=cfg['parameters']['min_count'],
        replace_word="<OOV>",
        size_word_n_gram=cfg['parameters']['ngram'],
        word_n_gram_min_count=cfg['parameters']['word_n_gram_min_count'],
        label_separator=cfg['parameters']['label_separator'],
        line_break_word=""
    )

    training_path = working_dir + cfg['dataset']['path'] + cfg['dataset']['train_fname']
    dictionary.fit(training_path)

    if pretrained_vocab:
        dictionary.update_vocab_from_word_set(pretrained_vocab)

    train_set = SentenceDataset(
        *dictionary.transform(training_path),
        size_vocab=dictionary.size_word_vocab,
        train=True
    )
    val_set = SentenceDataset(
        *dictionary.transform(working_dir + cfg['dataset']['path'] + cfg['dataset']['val_fname']),
        dictionary.size_word_vocab,
        train=False
    )
    test_set = SentenceDataset(
        *dictionary.transform(working_dir + cfg['dataset']['path'] + cfg['dataset']['test_fname']),
        dictionary.size_word_vocab,
        train=False
    )

    num_workers = 4
    train_data_loader = DataLoader(
        train_set,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers
    )

    val_data_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers
    )

    test_data_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers
    )

    metric = cfg['parameters']['metric']
    if metric == 'loss':
        mode = 'min'
    else:
        mode = 'max'

    early_stopping = EarlyStopping(mode=mode, patience=cfg['parameters']['patience'])

    epochs = cfg['parameters']['epochs']
    pretrained_word_vectors = None
    dim = cfg['parameters']['dim']
    pooling = cfg['parameters']['pooling']

    if pretrained_path:
        pretrained_word_vectors = initialise_word_embeddigns_from_pretrained_embeddings(
            pretrained_w2v, dictionary, OOV_initialized_method, rnd
        )
        dim = pretrained_word_vectors.shape[1]

    logger.info('Use {}\n'.format(device))
    logger.info('#training_data: {}, #val_data: {}, #test_data: {}\n'.format(
        len(train_data_loader.dataset), len(val_data_loader.dataset), len(test_data_loader.dataset)),
    )
    logger.info('In training data, the size of word vocab: {} ngram vocab: {}, total: {} \n'.format(
        dictionary.size_word_vocab, dictionary.size_ngram_vocab, dictionary.size_total_vocab
    )
    )

    model = SupervisedFastText(
        V=dictionary.size_total_vocab,
        num_classes=len(dictionary.label_vocab),
        embedding_dim=dim,
        pretrained_emb=pretrained_word_vectors,
        freeze=is_freeze,
        pooling=pooling
    ).to(device)

    optimizer = optim.SGD(model.parameters(), lr=cfg['parameters']['lr'])

    # parameters for update learning rate
    num_tokens = dictionary.num_words

    learning_rate_schedule = cfg['parameters']['lr_update_rate']
    total_num_processed_tokens_in_training = epochs * num_tokens
    num_processed_tokens = 0
    local_processed_tokens = 0
    N = len(train_data_loader.dataset)

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
        sum_loss = 0.
        correct = 0
        model.train()

        for sentence, label, n_tokens in train_data_loader:
            sentence, label = sentence.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(sentence)
            loss = F.nll_loss(output, label)
            loss.backward()
            optimizer.step()
            pred = output.argmax(1, keepdim=False)
            correct += pred.eq(label).sum().item()
            sum_loss += loss.item()

            # update learning rate
            # ref: https://github.com/facebookresearch/fastText/blob/6d7c77cd33b23eec26198fdfe10419476b5364c7/src/fasttext.cc#L656
            local_processed_tokens += n_tokens.item()
            if local_processed_tokens > learning_rate_schedule:
                num_processed_tokens += local_processed_tokens
                local_processed_tokens = 0
                progress = num_processed_tokens / total_num_processed_tokens_in_training
                optimizer.param_groups[0]['lr'] = cfg['parameters']['lr'] * (1. - progress)

        train_loss = sum_loss / N
        train_acc = correct / N
        # end training phase

        # validation
        val_loss, val_acc = evaluation(model, device, val_data_loader)

        progress = num_processed_tokens / total_num_processed_tokens_in_training  # approximated progress
        logger.info(
            '\rProgress: {:.1f}% Avg. train loss: {:.4f}, train acc: {:.1f}%, '
            'Avg. val loss: {:.4f}, val acc: {:.1f}%'.format(
                progress * 100., train_loss, train_acc * 100, val_loss, val_acc * 100
            )
        )

        # test
        test_loss, test_acc = evaluation(model, device, test_data_loader)
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

    logger.info('\nTest loss: {:.4f}, test acc.: {:.1f}%'.format(
        test_loss,
        test_acc * 100
    ))

    # logging_file
    output_path_fname = os.getcwd() + '/' + cfg['parameters']['logging_file']
    logger.info('Saving training history and evaluation scores in {}'.format(output_path_fname))
    with open(output_path_fname, 'w') as log_file:
        json.dump(learning_history, log_file)


if __name__ == '__main__':
    main()
