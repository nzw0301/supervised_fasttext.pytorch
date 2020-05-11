import json
import logging
import os
import random
from pathlib import Path

import hydra
import numpy as np
import optuna
import torch
import torch.nn.functional as F
from gensim.models import KeyedVectors
from hydra import utils
from optuna.samplers import TPESampler
from torch import optim

from supervised_fasttext.dataset import SentenceDataset
from supervised_fasttext.dictionary import SupervisedDictionary
from supervised_fasttext.experiments import evaluation
from supervised_fasttext.experiments import get_datasets, datasets2data_loaders
from supervised_fasttext.experiments import initialise_word_embeddigns_from_pretrained_embeddings
from supervised_fasttext.model import SupervisedFastText
from supervised_fasttext.utils import check_hydra_conf


class Objective(object):
    def __init__(self, hydra_cfg, logger):
        self.logger = logger
        self.hydra_cfg = hydra_cfg
        self.seed = hydra_cfg['parameters']['seed']
        self.metric = hydra_cfg['parameters']['metric']

        self.device = torch.device(
            'cuda:{}'.format(hydra_cfg['parameters']['gpu_id']) if torch.cuda.is_available() else 'cpu')

        working_dir = utils.get_original_cwd() + '/'
        training_path = working_dir + hydra_cfg['dataset']['path'] + hydra_cfg['dataset']['train_fname']
        is_replaced_OOV = hydra_cfg['parameters']['replace_OOV'] > 0

        # load embeddings
        pretrained_path = hydra_cfg['parameters']['pre_trained']
        pretrained_vocab = {}
        if pretrained_path:
            pretrained_path = working_dir + hydra_cfg['parameters']['pre_trained']
            self.logger.info('Loading pre-trained word embeddings {}\n'.format(pretrained_path))
            pretrained_w2v = KeyedVectors.load_word2vec_format(fname=pretrained_path)
            pretrained_vocab = set(pretrained_w2v.vocab.keys())
            assert hydra_cfg['parameters']['ngram'] == 1

        self.dictionary = SupervisedDictionary(
            replace_OOV_word=is_replaced_OOV,
            min_count=hydra_cfg['parameters']['min_count'],
            replace_word='<OOV>',
            size_word_n_gram=hydra_cfg['parameters']['ngram'],
            word_n_gram_min_count=hydra_cfg['parameters']['word_n_gram_min_count'],
            label_separator=hydra_cfg['parameters']['label_separator'],
            line_break_word=''
        )

        self.logger.info('Use {}\n'.format(self.device))

        self.dictionary.fit(training_path)

        if pretrained_vocab:
            self.dictionary.update_vocab_from_word_set(pretrained_vocab)

        self.train_set, self.val_set = get_datasets(
            cfg=hydra_cfg, dictionary=self.dictionary, working_dir=working_dir, training_path=training_path,
            include_test=False
        )

        pretrained_word_vectors = None
        dim = self.hydra_cfg['parameters']['dim']

        self.pooling = self.hydra_cfg['parameters']['pooling']

        OOV_initialized_method = self.hydra_cfg['parameters']['initialize_oov']
        self.is_freeze = self.hydra_cfg['parameters']['freeze'] > 0

        if pretrained_word_vectors:
            pretrained_word_vectors = initialise_word_embeddigns_from_pretrained_embeddings(
                pretrained_w2v, self.dictionary, OOV_initialized_method, rnd=np.random.RandomState(self.seed)
            )
            dim = pretrained_word_vectors.shape[1]
        self.pretrained_word_vectors = pretrained_word_vectors
        self.dim = dim

        self.logger.info('#training_data: {}, #val_data: {}\n'.format(
            len(self.train_set), len(self.val_set)
        ))
        self.logger.info('In training data, the size of word vocab: {} ngram vocab: {}, total: {} \n'.format(
            self.dictionary.size_word_vocab, self.dictionary.size_ngram_vocab, self.dictionary.size_total_vocab
        ))

    def __call__(self, trial: optuna.Trial):
        torch.manual_seed(self.seed)
        random.seed(self.seed)

        train_data_loader, val_data_loader = datasets2data_loaders(
            self.train_set, self.val_set, test_set=None, num_workers=1
        )

        epochs = self.hydra_cfg['parameters']['epochs']

        # Calculate an objective value by using the extra arguments.
        model = SupervisedFastText(
            V=self.dictionary.size_total_vocab,
            num_classes=len(self.dictionary.label_vocab),
            embedding_dim=self.dim,
            pretrained_emb=self.pretrained_word_vectors,
            freeze=self.is_freeze,
            pooling=self.pooling
        ).to(self.device)

        initial_lr = trial.suggest_loguniform(
            'lr',
            self.hydra_cfg['optuna']['lr_min'],
            self.hydra_cfg['optuna']['lr_max']
        )

        optimizer = optim.SGD(model.parameters(), lr=initial_lr)

        # parameters for update learning rate
        num_tokens = self.dictionary.num_words

        learning_rate_schedule = self.hydra_cfg['parameters']['lr_update_rate']
        total_num_processed_tokens_in_training = epochs * num_tokens
        num_processed_tokens = 0
        local_processed_tokens = 0
        N = len(train_data_loader.dataset)

        best_val_loss = np.finfo(0.).max
        best_val_acc = np.finfo(0.).min
        save_fname = os.getcwd() + '/' + '{}.pt'.format(trial.number)  # file name to store best model's weights

        for epoch in range(epochs):
            # begin training phase
            sum_loss = 0.
            correct = 0
            model.train()

            for sentence, label, n_tokens in train_data_loader:
                sentence, label = sentence.to(self.device), label.to(self.device)
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
                    optimizer.param_groups[0]['lr'] = initial_lr * (1. - progress)

            train_loss = sum_loss / N
            train_acc = correct / N
            # end training phase

            val_loss, val_acc = evaluation(model, self.device, val_data_loader)

            progress = num_processed_tokens / total_num_processed_tokens_in_training  # approximated progress
            self.logger.info(
                '\rProgress: {:.1f}% Avg. train loss: {:.4f}, train acc: {:.1f}%, '
                'Avg. val loss: {:.4f}, val acc: {:.1f}%'.format(
                    progress * 100., train_loss, train_acc * 100, val_loss, val_acc * 100
                )
            )

            if self.metric == 'loss':
                trial.report(val_loss, epoch)
            else:
                trial.report(val_acc, epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            # validation
            is_saved_model = False
            if self.metric == 'loss':
                if best_val_loss > val_loss:
                    best_val_loss = val_loss
                    best_val_acc = val_acc
                    is_saved_model = True
            else:
                if best_val_acc < val_acc:
                    best_val_loss = val_loss
                    best_val_acc = val_acc
                    is_saved_model = True

            if is_saved_model:
                torch.save(model.state_dict(), save_fname)

        trial.set_user_attr('val_loss', best_val_loss)
        trial.set_user_attr('val_acc', best_val_acc)
        trial.set_user_attr('model_path', save_fname)

        if self.metric == 'loss':
            return best_val_loss
        else:
            return best_val_acc


@hydra.main(config_path='../conf/config.yaml')
def main(cfg):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.terminator = ''
    logger.addHandler(stream_handler)

    check_hydra_conf(cfg)

    metric = cfg['parameters']['metric']
    if metric == 'loss':
        direction = 'minimize'
    else:
        direction = 'maximize'

    sampler = TPESampler(seed=cfg['parameters']['seed'], n_startup_trials=2)
    pruner = optuna.pruners.HyperbandPruner(max_resource=cfg['parameters']['epochs'])
    study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)
    objective = Objective(cfg, logger)
    study.optimize(objective, n_trials=cfg['optuna']['num_trials'], n_jobs=cfg['optuna']['n_jobs'])

    # logging_file
    trial = study.best_trial

    logger.info('\nVal. loss: {:.4f}, Val acc.: {:.1f}%'.format(
        trial.user_attrs['val_loss'],
        trial.user_attrs['val_acc'] * 100
    ))

    for key, value in trial.params.items():
        logger.info('    {}: {}'.format(key, value))

    # remove poor models
    target = Path(trial.user_attrs['model_path'])
    for path in target.parent.glob('*.pt'):
        if path != target:
            path.unlink()

    # evaluation
    # load test data loader
    working_dir = utils.get_original_cwd() + '/'
    test_set = SentenceDataset(
        *objective.dictionary.transform(working_dir + cfg['dataset']['path'] + cfg['dataset']['test_fname']),
        objective.dictionary.size_word_vocab,
        train=False
    )
    test_data_loader = torch.utils.data.dataloader.DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=1
    )

    device = objective.device
    # init model
    model = SupervisedFastText(
        V=objective.dictionary.size_total_vocab,
        num_classes=len(objective.dictionary.label_vocab),
        embedding_dim=objective.dim,
        pretrained_emb=None,
        freeze=True,
        pooling=objective.pooling
    ).to(device)

    # load model
    model.load_state_dict(torch.load(target, map_location=device))
    model = model.to(device)

    loss, acc = evaluation(model, device, test_data_loader, divide_by_num_data=True)
    results = trial.user_attrs
    results['test_loss'] = loss
    results['test_acc'] = acc

    output_path_fname = os.getcwd() + '/' + cfg['parameters']['logging_file']
    logger.info('Saving training history and evaluation scores in {}'.format(output_path_fname))
    with open(output_path_fname, 'w') as log_file:
        json.dump(results, log_file)


if __name__ == '__main__':
    main()
