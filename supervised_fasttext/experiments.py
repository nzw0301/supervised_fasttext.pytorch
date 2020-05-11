import numpy as np
import torch
import torch.nn.functional as F
from gensim.models import KeyedVectors
from torch.utils.data.dataloader import DataLoader

from supervised_fasttext.dataset import SentenceDataset
from supervised_fasttext.dictionary import SupervisedDictionary
from supervised_fasttext.utils import _valid_initialised_methods


def get_datasets(cfg, dictionary: SupervisedDictionary, working_dir: str, training_path: str, include_test: False):
    """
    :param cfg: Hydra's config instance
    :param dictionary: Tokenized SupervisedDictionary instance.
    :param working_dir: Working dir of this code.
    :param training_path: Path to training dataset.
    :param include_test: Boolean. Whether include test dataset or not.
    :return: Tuple of SentenceDataset instances.
    """
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
    if not include_test:
        return train_set, val_set

    else:
        test_set = SentenceDataset(
            *dictionary.transform(working_dir + cfg['dataset']['path'] + cfg['dataset']['test_fname']),
            dictionary.size_word_vocab,
            train=False
        )
        return train_set, val_set, test_set


def datasets2data_loaders(train_set=None, val_set=None, test_set=None, num_workers=1):
    """
    :param train_set: SentenceDataset's instance.
    :param val_set: SentenceDataset's instance.
    :param test_set: SentenceDataset's instance.
    :param num_workers: The number of workers of data loader. Larger values causes memory error.
    :return: List of DataLoaders.
    """
    data_loaders = []
    if train_set:
        train_data_loader = DataLoader(
            train_set,
            batch_size=1,
            shuffle=True,
            num_workers=num_workers
        )
        data_loaders.append(train_data_loader)
    if val_set:
        val_data_loader = DataLoader(
            val_set,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers
        )
        data_loaders.append(val_data_loader)
    if test_set:
        test_data_loader = DataLoader(
            test_set,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers
        )
        data_loaders.append(test_data_loader)

    return data_loaders


def get_data_loaders(cfg, dictionary: SupervisedDictionary, working_dir: str, training_path: str, num_workers=1):
    """
    :param cfg: Hydra's config file
    :param dictionary: Fitted SupervisedDictionary's instance
    :param working_dir: Working directory of this code.
    :param training_path: Path to training data.
    :param num_workers: The number of workers of data loader.
    :return: List of DataLoaders.
    """
    return datasets2data_loaders(
        get_datasets(cfg, dictionary, working_dir, training_path),
        num_workers=num_workers
    )


def evaluation(model, device, test_data_loader, divide_by_num_data=True):
    """
    :param model: Models
    :param device: PyTorch's device instance
    :param test_data_loader: Test data loader
    :param divide_by_num_data: taking mean or not over samples
    :return: loss and accuracy
    """
    model.eval()
    loss = 0.
    correct = 0
    N = len(test_data_loader)

    with torch.no_grad():
        for sentence, label, _ in test_data_loader:
            if len(sentence[0]) == 0:
                continue
            sentence, label = sentence.to(device), label.to(device)
            output = model(sentence)
            loss += F.nll_loss(output, label).item()
            pred = output.argmax(1, keepdim=False)
            correct += pred.eq(label).sum().item()

    if divide_by_num_data:
        return loss / N, correct / N
    else:
        return loss, correct


def initialise_word_embeddigns_from_pretrained_embeddings(
        embeddings: KeyedVectors, dictionary: SupervisedDictionary,
        OOV_initialised_method: str, rnd: np.random.RandomState
):
    """
    :param embeddings: Gensim's KeyedVectors.
    :param dictionary: Fitted SupervisedDictionary.
    :param OOV_initialised_method: initialisation method for OOV words. Mean of all word vectors or uniform.
    :param rnd: np.random.RandomState for reproducibility.
    :return: Torch's floatTensor contains word embeddings.
    """
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
