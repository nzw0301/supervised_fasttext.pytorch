import numpy as np
import torch


class SentenceDataset(torch.utils.data.dataset.Dataset):
    def __init__(
            self,
            data: np.ndarray,
            targets: np.ndarray,
            size_vocab: int,
            train=True,
    ):
        """
        :param data: numpy array of numpy array. Each element is np.array of int array.
        :param targets: numpy array of int. Each element represents label.
        :param size_vocab: the size of word vocab. It is used to calculate sentence lengths that don't contain n-gram.
        :param train: Boolean. If true, sentence length is stored, else it is empty.
        """

        self.data = data
        self.targets = targets
        self.train = train
        self.lengths = []
        if self.train:
            for sentence in data:
                self.lengths.append(np.sum(sentence < size_vocab))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> tuple:
        sample = torch.from_numpy(self.data[index])
        target = self.targets[index]
        if self.train:
            num_tokens = self.lengths[index]
        else:
            num_tokens = 0

        return sample, target, num_tokens
