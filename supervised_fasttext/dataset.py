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
        :param train: Boolean. If true, sentence length is stored and remove empty sentences, else it is empty.
        """

        self.data = []
        self.targets = []
        self.lengths = []
        self.train = train

        if self.train:
            for (sentence, label) in zip(data, targets):
                length = np.sum(sentence < size_vocab)
                if length > 0:
                    self.data.append(sentence)
                    self.targets.append(label)
                    self.lengths.append(length)
            self.data = np.array(self.data)
            self.targets = np.array(self.targets)
        else:
            self.data = data
            self.targets = targets


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
