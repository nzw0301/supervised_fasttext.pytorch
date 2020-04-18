import torch
import numpy


class SentenceDataset(torch.utils.data.dataset.Dataset):
    def __init__(
            self,
             data: numpy.ndarray,
             targets: numpy.ndarray,
             train=True
    ):
        """
        :param data: numpy array of numpy array. Each element is np.array of itnt array.
        :param targets: numpy array of int. Each element represents label.
        :param train: Boolean.
        """

        self.data = data
        self.targets = targets
        self.train = train

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> tuple:
        sample = torch.from_numpy(self.data[index])
        target = self.targets[index]
        return sample, target
