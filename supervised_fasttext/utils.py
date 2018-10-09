import numpy as np


class EarlyStopping(object):
    _valid_modes = ['min', 'max']

    def __init__(self, mode='min', min_delta=0., patience=10):
        if mode not in self._valid_modes:
            raise ValueError('mode {} is not supported. You must pass one of [{}] to `mode`.'.format(
                mode, ', '.join(self._valid_modes)))
        if patience <= 0:
            raise ValueError('`patient` must be positive.')

        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.num_bad_epochs = 0

        if mode == 'min':
            self._is_better = lambda a, best: a < best - self.min_delta
            self.best = np.finfo(np.float(0.)).max
        else:
            self._is_better = lambda a, best: a > best + self.min_delta
            self.best = np.finfo(np.float(0.)).min

    def is_stopped(self, metric):
        """
        :param metric: monitored value such as validation accuracy, validation loss.
        :return: Boolean
        """

        if np.isnan(metric):
            raise ValueError('The metric becomes `nan`. Stop training.')

        if self._is_better(metric, self.best):
            self.num_bad_epochs = 0
            self.best = metric
        else:
            self.num_bad_epochs += 1

        return self.num_bad_epochs >= self.patience
