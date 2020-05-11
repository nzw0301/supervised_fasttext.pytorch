import numpy as np

_valid_initialised_methods = ['uniform', 'mean']


def check_hydra_conf(cfg) -> None:
    """
    Validate the hydra's config parameters
    :param cfg: Hydra's config instance.
    :return: None
    """
    assert 0 < cfg['parameters']['dim']
    assert 0 < cfg['parameters']['min_count']
    assert 0 < cfg['parameters']['epochs']
    assert 0. < cfg['parameters']['lr_update_rate']
    assert 0 < cfg['parameters']['patience']
    assert cfg['parameters']['metric'] in ['loss', 'acc']
    assert cfg['parameters']['initialize_oov'] in _valid_initialised_methods
    assert 0. < cfg['optuna']['lr_min']
    assert cfg['optuna']['lr_min'] <= cfg['optuna']['lr_max']
    assert cfg['optuna']['num_trials'] > 0


class EarlyStopping(object):
    _valid_modes = ['min', 'max']

    def __init__(
            self,
            mode='min',
            min_delta=0.,
            patience=10
    ):
        assert mode in self._valid_modes, \
            'mode {} is not supported. You must pass one of [{}] to `mode`.'.format(mode, ', '.join(self._valid_modes))
        assert patience > 0, \
            '`patient` must be positive.'

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

    def is_stopped(self, metric: float):
        """
        :param metric: Monitored value such as validation accuracy or validation loss.
        :return: Boolean
        """

        assert not np.isnan(metric), 'The metric becomes `nan`. Stop training.'

        if self._is_better(metric, self.best):
            self.num_bad_epochs = 0
            self.best = metric
        else:
            self.num_bad_epochs += 1

        return self.num_bad_epochs >= self.patience
