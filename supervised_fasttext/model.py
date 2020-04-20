import torch
import torch.nn.functional as F
from torch import nn


class SupervisedFastText(nn.Module):
    def __init__(
            self,
            V: int,
            num_classes: int,
            embedding_dim=10,
            pretrained_emb=None,
            freeze=True,
            pooling="mean"
    ):
        """
        :param V: the size of set of words and n-grams
        :param num_classes: the number of classes
        :param embedding_dim: the number of dimensions of word vector
        :param pretrained_emb: torch.floatTensor. Pretrained word embeddings.
        :param freeze: Boolean. If it is true and `pre_trained_emb` is not None,
            word embeddings are re-trained on the supervised data.
        :param pooling: pooling method over words and ngrams in the sentence.
            Valid values in [mean, sum, min, max, min-max]
        """
        super(SupervisedFastText, self).__init__()

        self.embedding_dim = embedding_dim
        self.pooling = self._define_pooling(pooling)
        self.input2embeddings = nn.Embedding(num_embeddings=V, embedding_dim=embedding_dim, sparse=True)

        if pooling == "min-max":
            num_hidden = self.embedding_dim * 2
        else:
            num_hidden = self.embedding_dim

        self.hidden2out = nn.Linear(in_features=num_hidden, out_features=num_classes)

        # https://github.com/facebookresearch/fastText/blob/25d0bb04bf43d8b674fe9ae5722ef65a0856f5d6/src/fasttext.cc#L669
        if pretrained_emb is None:
            self.reset_parameters_input2hidden()
        else:
            self.input2embeddings = nn.Embedding.from_pretrained(pretrained_emb, freeze)

        self.reset_parameters_hidden2output()

    def reset_parameters_input2hidden(self):
        upper = 1. / self.embedding_dim
        self.input2embeddings.weight.data.uniform_(-upper, upper)

    def reset_parameters_hidden2output(self):
        self.hidden2out.weight.data.zero_()

    @staticmethod
    def _define_pooling(pool: str):
        if pool == "mean":
            return lambda x: torch.mean(x, dim=1)
        if pool == "sum":
            return lambda x: torch.sum(x, dim=1)
        elif pool == "max":
            return lambda x: torch.max(x, dim=1)[0]
        elif pool == "min":
            return lambda x: torch.min(x, dim=1)[0]
        elif pool == "min-max":
            return lambda x: torch.cat((torch.min(x, dim=1)[0], torch.max(x, dim=1)[0]), dim=1)
        else:
            raise ValueError("{} is not supported".format(pool))

    def forward(
            self,
            input_bags: torch.Tensor
    ):
        """
        :param input_bags: a bag-of-words. shape: (1, _)
        :return: log prob for labels. shape: (1, `num_classes`)
        """

        embeddings = self.input2embeddings(input_bags)
        hidden = self.pooling(embeddings)
        return F.log_softmax(self.hidden2out(hidden), dim=1)
