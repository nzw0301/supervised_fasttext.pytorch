import torch
import torch.nn.functional as F
from torch import nn


class SupervisedFastText(nn.Module):
    def __init__(self, V: int, num_classes: int, embedding_dim=10, pre_trained_emb=None, freeze=True):
        """
        :param V: the size of set of words and n-grams
        :param num_classes: the number of classes
        :param embedding_dim: the number of dimensions of word vector
        """
        super(SupervisedFastText, self).__init__()

        self.embedding_dim = embedding_dim
        self.input2embeddings = nn.Embedding(num_embeddings=V, embedding_dim=embedding_dim, sparse=True)
        self.hidden2out = nn.Linear(in_features=embedding_dim, out_features=num_classes)

        # https://github.com/facebookresearch/fastText/blob/25d0bb04bf43d8b674fe9ae5722ef65a0856f5d6/src/fasttext.cc#L669
        if pre_trained_emb is None:
            self.reset_parameters_input2hidden()
        else:
            raise ValueError("Unimplemented.")
            # TODO: load pre-trained from word2vec format
            # self.input2embeddings.from_pretrained(pre_trained_emb, freeze)

        self.reset_parameters_hidden2output()

    def reset_parameters_input2hidden(self):
        upper = 1. / self.embedding_dim
        self.input2embeddings.weight.data.uniform_(-upper, upper)

    def reset_parameters_hidden2output(self):
        self.hidden2out.weight.data.zero_()

    def forward(self, input_bags: torch.Tensor):
        """
        :param input_bags: a bag-of-words. shape: (1, _)
        :return: log prob for labels. shape: (1, `num_classes`)
        """
        hidden = torch.mean(self.input2embeddings(input_bags), dim=1)
        return F.log_softmax(self.hidden2out(hidden), dim=1) # or addaptive softmax
