import argparse

import torch
from torch import optim
import torch.nn.functional as F
from .model import SupervisedFastText

from torchtext.data import TabularDataset, BucketIterator, Field, LabelField


def test(args, model, device, test_iter):
    model.eval()
    test_loss = 0.
    correct = 0
    N = len(test_iter)
    test_iter.init_epoch()

    with torch.no_grad():
        for batch in test_iter:
            data, target = batch.text, batch.label
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target).item()
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    return test_loss / N, 100. * correct / N


def main():
    parser = argparse.ArgumentParser(description='PyTorch supervised fastText example')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--lr-update-rate', type=int, default=100, metavar='ulr',
                        help='change learning rate schedule (default: 100)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--path', type=str, default='./',
                        help='the path to the data files (default: ./)')
    parser.add_argument('--train', type=str, default='train.tsv',
                        help='the file name of training data (default: train.tsv)')
    parser.add_argument('--test', type=str, default='test.tsv',
                        help='the fine name of test data (default: train.tsv)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device('cuda' if use_cuda else 'cpu')

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    path = args.path
    TEXT = Field(pad_token=None, unk_token=None, batch_first=True)  # do not use padding and <unk>
    LABEL = LabelField()

    # TODO load val data or split val data
    train_data, test_data = TabularDataset.splits(
        path=path, train=args.train,
        test=args.test, format='tsv',
        fields=[('label', LABEL), ('text', TEXT)])

    TEXT.build_vocab(train_data)
    LABEL.build_vocab(train_data)
    train_iter = BucketIterator(dataset=train_data, batch_size=1, repeat=False)
    test_iter = BucketIterator(dataset=test_data, batch_size=1, repeat=False)

    epochs = args.epochs
    model = SupervisedFastText(V=len(TEXT.vocab), num_classes=len(LABEL.vocab)).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    # parameters for update learning rate
    num_tokens = 0
    train_iter.init_epoch()
    for batch in train_iter:
        num_tokens += batch.text.shape[1]
    learning_rate_schedule = args.lr_update_rate
    total_num_processed_tokens_in_training = epochs * num_tokens
    num_processed_tokens = 0
    local_processed_tokes = 0

    for epoch in range(1, epochs + 1):
        # begin train phase
        # NOTE: to update learning rate, I do not use `train` function
        sum_loss = 0.
        N = len(train_iter)

        model.train()
        train_iter.init_epoch()
        for batch_idx, batch in enumerate(train_iter):
            data, target = batch.text, batch.label
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()

            # update learning rate
            local_processed_tokes += data.shape[1]
            if local_processed_tokes > learning_rate_schedule:
                num_processed_tokens += local_processed_tokes
                local_processed_tokes = 0
                progress = num_processed_tokens / total_num_processed_tokens_in_training
                optimizer.param_groups[0]['lr'] = args.lr * (1. - progress)
        train_loss = sum_loss / N
        # end train phase

        val_loss, val_acc = test(args, model, device, test_iter)

        progress = num_processed_tokens / total_num_processed_tokens_in_training  # approximated progress
        print('Progreess: {:.7f} Average train loss: {:.4f}, Average val loss: {:.4f}, Accuracy: {:.1f}%'.format(
            progress, train_loss, val_loss, val_acc
        ))


if __name__ == '__main__':
    main()
