import torch
import torch.nn as nn
from utils.tools import Accumulator


def accuracy_for_classify(y_hat: torch.Tensor, y: torch.Tensor):
    """Compute the number of correct predictions."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = torch.argmax(y_hat, axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(torch.sum(cmp.type(y.dtype)))


def evaluate_accuracy_gpu(net, accuracy, test_iter=None, loss=None, device=None):
    """Compute the accuracy for a model on a dataset using a GPU."""
    if test_iter is None:
        return None, None
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    metric = Accumulator(3) if loss else Accumulator(2)
    print(len(metric))

    with torch.no_grad():
        for X, y in test_iter:
            batch_size = X.shape[0]
            if isinstance(X, list):
                # Required for BERT Fine-tuning (to be covered later)
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            if loss:
                l = loss(y_hat, y)
                metric.add(l * batch_size, accuracy(y_hat, y), batch_size)
                return metric[0] / metric[2], metric[1] / metric[2]
            metric.add(accuracy(y_hat, y), batch_size)
            return metric[0] / metric[1], None


def evaluate_loss(net, data_iter, loss):
    """Evaluate the loss of a model on the given dataset."""
    metric = Accumulator(2)  # Sum of losses, data instances
    for X, y in data_iter:
        out = net(X)
        y = torch.reshape(y, out.shape)
        l = loss(out, y)
        metric.add(torch.sum(l), torch.numel(l))
    return metric[0] / metric[1]
