import torch
import torch.nn as nn
from utils.eval import evaluate_accuracy_gpu
from utils.tools import Accumulator, Timer
from utils.ploter import Animator


class Trainer:
    def __init__(self, model, optimizer, loss, device=None, init_weights=False):
        if not isinstance(model, nn.Module):
            raise ValueError("model must be a instance of nn.Module")
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.loss = loss
        # init weight
        if init_weights:

            def init_weights_fun(m):
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    nn.init.xavier_uniform_(m.weight)

            model.apply(init_weights_fun)
        # to device
        if device:
            self.model.to(device)
        self.figsize = (3.5, 2.5)

    def set_figsize(self, figsize):
        """set figsize for plot"""
        self.figsize = figsize

    def train(self, train_iter, num_epochs, accuracy, test_iter=None):
        legend = ["train loss", "train acc", "test loss", "test acc"]
        if test_iter is None:
            legend = legend[:2]
        animator = Animator(
            xlabel="epoch", xlim=[1, num_epochs], legend=legend, figsize=self.figsize
        )
        timer, num_batches = Timer(), len(train_iter)
        for epoch in range(num_epochs):
            metric = Accumulator(3)
            self.model.train()
            for i, (inputs, y) in enumerate(train_iter):
                timer.start()
                self.optimizer.zero_grad()
                inputs, y = inputs.to(self.device), y.to(self.device)
                y_hat = self.model(inputs)
                l = self.loss(y_hat, y)
                l.backward()
                self.optimizer.step()
                with torch.no_grad():
                    metric.add(l * inputs.shape[0], accuracy(y_hat, y), inputs.shape[0])
                timer.stop()
                train_l = metric[0] / metric[2]
                train_acc = metric[1] / metric[2]
                if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                    if test_iter:
                        animator.add(
                            epoch + (i + 1) / num_batches,
                            (train_l, train_acc, None, None),
                        )
                    else:
                        animator.add(
                            epoch + (i + 1) / num_batches, (train_l, train_acc)
                        )
            test_l, test_acc = evaluate_accuracy_gpu(
                self.model, accuracy, test_iter, self.loss, self.device
            )
            if test_iter:
                animator.add(epoch + 1, (None, None, test_l, test_acc))
        print(
            f"{metric[2] * num_epochs / timer.sum():.1f} examples/sec "
            f"on {str(self.device)}"
        )
        message = f"train loss {train_l:.3f}, train acc {train_acc:.3f}"
        if test_iter:
            message += f", test loss {test_l:.3f}, test acc {test_acc:.3f}"
        print(message)
