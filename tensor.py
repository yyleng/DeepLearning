import torch
import torch.nn as nn
from utils import trainer
from utils import dataloader
from utils import eval

num_inputs, num_outputs, num_hiddens = 784, 10, 256
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(num_inputs, num_hiddens),
    nn.ReLU(),
    nn.Linear(num_hiddens, num_outputs),
)

batch_size = 256
train_iter, test_iter = dataloader.load_offical_data(
    batch_size, dataloader.DatasetName.FASHION_MNIST
)

num_epochs, lr = 100, 0.1

clasify_trainer = trainer.Trainer(
    model=net,
    optimizer=torch.optim.SGD(net.parameters(), lr=lr),
    loss=nn.CrossEntropyLoss(),
    device=None,
    init_weights=True,
)
clasify_trainer.set_figsize((7, 5))
clasify_trainer.train(train_iter, num_epochs, eval.accuracy_for_classify, test_iter)
