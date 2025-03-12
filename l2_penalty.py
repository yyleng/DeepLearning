import torch
import torch.nn as nn
from d2l import torch as d2l
from utils import dataloader
from utils import ploter
from utils import alog
from utils import eval

n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5

# 模型函数原型
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05

# 生成数据集
train_data = dataloader.synthetic_data(true_w, true_b, n_train)
test_data = dataloader.synthetic_data(true_w, true_b, n_test)

# 加载数据集
train_iter = dataloader.load_array(train_data, batch_size)
test_iter = dataloader.load_array(test_data, batch_size, shuffle=False)


# 初始化模型参数
def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return w, b


# L2范数惩罚
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2


def train(lambd):
    w, b = init_params()
    net, loss = lambda X: torch.matmul(X, w) + b, nn.MSELoss()
    num_epochs, lr = 100, 0.003
    animator = ploter.Animator(
        xlabel="epochs",
        ylabel="loss",
        yscale="log",
        xlim=[5, num_epochs],
        legend=["train", "test"],
    )
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            alog.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(
                epoch + 1,
                (
                    eval.evaluate_loss(net, train_iter, loss),
                    eval.evaluate_loss(net, test_iter, loss),
                ),
            )
    print("L2 norm of w:", torch.norm(w).item())


train(0)
torch.optim.SGD()
