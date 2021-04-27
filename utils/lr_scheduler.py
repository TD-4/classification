import math
from torch.optim.lr_scheduler import StepLR


class StepLR_(StepLR):
    def __init__(self, optimizer, step_size=1, gamma=0.5):
        super(StepLR_, self).__init__(optimizer, step_size, gamma)


if __name__ == "__main__":
    import torchvision
    import torch
    import matplotlib.pylab as plt

    resnet = torchvision.models.resnet34()
    params = {
        "lr": 0.01,
        "weight_decay": 0.001,
        "momentum": 0.9
    }
    optimizer = torch.optim.SGD(params=resnet.parameters(), **params)

    epochs = 200
    iters_per_epoch = 10
    lrs = []
    mementums = []
    lr_scheduler = StepLR_(optimizer, 10, 0.1)

    for epoch in range(epochs):
        for i in range(iters_per_epoch):
            pass
        lr_scheduler.step()
        lrs.append(optimizer.param_groups[0]['lr'])

    plt.ylabel("learning rate")
    plt.xlabel("iteration")
    plt.plot(lrs)
    plt.show()

