import math
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import _LRScheduler


class StepLR_(StepLR):
    def __init__(self, optimizer, step_size=1, gamma=0.5):
        super(StepLR_, self).__init__(optimizer, step_size, gamma)


class Poly(_LRScheduler):
    def __init__(self, optimizer, num_epochs, iters_per_epoch=0, warmup_epochs=0, last_epoch=-1):
        self.iters_per_epoch = iters_per_epoch
        self.cur_iter = 0
        self.N = num_epochs * iters_per_epoch
        self.warmup_iters = warmup_epochs * iters_per_epoch
        super(Poly, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        T = self.last_epoch * self.iters_per_epoch + self.cur_iter
        factor = pow((1 - 1.0 * T / self.N), 0.9)
        if self.warmup_iters > 0 and T < self.warmup_iters:
            factor = 1.0 * T / self.warmup_iters

        self.cur_iter %= self.iters_per_epoch
        self.cur_iter += 1
        return [base_lr * factor for base_lr in self.base_lrs]


class ConsineAnnealingLR_(CosineAnnealingLR):
    def __init__(self, optimizer, step_size=1, gamma=0.5):
        super(ConsineAnnealingLR_, self).__init__(optimizer, T_max=10, eta_min=0)


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
    # lr_scheduler = StepLR_(optimizer, 10, 0.1)
    lr_scheduler = ConsineAnnealingLR_(optimizer, 10, 0)
    for epoch in range(epochs):
        for i in range(iters_per_epoch):
            pass
        lr_scheduler.step()
        lrs.append(optimizer.param_groups[0]['lr'])

    plt.ylabel("learning rate")
    plt.xlabel("iteration")
    plt.plot(lrs)
    plt.show()

