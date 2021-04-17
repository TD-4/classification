import math
from torch.optim.lr_scheduler import StepLR


class StepLR_(StepLR):
    def __init__(self, optimizer, step_size=1, gamma=0.5):
        super(StepLR_, self).__init__(optimizer, step_size, gamma)
