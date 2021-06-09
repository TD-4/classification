# _*_coding:utf-8_*_
# @auther:FelixFu
# @Data:2021.4.16
# @github:https://github.com/felixfu520

import torch
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F
from torchvision.models.alexnet import alexnet


class AlexNet(torch.nn.Module):
    def __init__(self, num_class=1000, in_channels=3, pretrained=False, freeze_bn=False):
        super(AlexNet, self).__init__()
        self.model = alexnet(pretrained=pretrained)
        self.model.features[0] = torch.nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=11,
                                                 stride=4, padding=2)
        self.model.classifier[-1] = torch.nn.Linear(4096, num_class, bias=True)

    def forward(self,x):
        return self.model(x)