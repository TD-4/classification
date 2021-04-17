# _*_coding:utf-8_*_
# @auther:FelixFu
# @Date: 2021.4.16
# @github:https://github.com/felixfu520

import torch
from torchvision.models.resnet import resnet18
from torchvision.models.resnet import resnet34
from torchvision.models.resnet import resnet50
from torchvision.models.resnet import resnet101
from torchvision.models.resnet import resnet152


class Resnet18(torch.nn.Module):
    def __init__(self, num_class=1000, in_channels=3, pretrained=False, freeze_bn=False):
        super(Resnet18, self).__init__()
        self.model = resnet18(pretrained=pretrained)
        self.model.conv1 = torch.nn.Conv2d(in_channels=in_channels,out_channels=64, kernel_size=7,
                                      stride=2, padding=3,bias=False)
        self.model.fc = torch.nn.Linear(512, num_class)

    def forward(self, x):
        return self.model(x)