# _*_coding:utf-8_*_
# @auther:FelixFu
# @Date: 2021.4.16
# @github:https://github.com/felixfu520

import torch
from torchvision.models.densenet import densenet169
from torchvision.models.densenet import densenet121
from torchvision.models.densenet import densenet161
from torchvision.models.densenet import densenet201


class Densenet121(torch.nn.Module):
    def __init__(self, num_class=1000, in_channels=3, pretrained=False, **kwargs):
        super(Densenet121, self).__init__()
        self.model = densenet121(pretrained=pretrained)
        self.model.features[0] = torch.nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7,
                                      stride=2, padding=3, bias=False)
        self.model.classifier = torch.nn.Linear(in_features=1024, out_features=num_class, bias=True)

    def forward(self, x):
        return self.model(x)