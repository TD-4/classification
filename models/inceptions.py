# _*_coding:utf-8_*_
# @auther:FelixFu
# @Date: 2021.4.16
# @github:https://github.com/felixfu520

import torch
from torchvision.models.inception import inception_v3


class inception_v3_(torch.nn.Module):
    def __init__(self, num_class=1000, in_channels=3, pretrained=False, **kwargs):
        super(inception_v3_, self).__init__()
        self.model = inception_v3(pretrained=pretrained)
        self.model.Conv2d_1a_3x3.conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3,
                                      stride=2, bias=False)
        self.model.fc = torch.nn.Linear(in_features=2048, out_features=num_class, bias=True)

    def forward(self, x):
        return self.model(x)
