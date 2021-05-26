# _*_coding:utf-8_*_
# @auther:FelixFu
# @Date: 2021.4.16
# @github:https://github.com/felixfu520

import torch
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F
from torchvision.models.alexnet import alexnet


class AlexNet_(torch.nn.Module):
    def __init__(self, num_class=1000, in_channels=3, pretrained=False, freeze_bn=False):
        super(AlexNet_, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=48, kernel_size=11, stride=4),  # (b x 48 x 35 x 35)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.000099999, beta=0.75, k=2),  # section 3.3
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 48 x 17 x 17)

            nn.Conv2d(48, 128, 5, padding=2),  # (b x 128 x 17 x 17)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.00001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 128 x 8 x 8)

            nn.Conv2d(128, 192, 3, padding=1),  # (b x 192 x 8 x 8)
            nn.ReLU(),
            nn.Conv2d(192, 192, 3, padding=1, groups=2),  # (b x 192 x 8 x 8)
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),  # (b x 192 x 4 x 4)
        )
        # classifier is just a name for linear layers
        self.classifier = nn.Sequential(
            nn.Linear(in_features=3072, out_features=419),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=419, out_features=num_class),

        )
        self.init_bias()  # initialize bias

    def init_bias(self):
        cur_path = os.path.abspath(os.path.dirname(__file__))
        conv0_w = np.load(os.path.join(cur_path, "weights/l0w.npy"))
        self.net[0].weight.data = torch.from_numpy(conv0_w)
        conv0_b = np.load(os.path.join(cur_path, "weights/l0b.npy"))
        self.net[0].bias.data = torch.from_numpy(conv0_b)

        conv4_w = np.load(os.path.join(cur_path, "weights/l4w.npy"))
        self.net[4].weight.data = torch.from_numpy(conv4_w)
        conv4_b = np.load(os.path.join(cur_path, "weights/l4b.npy"))
        self.net[4].bias.data = torch.from_numpy(conv4_b)

        conv8_w = np.load(os.path.join(cur_path, "weights/l8w.npy"))
        self.net[8].weight.data = torch.from_numpy(conv8_w)
        conv8_b = np.load(os.path.join(cur_path, "weights/l8b.npy"))
        self.net[8].bias.data = torch.from_numpy(conv8_b)

        conv10_w = np.load(os.path.join(cur_path, "weights/l10w.npy"))
        self.net[10].weight.data = torch.from_numpy(conv10_w)
        conv10_b = np.load(os.path.join(cur_path, "weights/l10b.npy"))
        self.net[10].bias.data = torch.from_numpy(conv10_b)

        line13_w = np.load(os.path.join(cur_path, "weights/l13w.npy"))
        self.classifier[0].weight.data = torch.from_numpy(line13_w)
        line13_b = np.load(os.path.join(cur_path, "weights/l13b.npy"))
        self.classifier[0].bias.data = torch.from_numpy(line13_b)

        line16_w = np.load(os.path.join(cur_path, "weights/l16w.npy"))
        self.classifier[3].weight.data = torch.from_numpy(line16_w)
        line16_b = np.load(os.path.join(cur_path, "weights/l16b.npy"))
        self.classifier[3].bias.data = torch.from_numpy(line16_b)

    def forward(self, x):
        """
        Pass the input through the net.
        Args:
            x (Tensor): input tensor
        Returns:
            output (Tensor): output tensor
        """
        x = self.net(x)
        x = x.view(-1, 192 * 4 * 4)  # reduce the dimensions for linear layer input
        x = self.classifier(x)
        return x
        # return F.log_softmax(x, dim=1)

class _AlexNet(torch.nn.Module):
    def __init__(self, num_class=1000, in_channels=3, pretrained=False, freeze_bn=False):
        super(_AlexNet, self).__init__()
        self.model = alexnet(pretrained=pretrained)
        self.model.features[0] = torch.nn.Conv2d(in_channels=in_channels,out_channels=64, kernel_size=11,
                                      stride=4, padding=2)
        self.model.classifier[-1] = torch.nn.Linear(4096, num_class, bias=True)

    def forward(self, x):
        return self.model(x)