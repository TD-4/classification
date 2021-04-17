# _*_coding:utf-8_*_
# @auther:FelixFu
# @Date: 2021.4.14
# @github:https://github.com/felixfu520

import os
import json
import argparse

import torch
from torchsummary import summary

from utils import Logger
import dataloaders
import models
from utils import losses
from trainer import Trainer

import warnings
warnings.filterwarnings("ignore")


def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


def main(config, resume):
    train_logger = Logger()

    # DATA LOADERS
    train_loader = get_instance(dataloaders, 'train_loader', config)
    # # Test train_loader
    # import cv2
    # import numpy as np
    # for data, target in train_loader:
    #     # 使用matplotlib测试
    #     import matplotlib.pyplot as plt
    #     plt.imshow(data.numpy()[0].transpose((1, 2, 0)), cmap='Greys_r')
    #     plt.show()
    #
    #     # 使用cv2测试
    #     # cv2.imshow("image", data.numpy()[0].transpose((1, 2, 0)))
    #     # cv2.waitKey()
    val_loader = get_instance(dataloaders, 'val_loader', config)
    # # Test val_loader
    # import cv2
    # import numpy as np
    # for data, target in val_loader:
    #     # 使用matplotlib测试
    #     import matplotlib.pyplot as plt
    #     plt.imshow(data.numpy()[0].transpose((1, 2, 0)), cmap='Greys_r')
    #     plt.show()
    #
    #     # 使用cv2测试
    #     # cv2.imshow("image", data.numpy()[0].transpose((1, 2, 0)))
    #     # cv2.waitKey()

    # MODEL
    model = get_instance(models, 'arch', config, train_loader.dataset.num_classes)
    # print(f'\n{model}\n')
    # summary(model, (1, 224, 224))

    # LOSS
    loss = getattr(losses, config['loss'])(ignore_index=config['ignore_index'])

    # TRAINING
    trainer = Trainer(
        model=model,
        loss=loss,
        resume=resume,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        train_logger=train_logger)

    trainer.train()


if __name__ == '__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--configs', default='configs/MNIST_ResNet18_MSE_SGD.json', type=str,
                        help='Path to the configs file (default: configs.json)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()

    config = json.load(open(args.configs))
    if args.resume:
        print("resume ......")
        config = torch.load(args.resume)['configs']
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    main(config, args.resume)

