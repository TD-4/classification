# _*_coding:utf-8_*_
# @auther:FelixFu
# @Date: 2021.4.14
# @github:https://github.com/felixfu520

import os
import json
import argparse
import numpy as np

import torch
from torchsummary import summary

from utils import Logger
import dataloaders
import models
from utils import losses
from trainer import Trainer

import warnings
warnings.filterwarnings("ignore")

# 绘制数据时用到
import matplotlib.pyplot as plt
from torchvision import transforms
from utils import transforms as local_transforms


def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


def main(config, resume):
    train_logger = Logger()

    # DATA LOADERS
    train_loader = get_instance(dataloaders, 'train_loader', config)
    # # Test train_loader
    # for data, target, image_path in train_loader:
    #     # 使用matplotlib测试
    #     MEAN = [0.3858034032292721]
    #     STD = [0.12712721340420535]
    #     restore_transform = transforms.Compose([local_transforms.DeNormalize(MEAN, STD), transforms.ToPILImage()])
    #     for i, data_i in enumerate(data):
    #         image = restore_transform(data_i)
    #         plt.imshow(image, cmap="gray")
    #         plt.text(156, 1, str(target.numpy()[i]))
    #         plt.text(-100,-10, str(image_path[i]))
    #         plt.show()

    val_loader = get_instance(dataloaders, 'val_loader', config)

    # MODEL
    model = get_instance(models, 'arch', config, train_loader.dataset.num_classes)
    print(f'\n{model}\n')
    # summary(model, (1, 150, 150), device="cpu")

    # LOSS
    weight = torch.from_numpy(np.array(config['weight'])).float()
    loss = getattr(losses, config['loss'])(ignore_index=config['ignore_index'], weight=weight)

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
    parser = argparse.ArgumentParser(description='LCD Classification Training')
    parser.add_argument('-c', '--configs', default='configs/BDD_ResNet34_CEL_SGD.json', type=str,
                        help='Path to the configs file (default: configs.json)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()

    config = json.load(open(args.configs))
    if args.resume:
        print("Resume config......")
        config = torch.load(args.resume)['configs']
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    main(config, args.resume)

