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
    # for data, target in train_loader:
    #     # 使用matplotlib测试
    #     MEAN = [0.39755441968379984]
    #     STD = [0.09066523780114362]
    #     restore_transform = transforms.Compose([local_transforms.DeNormalize(MEAN, STD), transforms.ToPILImage()])
    #     image = restore_transform(data[0])
    #     plt.imshow(image, cmap="gray")
    #     plt.text(226, 1, str(target.numpy()[0]))
    #     plt.show()
    #     # print(target.numpy()[0])   # , cmap="gray"
    #     # print("Finshed")

    val_loader = get_instance(dataloaders, 'val_loader', config)
    # # Test val_loader
    # for data, target in val_loader:
    #     # 使用matplotlib测试
    #     MEAN = [0.39755441968379984]
    #     STD = [0.09066523780114362]
    #     restore_transform = transforms.Compose([local_transforms.DeNormalize(MEAN, STD), transforms.ToPILImage()])
    #     image = restore_transform(data[0])
    #     plt.imshow(image, cmap="gray")
    #     plt.text(226, 1, str(target.numpy()[0]))
    #     # plt.show()
    #     # print(target.numpy()[0])   # , cmap="gray"
    #     # print("Finshed")

    # MODEL
    model = get_instance(models, 'arch', config, train_loader.dataset.num_classes)
    print(f'\n{model}\n')
    summary(model, (1, 224, 224), device="cpu")

    # LOSS
    weight = torch.from_numpy(np.array(config['weight'])).float()
    loss = getattr(losses, config['loss'])(ignore_index=config['ignore_index'], weight=weight)
    # loss = getattr(losses, config['loss'])(ignore_index=config['ignore_index'])

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
    parser.add_argument('-c', '--configs', default='configs/MDD_AlexNet_CEL_SGD.json', type=str,
                        help='Path to the configs file (default: configs.json)')
    parser.add_argument('-r', '--resume', default="", type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()

    config = json.load(open(args.configs))
    if args.resume:
        print("resume config......")
        if True:  # 使用pth中的config
            config = torch.load(args.resume)['configs']
            # config["train_loader"]["args"]["batch_size"] = 20
            # config["val_loader"]["args"]["batch_size"] = 20
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    main(config, args.resume)

