import argparse
import scipy
import os
import shutil
import numpy as np
import json
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from scipy import ndimage
from tqdm import tqdm
from math import ceil
from glob import glob
from itertools import chain
from PIL import Image
import dataloaders
import models
from collections import OrderedDict
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
import matplotlib.pyplot as plt
from torchvision import transforms
from utils import transforms as local_transforms
import warnings
warnings.filterwarnings("ignore")



def confusionMatrix(preds, labels, conf_matrix):
    preds = torch.argmax(preds, 1)
    for t, p in zip(labels, preds):
        conf_matrix[t][p] += 1
    return conf_matrix

def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


def main():
    args = parse_arguments()
    config = json.load(open(args.configs))
    config["val_loader"]["args"]["shuffle"] = True

    # 1、Load Data
    if args.images is None:  # 如果args.images为None，则使用configs.json中的val集
        val_loader = get_instance(dataloaders, 'val_loader', config)
        # # Test train_loader
        # for data, target, image_path in val_loader:
        #     # 使用matplotlib测试
        #     MEAN = [0.3858034032292721]
        #     STD = [0.12712721340420535]
        #     # restore_transform = transforms.Compose([local_transforms.DeNormalize(MEAN, STD), transforms.ToPILImage()])    # 有归一化
        #     restore_transform = transforms.Compose([transforms.ToPILImage()])
        #     for i, data_i in enumerate(data):
        #         image = restore_transform(data_i)
        #         plt.imshow(image, cmap="gray")
        #         plt.text(156, 1, str(target.numpy()[i]))
        #         plt.text(-100,-10, str(image_path[i]))
        #         plt.show()
    else:   # 如果args.images不为None，则使用images文件夹中的图片 TODO:
        image_folders = args.images
        all_images = [os.path.join(image_folders, name) for name in os.listdir(image_folders)]

    # Model
    num_classes = 29
    model = getattr(models, config['arch']['type'])(num_classes, **config['arch']['args'])
    availble_gpus = list(range(torch.cuda.device_count()))
    device = torch.device('cuda:0' if len(availble_gpus) > 0 else 'cpu')

    # Load checkpoint
    checkpoint = torch.load(args.model, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint.keys():
        checkpoint = checkpoint['state_dict']
    # If during training, we used data parallel
    if 'module' in list(checkpoint.keys())[0] and not isinstance(model, torch.nn.DataParallel):
        # for gpu inference, use data parallel
        if "cuda" in device.type:
            model = torch.nn.DataParallel(model)
        else:
        # for cpu inference, remove module
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k[7:]
                new_state_dict[name] = v
            checkpoint = new_state_dict
    # load
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    # 获取labels，即类别名称
    label_path = os.path.join(config["train_loader"]["args"]["data_dir"], "labels.txt")
    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            labels.append(line.split()[0])
    conf_matrix = [[0 for j in range(num_classes)] for i in range(num_classes)]
    tbar = tqdm(val_loader, ncols=130)
    with torch.no_grad():
        for batch_idx, (data, target, image_path) in enumerate(tbar):
            if device == torch.device("cuda:0"):
                data, target = data.to(device), target.to(device)
            # LOSS
            output = model(data)
            for o,l,p in zip(output.cpu().numpy(),target.cpu().numpy(),image_path):
                if o.argmax() == l: # 预测正确
                    pass
                else:   # 预测错误
                    image_id = p.split("/")[-1]
                    pre_image_id = "label-" + str(l) + "-" + labels[l] + "___pred-" + str(o.argmax()) + "-" + labels[o.argmax()] + "___"
                    output_path = os.path.join(args.output, pre_image_id + image_id)
                    shutil.copy(p.encode('utf8'), output_path.encode('utf8'))

            confusion_matrix = confusionMatrix(output, target, conf_matrix)

    # print 混淆矩阵
    print("{0:10}".format(""), end="")
    for name in labels:
        print("{0:10}".format(name), end="")
    print("{0:10}".format("Precision"))
    for i in range(num_classes):
        print("{0:10}".format(labels[i]), end="")
        for j in range(num_classes):
            if i == j:
                print("{0:10}".format(str("-" + str(confusion_matrix[i][j])) + "-"), end="")
            else:
                print("{0:10}".format(str(confusion_matrix[i][j])), end="")
        precision = 0.0 + confusion_matrix[i][i] / sum(confusion_matrix[i])
        print("{0:.4f}".format(precision))


def parse_arguments():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('-c', '--configs', default='saved/BDD-Resnet34/06-10_02-00/configs.json', type=str,
                        help='The configs used to train the model')
    parser.add_argument('-m', '--model', default='saved/BDD-Resnet34/06-10_02-00/best_model.pth', type=str,
                        help='Path to the .pth model checkpoint to be used in the prediction')
    parser.add_argument('-i', '--images', default=None, type=str,
                        help='Path to the images to be segmented')
    parser.add_argument('-o', '--output', default='outputs', type=str,
                        help='Output Path')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()
