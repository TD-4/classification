import argparse
import scipy
import os
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


def _val_augmentation(image_path, crop_size=224, in_channels=1,histogram=False):
    if in_channels == 1:
        # 修改支持中文路径
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    elif in_channels == 3:
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if crop_size:
        if in_channels == 1:
            h, w = image.shape
        else:
            h, w, _ = image.shape
        # Scale the smaller side to crop size
        if h < w:
            h, w = (crop_size, int(crop_size * w / h))
        else:
            h, w = (int(crop_size * h / w), crop_size)

        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)

        # Center Crop
        h, w = image.shape
        start_h = (h - crop_size) // 2
        start_w = (w - crop_size) // 2
        end_h = start_h + crop_size
        end_w = start_w + crop_size
        image = image[start_h:end_h, start_w:end_w]

    # Histogram
    if histogram and in_channels == 1:
        rows, cols = image.shape
        flat_gray = image.reshape((cols * rows,)).tolist()
        A = min(flat_gray)
        B = max(flat_gray)
        image = np.uint8(255 / (B - A) * (image - A) + 0.5)

    return image

def main():
    args = parse_arguments()
    config = json.load(open(args.configs))

    # Dataset used for training the model
    loader = getattr(dataloaders, config['train_loader']['type'])(**config['train_loader']['args'])
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(loader.MEAN, loader.STD)
    num_classes = loader.dataset.num_classes

    # Model
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

    image_folders = args.images
    all_images = [os.path.join(image_folders, name) for name in os.listdir(image_folders)]

    label_maps = {}
    with open("dataloaders/labels/MedMnist_labels.txt", 'r') as f:
        for line in f.readlines():
            label_maps[line.strip().split(':')[0]] = line.strip().split(':')[1]


    result = {}
    with torch.no_grad():
        tbar = tqdm(all_images, ncols=100)
        for img_file in tbar:
            image = _val_augmentation(img_file)
            input = normalize(to_tensor(image)).unsqueeze(0)

            prediction = model(input.to(device))
            prediction = prediction.squeeze(0).cpu().numpy()
            prediction = F.softmax(torch.from_numpy(prediction), dim=0).argmax(0).cpu().numpy()
            result[img_file] = label_maps[str(prediction.item())]

    with open(args.output + "/result.json", 'w') as f:
        js = json.dumps(result)
        f.write(js)
        f.close()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('-c', '--configs', default='configs/MNIST_Resnet18_MSE_SGD.json',type=str,
                        help='The configs used to train the model')
    parser.add_argument('-m', '--model', default='saved/Resnet18/04-17_17-01/checkpoint-epoch1.pth', type=str,
                        help='Path to the .pth model checkpoint to be used in the prediction')
    parser.add_argument('-i', '--images', default=None, type=str,
                        help='Path to the images to be segmented')
    parser.add_argument('-o', '--output', default='outputs', type=str,  
                        help='Output Path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
