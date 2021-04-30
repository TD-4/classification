import numpy as np
import os
import time
import argparse
import json
from PIL import Image
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torchvision.transforms import ToTensor
import torch.nn.functional as F

import onnxruntime as ort

from torchvision import transforms


def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


def norm_myself(tensor, mean, std):
    return (tensor -mean) /std


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


def test_pytorch(args):
    # ----------------------------set the device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('running on device ' + str(device))

    # ----------------------------load the image
    # Dataset used for training the model
    MEAN = [0.45734706]
    STD = [0.23965294]
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(MEAN, STD)
    num_classes = 11

    image_folders = args.images
    all_images = [os.path.join(image_folders, name) for name in os.listdir(image_folders)]

    # ----------------------------load the model
    import models
    num_classes = 11
    config = json.load(open(args.configs))
    model = get_instance(models, 'arch', config, num_classes).to(device)  # 定义模型
    checkpoint = torch.load(args.model)
    base_weights = checkpoint["state_dict"]  # module.model.conv1.weight格式，而model的权重是model.conv1.weight格式
    print('Loading base network...')

    from collections import OrderedDict  # 导入此模块
    new_state_dict = OrderedDict()
    for k, v in base_weights.items():
        name = k[7:]  # remove `module.`，即只取module.model.conv1.weights的后面几位
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)  # 加载权重
    print("loaded weight")
    model.eval()

    # ----------------------------get output
    t3 = time.time()
    result = []
    with torch.no_grad():
        tbar = tqdm(all_images, ncols=100)
        for img_file in tbar:
            image = _val_augmentation(img_file)
            # input[channel] = (input[channel] - mean[channel]) / std[channel]
            input = normalize(to_tensor(image)).unsqueeze(0)

            prediction = model(input.to(device))  # tensor([[ 7.4939, -2.3476, -0.6207, -1.6629, -2.1027,  2.9507,  1.4629, -0.5033, -2.9737, -1.3985, -3.4913]], device='cuda:0')
            prediction = prediction.squeeze(0).cpu().numpy()
            prediction = F.softmax(torch.from_numpy(prediction), dim=0).argmax(0).cpu().numpy()
            result.append(prediction.item())
    t4 = time.time()
    print("Inference time with the PyTorch model: {}".format(t4 - t3))
    print("Inference result:", result)


def test_onnx(args):
    # ----------------------------set the device
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print('running on device ' + str(device))

    # ----------------------------load the image
    # Dataset used for training the model
    MEAN = [0.45734706]
    STD = [0.23965294]

    image_folders = args.images
    all_images = [os.path.join(image_folders, name) for name in os.listdir(image_folders)]

    # ----------------------------load the model
    sess = ort.InferenceSession(args.onnx)
    print('Loading onnx...')

    # 模型输入
    input_name = sess.get_inputs()[0].name
    # print("Input name  :", input_name)
    # input_shape = sess.get_inputs()[0].shape
    # print("Input shape :", input_shape)
    # input_type = sess.get_inputs()[0].type
    # print("Input type  :", input_type)

    # 模型输出
    output_name = sess.get_outputs()[0].name
    # print("Output name  :", output_name)
    # output_shape = sess.get_outputs()[0].shape
    # print("Output shape :", output_shape)
    # output_type = sess.get_outputs()[0].type
    # print("Output type  :", output_type)

    # ----------------------------get output
    t3 = time.time()
    result = []

    tbar = tqdm(all_images, ncols=100)
    for img_file in tbar:
        image = _val_augmentation(img_file)

        if image.max() > 1:
            image = image / 255
        # input[channel] = (input[channel] - mean[channel]) / std[channel]
        image = norm_myself(image, MEAN, STD)
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        prediction = sess.run([output_name], {input_name: image})   #  [[ 7.4938965  -2.3475928  -0.6207006  -1.6629311  -2.1027133   2.9506874,   1.4629353  -0.50333536 -2.9737248  -1.3985391  -3.4913282 ]]
        # softmax
        prediction = np.exp(prediction[0][0])/np.sum(np.exp(prediction[0][0]), axis=0)
        prediction = np.argmax(prediction)
        result.append(prediction.item())
    t4 = time.time()
    print("Inference time with the ONNX model: {}".format(t4 - t3))
    print("Inference result:", result)


if __name__ == "__main__":
    # exporter settings
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--configs', default='../../saved/OLED-Resnet18/04-28_09-02/configs.json', type=str,
                        help='Path to the configs file (default: configs.json)')
    parser.add_argument('--onnx', type=str, default='oled_resnet18.onnx',
                        help="set model checkpoint path")
    parser.add_argument('--model', type=str, default='../../saved/OLED-Resnet18/04-28_09-02/best_model.pth',
                        help="set model checkpoint path")
    parser.add_argument('--images', type=str, default="oled_one", help='input image to use')

    args = parser.parse_args()
    # print(args)

    # pytorch测试
    output_pytorch = test_pytorch(args)

    # onnx 测试
    test_onnx(args)

