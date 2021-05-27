import argparse
import torch

from PIL import Image
from torchvision.transforms import ToTensor
import json
import numpy as np


def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


if __name__ == "__main__":

    # exporter settings
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--configs', default='/home/felixfu/cds/classification/saved/MDD-Resnet34/05-27_09-14/configs.json', type=str,
                        help='Path to the configs file (default: configs.json)')
    parser.add_argument('--model', type=str, default='/home/felixfu/cds/classification/saved/MDD-Resnet34/05-27_09-14/best_model.pth', help="set model checkpoint path")
    parser.add_argument('--model_out', type=str, default='mdd_resnet34.onnx')
    parser.add_argument('--image', type=str, default="", help='input image to use')

    args = parser.parse_args()
    print(args)

    # ----------------------------set the device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('running on device ' + str(device))

    # ----------------------------load the image
    if args.image:
        # img = Image.open(args.image)
        # img_to_tensor = ToTensor()
        # input = img_to_tensor(img).view(1, -1, img.size[1], img.size[0]).to(device)
        pass
    else:
        pixels = 224
        # img = np.random.rand(pixels, pixels, 1)
        # input = torch.from_numpy(img).view(1, 1, pixels, pixels).float().to(device)
        input = torch.zeros([1, 1, 224, 224], dtype=torch.float32).to(device)
    print("input size is..", input.shape)

    # ----------------------------load the model
    import models
    num_classes = 21
    config = json.load(open(args.configs))
    model = get_instance(models, 'arch', config, num_classes).to(device)  # 定义模型

    checkpoint = torch.load(args.model)
    base_weights = checkpoint["state_dict"]   # module.model.conv1.weight格式，而model的权重是model.conv1.weight格式
    print('Loading base network...')

    from collections import OrderedDict  # 导入此模块
    new_state_dict = OrderedDict()
    for k, v in base_weights.items():
        name = k[7:]  # remove `module.`，即只取module.model.conv1.weights的后面几位
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)  # 加载权重

    print("loaded weight")

    # ----------------------------export the model
    input_names = ["input_0"]
    output_names = ["output_0"]

    print('exporting model to ONNX...')
    torch.onnx.export(model, input, args.model_out, verbose=True, input_names=input_names, output_names=output_names, opset_version=9)
    print('model exported to {:s}'.format(args.model_out))
