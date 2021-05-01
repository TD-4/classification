import os
import time
import cv2
import json
from tqdm import tqdm
import argparse
from PIL import Image
import numpy as np

import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from functools import reduce

import models
from torchvision import transforms


# ------------------ ------------------ Pytorch ------------------ ------------------
def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


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
    result_argmax = []
    with torch.no_grad():
        tbar = tqdm(all_images, ncols=100)
        for img_file in tbar:
            image = _val_augmentation(img_file)
            # input[channel] = (input[channel] - mean[channel]) / std[channel]
            input = normalize(to_tensor(image)).unsqueeze(0)

            prediction = model(input.to(device))  # tensor([[ 7.4939, -2.3476, -0.6207, -1.6629, -2.1027,  2.9507,  1.4629, -0.5033, -2.9737, -1.3985, -3.4913]], device='cuda:0')
            prediction = prediction.squeeze(0).cpu().numpy()
            result.append(prediction)
            prediction = F.softmax(torch.from_numpy(prediction), dim=0).argmax(0).cpu().numpy()
            result_argmax.append(prediction)
    t4 = time.time()
    print("Inference time with the PyTorch model: {}".format(t4 - t3))
    print("Inference result:", result)
    return result, result_argmax


# ------------------ ------------------ TensorRT ------------------ ------------------
def norm_myself(tensor, mean, std):
    return (tensor -mean) /std


# -------------------------------------Prepare some useful functions
TRT_LOGGER = trt.Logger()  # This logger is required to build an engine


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        """Within this context, host_mom means the cpu memory and device means the GPU memory
        """
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


# -------------------------------------ONNX to Engine
def get_engine(max_batch_size=1, onnx_file_path="", engine_file_path="", fp16_mode=False, int8_mode=False, save_engine=False):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine(max_batch_size, save_engine):
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        # create_network() without parameters will make parser.parse() return False
        with trt.Builder(TRT_LOGGER) as builder, \
                builder.create_network(flags=1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
                trt.OnnxParser(network, TRT_LOGGER) as parser:

            builder.max_workspace_size = 1 << 30  # Your workspace size
            builder.max_batch_size = max_batch_size
            # pdb.set_trace()
            builder.fp16_mode = fp16_mode  # Default: False
            builder.int8_mode = int8_mode  # Default: False
            if int8_mode:
                # To be updated
                raise NotImplementedError

            # Parse model file
            if not os.path.exists(onnx_file_path):
                quit('ONNX file {} not found'.format(onnx_file_path))
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                parser.parse(model.read())
            print('Completed parsing of ONNX file')

            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            # pdb.set_trace()
            # network.mark_output(network.get_layer(network.num_layers-1).get_output(0)) # Riz
            # network.mark_output(network.get_layer(network.num_layers-1).get_output(1)) # Riz

            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")

            if save_engine:
                with open(engine_file_path, "wb") as f:
                    f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, load it instead of building a new one.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine(max_batch_size, save_engine)


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer data from CPU to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def postprocess_the_outputs(h_outputs, shape_of_output):
    # slice for the real size of output
    output_size = reduce(lambda x, y: x * y, shape_of_output)
    h_outputs = h_outputs[:output_size]
    h_outputs = h_outputs.reshape(*shape_of_output)
    return h_outputs


def test_tensorrt(args):
    # ----------------------------load the image
    # Dataset used for training the model
    MEAN = [0.45734706]
    STD = [0.23965294]

    max_batch_size = 1   # The batch size of input mush be smaller the max_batch_size once the engine is built

    image_folders = args.images
    all_images = [os.path.join(image_folders, name) for name in os.listdir(image_folders)]
    # if args.images:
    #     # img = Image.open(args.image)
    #     # img_to_tensor = ToTensor()
    #     # input_pth = img_to_tensor(img).view(1, -1, img.size[1], img.size[0]).to(device)
    #     pass
    # else:
    #     x_input = np.random.rand(max_batch_size, 1, 224, 224).astype(dtype=np.float32)

    # ----------------------------load the model
    # These two modes are dependent on hardwares
    fp16_mode = False
    int8_mode = False

    # Build an engine
    engine = get_engine(max_batch_size, args.onnx, args.engine, fp16_mode, int8_mode)
    # Create the context for this engine
    context = engine.create_execution_context()
    # Allocate buffers for input and output
    inputs, outputs, bindings, stream = allocate_buffers(engine)  # input, output: host # bindings

    # ----------------------------get output
    t3 = time.time()
    result = []
    result_argmax = []
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
        x_input = image
        # x_input = np.random.rand(max_batch_size, 1, 224, 224).astype(dtype=np.float32)

        # Do inference
        shape_of_output = (max_batch_size, 11)  # 11 is num_classes
        # Load data to the buffer
        inputs[0].host = x_input.reshape(-1)
        # inputs[1].host = ... for multiple input
        t1 = time.time()
        trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)  # numpy data [ 7.493896   -2.347594   -0.6206987  -1.6629304  -2.1027129   2.9506874,  1.4629362  -0.50333786 -2.9737246  -1.3985405  -3.4913301 ]
        t2 = time.time()
        output_from_trt_engine = postprocess_the_outputs(trt_outputs[0], shape_of_output)

        # softmax
        result.append(output_from_trt_engine[0])
        prediction = np.exp(output_from_trt_engine[0])/np.sum(np.exp(output_from_trt_engine[0]), axis=0)
        prediction = np.argmax(prediction)
        result_argmax.append(prediction.item())
    t4 = time.time()
    print("Inference time with the TensorRT model: {}".format(t4 - t3))
    print("Inference result:", result)
    return result,result_argmax


if __name__ == "__main__":
    # exporter settings
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--configs', default='../../saved/OLED-Resnet18/04-28_09-02/configs.json', type=str,
                        help='Path to the configs file (default: configs.json)')
    parser.add_argument('--onnx', type=str, default='oled_resnet18.onnx',
                        help="set model checkpoint path")
    parser.add_argument('--model', type=str, default='../../saved/OLED-Resnet18/04-28_09-02/best_model.pth',
                        help="set model checkpoint path")
    parser.add_argument('--engine', type=str, default='oled_resnet.engine')
    parser.add_argument('--images', type=str, default="oled_one", help='input image to use')

    args = parser.parse_args()
    print(args)

    # -------------- tensorrt
    output_from_trt_engine, result = test_tensorrt(args)

    # -------------- pytorch
    output_from_pytorch_model, result = test_pytorch(args)

    # 对比
    mse = np.mean((output_from_trt_engine[0] - output_from_pytorch_model[0])**2)
    # print("Inference time with the TensorRT engine: {}".format(t2-t1))
    # print("Inference time with the PyTorch model: {}".format(t4-t3))
    print('MSE Error = {}'.format(mse))