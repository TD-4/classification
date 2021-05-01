# ONNX(Python)

## 环境
`docker pull fusimeng/tensorrt:10.2-7.0`
1. CUDA 10.2 cuDNN 7.6
2. Tensorrt 7.0
3. Opencv(GPU) 4.2
4. onnx 1.9.0
5. onnxruntime 1.7.0

## 代码使用
### PTH to ONNX
`python pth2onnx.py -c 配置文件的路径 --model pth文件的路径 --model_out onnx文件的输出路径 --image 输入图片的路径 `

### ONNX 测试
`python onnx_runtime.py -c 配置文件路径 --onnx onnx模型路径 --model pth文件路径 --images 测试图片所在的文件夹`


### 对比结果
```
ssh://root@gpu.aiserver.cn:46068/usr/bin/python3 -u /root/cds/classification/deploy/onnx_python/onnx_runtime.py
running on device cuda:0
Loading base network...
loaded weight
100%|████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 176.52it/s]
Inference time with the PyTorch model: 0.007378816604614258
Inference result: [0]
Loading onnx...
100%|█████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 99.31it/s]
Inference time with the ONNX model: 0.010536909103393555
Inference result: [0]
```

### GPU 版本
https://github.com/microsoft/onnxruntime/tree/v1.7.0
