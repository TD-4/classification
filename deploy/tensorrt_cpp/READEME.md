# TensorRT(C++)

## 环境
`docker pull fusimeng/cds:cuda10.2-torch1.8.1-opencv_gpu4.5.2-tensorrt7.2.2.3-onnx1.9.0-onnxruntime1.7.0`

[安装过程见](../../Environment-docker.md)

```
docker run -itd -v /home/felixfu/cds/:/root/cds -v /home/felixfu/Downloads/:/root/Downloads -v /home/felixfu/data:/root/data -p 5901:22 -p 6006:6006 --name ff fusimeng/cds:cuda10.2-torch1.8.1-opencv_gpu4.5.2-tensorrt7.2.2.3-onnx1.9.0-onnxruntime1.7.0
docker exec -it ff bash
service ssh start
```
## 代码使用
