# Classification in PyTorch
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

<!-- TOC -->

- [Classification in PyTorch](#Classification in PyTorch)
  - [Requirements](#requirements)
  - [Main Features](#main-features)
    - [Models](#models)
    - [Datasets](#datasets)
    - [Losses](#losses)
    - [Learning rate schedulers](#learning-rate-schedulers)
    - [Data augmentation](#data-augmentation)
  - [Training](#training)
  - [Inference](#inference)
  - [Code structure](#code-structure)
  - [Config file format](#config-file-format)
  - [Acknowledgement](#acknowledgement)

<!-- /TOC -->

This repo contains a PyTorch an implementation of different classification models for different datasets.

## Requirements
PyTorch and Torchvision needs to be installed before running the scripts, together with `PIL` and `opencv` for data-preprocessing and `tqdm` for showing the training progress. PyTorch v1.1 is supported (using the new supported tensoboard); can work with ealier versions, but instead of using tensoboard, use tensoboardX.

```bash
pip install -r requirements.txt
```

or for a local installation

```bash
pip install --user -r requirements.txt
```

## Main Features

- A clear and easy to navigate structure,
- A `json` config file with a lot of possibilities for parameter tuning,
- Supports various models, losses, Lr schedulers, data augmentations and datasets,

**So, what's available ?**

### Models 
- (**Resnet18**) 

### Datasets

- **MedMnist:** 

### Losses
In addition to the Cross-Entorpy loss, there is also
- **暂无**, nothing

### Learning rate schedulers
- **StepLR**, nothing.

### Data augmentation
All of the data augmentations are implemented using OpenCV in `\base\base_dataset.py`, which are: rotation (between -10 and 10 degrees), random croping between 0.5 and 2 of the selected `crop_size`, random h-flip and blurring

## Training
To train a model, first download the dataset to be used to train the model, then choose the desired architecture, add the correct path to the dataset and set the desired hyperparameters (the config file is detailed below), then simply run:

```bash
python train.py --configs configs.json
```

The training will automatically be run on the GPUs (if more that one is detected and  multipple GPUs were selected in the config file, `torch.nn.DataParalled` is used for multi-gpu training), if not the CPU is used. The log files will be saved in `saved\runs` and the `.pth` chekpoints in `saved\`, to monitor the training using tensorboard, please run:

```bash
tensorboard --logdir saved
```


## Inference

For inference, we need a PyTorch trained model, the images we'd like to segment and the config used in training (to load the correct model and other parameters), 

```bash
python inference.py --configs configs.json --model best_model.pth --images images_folder
```

The predictions will be saved as `.png` images using the default palette in the passed fodler name, if not, `outputs\` is used, for Pacal VOC the default palette is:


Here are the parameters availble for inference:
```
--output       The folder where the results will be saved (default: outputs).
--images       Folder containing the images to segment.
--model        Path to the trained model.
--config       The config file used for training the model.
```

## Code structure
The code structure is based on [pytorch-template](https://github.com/victoresque/pytorch-template/blob/master/README.md)

  ```
  pytorch-template/
  │
  ├── train.py - main script to start training
  ├── inference.py - inference using a trained model
  ├── trainer.py - the main trained
  ├── config.json - holds configuration for training
  │
  ├── base/ - abstract base classes
  │   ├── base_data_loader.py
  │   ├── base_model.py
  │   ├── base_dataset.py - All the data augmentations are implemented here
  │   └── base_trainer.py
  │
  ├── dataloader/ - loading the data for different segmentation datasets
  │
  ├── models/ - contains semantic segmentation models
  │
  ├── saved/
  │   ├── runs/ - trained models are saved here
  │   └── log/ - default logdir for tensorboard and logging output
  │  
  └── utils/ - small utility functions
      ├── losses.py - losses used in training the model
      ├── metrics.py - evaluation metrics used
      └── lr_scheduler - learning rate schedulers 
  ```

## Config file format
Config files are in `.json` format:
```javascript
{
    "name": "Resnet18",
    "n_gpu": 0,
    "use_synch_bn": false,

    "arch": {
        "type": "Resnet18",
        "args": {
            "in_channels": 1,
            "pretrained": false,
            "freeze_bn": false
        }
    },

    "train_loader": {
        "type": "MedMnist",
        "args":{
            "data_dir": "data/MedMnist/val",

            "base_size": 256,
            "crop_size": 224,
            "augment": true,
            "scale": true,
            "flip": false,
            "rotate": false,
            "blur": false,
            "histogram": false,

            "num_workers": 8,
            "batch_size": 4,
            "shuffle": true,

            "in_channels": 1
        }
    },

    "val_loader": {
        "type": "MedMnist",
        "args":{
            "data_dir": "data/MedMnist/val",

            "crop_size": 224,
            "histogram": false,

            "batch_size": 4,
            "num_workers": 4,
            "in_channels": 1,

            "val": true
        }
    },

    "loss": "CrossEntropyLoss2d",
    "ignore_index": 255,

    "optimizer": {
        "type": "SGD",
        "differential_lr": false,
        "args":{
            "lr": 0.01,
            "weight_decay": 1e-4,
            "momentum": 0.9
        }
    },
    "lr_scheduler": {
        "type": "StepLR_",
        "args": {
            "step_size": 1,
            "gamma": 0.5
        }
    },

    "trainer": {
        "epochs": 1,
        "save_dir": "saved/",
        "save_period": 1,
  
        "monitor": "max top1",
        "early_stop": 10,
        
        "tensorboard": true,
        "log_dir": "saved/runs",
        "log_per_iter": 20,

        "val": true,
        "val_per_epochs": 1
    }
}

```



## Acknowledgement
- [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding)
- [Pytorch-Template](https://github.com/victoresque/pytorch-template/blob/master/README.m)
- [Synchronized-BatchNorm-PyTorch](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch)
