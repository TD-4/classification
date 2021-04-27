# _*_coding:utf-8_*_
# @auther:FelixFu
# @Date: 2021.4.14
# @github:https://github.com/felixfu520

# Originally written by Kazuto Nakashima
# https://github.com/kazuto1011/deeplab-pytorch

import numpy as np
import os
import cv2
from itertools import chain
from glob import glob

from base import BaseDataSet, BaseDataLoader


class OLEDDataset(BaseDataSet):
    def __init__(self, **kwargs):
        self.num_classes = 11
        super(OLEDDataset, self).__init__(**kwargs)

    def _set_files(self):
        """
        功能：获取所有文件的文件名和标签
        """
        all_data_path, labels = [], []
        image_folders = list(map(lambda x: self.root + "/" + x, os.listdir(self.root)))
        all_images = list(map(lambda x: glob(x + "/*"), image_folders))
        all_images = list(chain.from_iterable(all_images))

        # imageNameSet = set()
        # imageNameList = list()
        for file in all_images:
            label = os.path.dirname(file).split("/")[-1]
            all_data_path.append(file)
            labels.append(label)
            # if label not in imageNameSet:
            #     imageNameSet.add(label)
            #     imageNameList.append(label)
            # labels.append(imageNameList.index(label))
        #
        # label_txt_path = os.path.split(os.path.realpath(__file__))[0]
        # label_txt_path = os.path.join(label_txt_path, "labels", "OLED_labels.txt")
        # with open(label_txt_path, "w") as f:
        #     for i, name in enumerate(imageNameList):
        #         f.write(str(i) + ":" + name + "\n")
        self.files = list(zip(all_data_path, labels))

    def _load_data(self, index):
        """
        功能：通过文件名获得，图片和类别
        :param index:
        :return:
        """
        image_path, label = self.files[index]
        if self.in_channels == 1:
            # 修改支持中文路径
            img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        elif self.in_channels == 3:
            img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        return img, label


class OLED(BaseDataLoader):
    def __init__(self, data_dir,
                 base_size=None, crop_size=None, augment=False, scale=True, flip=False, rotate=False, blur=False, histogram=False,
                 batch_size=1, num_workers=1, shuffle=False,
                 in_channels=3, val=False):
        if in_channels == 3:
            self.MEAN = [0.45734706, 0.43338275, 0.40058118]
            self.STD = [0.23965294, 0.23532275, 0.2398498]
        else:
            self.MEAN = [0.45734706]
            self.STD = [0.23965294]
        kwargs = {
            'root': data_dir,

            'mean': self.MEAN, 'std': self.STD,

            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'histogram': histogram,

            'in_channels': in_channels,

            'val': val
        }

        self.dataset = OLEDDataset(**kwargs)
        super(OLED, self).__init__(self.dataset, batch_size, shuffle, num_workers)


