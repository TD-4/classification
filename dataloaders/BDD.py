# _*_coding:utf-8_*_
# @auther:FelixFu
# @Date: 2021.4.14
# @github:https://github.com/felixfu520

import numpy as np
import os
import cv2

from base import BaseDataSet, BaseDataLoader


class BDDDataset(BaseDataSet):
    def __init__(self, **kwargs):
        self.num_classes = 29
        super(BDDDataset, self).__init__(**kwargs)

    def _set_files(self):
        """获取所有文件的文件名和标签
        """
        if self.val:
            list_path = os.path.join(self.root, "testlist.txt")
        else:
            list_path = os.path.join(self.root, "trainlist.txt")

        images, labels = [], []
        with open(list_path, 'r', encoding='utf-8') as images_labels:
            for image_label in images_labels:
                images.append(image_label.split(",,,")[0])
                labels.append(image_label.split(",,,")[1])

        self.files = list(zip(images, labels))

    def _load_data(self, index):
        """通过文件名获得，图片和类别
        :param index:
        :return: ndarray
        """
        image_path, label = self.files[index]
        # image_path = image_path.encode('utf8', errors='surrogateescape').decode('utf-8')
        if self.in_channels == 1:
            # 修改支持中文路径
            img = cv2.imdecode(np.fromfile(image_path.encode('utf8'), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        elif self.in_channels == 3:
            img = cv2.imdecode(np.fromfile(image_path.encode('utf8'), dtype=np.uint8), cv2.IMREAD_COLOR)
        return img, label, image_path


class BDD(BaseDataLoader):
    def __init__(self, data_dir,
                 base_size=None, crop_size=None, augment=False, scale=True, flip=False, rotate=False, blur=False, histogram=False,
                 batch_size=1, num_workers=1, shuffle=False,
                 in_channels=3, val=False):
        if in_channels == 3:
            self.MEAN = [0.45734706, 0.43338275, 0.40058118]    # mean & std在不同数据集上要不同，每个数据集算一次，不能复用
            self.STD = [0.23965294, 0.23532275, 0.2398498]
        else:
            self.MEAN = [0.6790830900388274]  # [0.3858034032292721]
            self.STD = [0.25241563832076486]  # [0.12712721340420535]
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

        self.dataset = BDDDataset(**kwargs)
        super(BDD, self).__init__(self.dataset, batch_size, shuffle, num_workers)


