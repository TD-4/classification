import numpy as np
import os
import cv2
import re
from itertools import chain
from glob import glob


def get_labels(root_path="classification/BDD"):
    """
    args:
        root_path:数据集的路径。此文件夹下有多个文件夹，每个文件夹里面有对应图片
            eg.
            BDtest     BKLMtest   DBHBtest   flagtest    HYQPGtest   LLMtrain  MEWtrain       WLtrain
            BDtrain    BKLMtrain  DBHBtrain  flagtrain   HYQPGtrain  LLtest    Othertest      XBKLTtest
            BHHtest    BLtest     DBLBtest   HHBTtest    HYXtest     LLtrain   Othertrain     XBKLTtrain
            BHHtrain   BLtrain    DBLBtrain  HHBTtrain   HYXtrain    LMtest    Paotest        ZWtest
            ...
    return:
        在root_path目录下生成labels.txt文件
    """
    folders = os.listdir(root_path)
    folders = [f[:-5] for f in folders if re.search("train", f)]
    with open(os.path.join(root_path, "labels.txt"), 'w') as f:
        for i, name in enumerate(folders):
            f.write(str(name) + " "+ str(i) + "\n")


def gen_txt(root_path="classification/BDD"):
    """
    args:
    root_path:数据集的路径。此文件夹下有多个文件夹，每个文件夹里面有对应图片
        eg.
        BDtest     BKLMtest   DBHBtest   flagtest    HYQPGtest   LLMtrain  MEWtrain       WLtrain
        BDtrain    BKLMtrain  DBHBtrain  flagtrain   HYQPGtrain  LLtest    Othertest      XBKLTtest
        BHHtest    BLtest     DBLBtest   HHBTtest    HYXtest     LLtrain   Othertrain     XBKLTtrain
        BHHtrain   BLtrain    DBLBtrain  HHBTtrain   HYXtrain    LMtest    Paotest        ZWtest
        labels.txt
        ...
    return:
        在root_path目录下生成trainlist.txt和testlist.txt文件
    """
    # 获取所有图片
    image_folders = list(map(lambda x: root_path + "/" + x, os.listdir(root_path)))
    all_images = list(map(lambda x: glob(x + "/*"), image_folders))
    all_images = list(chain.from_iterable(all_images))
    all_images = [f for f in all_images if re.search("bmp", f)]

    # 获取标签
    labels = {}
    with open(os.path.join(root_path, "labels.txt")) as file:
        for line in file:
            labels[line.split()[0]] = line.split()[1]   # ｛name:id｝


    train_images, test_images, train_labels, test_labels = [], [], [], []
    for file in all_images:
        label_name = os.path.dirname(file).split("/")[-1]
        if re.search("train", label_name):
            label_name = label_name[:-5]
            # b = bytes(file, encoding='gbk')
            # file2 = b.decode('utf-8')
            train_images.append(file)
            train_labels.append(labels[label_name])
        else:
            label_name = label_name[:-4]
            # b = bytes(file, encoding='gbk')
            # file2 = b.decode('utf-8')
            test_images.append(file)
            test_labels.append(labels[label_name])


    trainlist = os.path.join(root_path, "trainlist.txt")
    with open(trainlist, "w", encoding='utf-8') as f:
        for img_path, label in zip(train_images, train_labels):
            f.write(str(img_path) + ",,," + label + "\n")

    testlist = os.path.join(root_path,"testlist.txt")
    with open(testlist, "w", encoding='utf-8') as f:
        for img_path, label in zip(test_images, test_labels):
            f.write(str(img_path) + ",,," + label + "\n")


def gen_mean_std(root_path="classification/BDD"):
    """
    获得mean & std
    """
    gray_channel = 0
    count = 0
    with open(os.path.join(root_path, "trainlist.txt")) as file:
        for line in file:
            img = cv2.imread(line.split(",,,")[0], cv2.IMREAD_GRAYSCALE) / 255.0
            gray_channel += np.sum(img)
            count += 1
    gray_channel_mean = gray_channel / (count * 150 * 150)

    gray_channel = 0
    count = 0
    with open(os.path.join(root_path, "trainlist.txt")) as file:
        for line in file:
            img = cv2.imread(line.split(",,,")[0], cv2.IMREAD_GRAYSCALE) / 255.0
            gray_channel = gray_channel + np.sum((img - gray_channel_mean)**2)
            count += 1
    gray_channel_std = np.sqrt(gray_channel / (count * 150 * 150))

    print("mean:", gray_channel_mean)
    print("std:", gray_channel_std)


if __name__ == "__main__":
    # get_labels()  # 获取labels.txt
    # gen_txt()   # 获取trainlist.txt和testlist.txt
    gen_mean_std()
