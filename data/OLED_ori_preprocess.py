# _*_coding:utf-8_*_
# @auther:FelixFu
# @Date: 2021.4.14
# @github:https://github.com/felixfu520

import os
import random
import shutil


def split_trainval(root_path, output_path, shuffle=False, ratio=0.1):
    images_folders = [folder_name for folder_name in os.listdir(root_path) if os.path.isdir(os.path.join(root_path,folder_name))]

    for folder in images_folders:
        one_label_imgs = os.listdir(os.path.join(root_path, folder))
        n_total = len(one_label_imgs)
        offset = int(n_total * ratio)

        if shuffle:
            random.shuffle(one_label_imgs)

        sublist_val = one_label_imgs[:offset]
        sublist_train = one_label_imgs[offset:]

        count = 0
        for img in sublist_train:
            src_path = os.path.join(root_path, folder)
            dst_path = os.path.join(output_path, "train", folder)
            if not os.path.exists(dst_path):
                os.mkdir(dst_path)
            shutil.copy(os.path.join(src_path, img), os.path.join(dst_path, img))
            count += 1

        for img in sublist_val:
            src_path = os.path.join(root_path, folder)
            dst_path = os.path.join(output_path, "val", folder)
            if not os.path.exists(dst_path):
                os.mkdir(dst_path)
            shutil.copy(os.path.join(src_path, img), os.path.join(dst_path, img))
            count += 1
        assert count == len(one_label_imgs)


if __name__ == "__main__":
    root_path = "/home/felixfu/data/classification/OLED_ori"
    output_path = "/home/felixfu/data/classification/OLED"
    split_trainval(root_path, output_path, shuffle=True, ratio=0.1)