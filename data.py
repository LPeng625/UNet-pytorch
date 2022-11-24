import os

import numpy as np
import torch
from torch.utils.data import Dataset
from utils import *
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor()
])


class MyDataset(Dataset):
    def __init__(self, path):
        self.path = path
        # 源码中直接读取路径，需要对路径中的图片进行筛选，比较麻烦
        # self.name = os.listdir(os.path.join(path, 'SegmentationClass'))
        # 通过读取train.txt文件来获取可用于分割的训练图片
        with open(os.path.join(self.path, "ImageSets/Segmentation/train.txt"), "r") as f:
            self.name = f.readlines()


    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        segment_name = self.name[index].strip('\n') # xx.png
        segment_path = os.path.join(self.path, 'SegmentationClass', segment_name + '.png')
        image_path = os.path.join(self.path, 'JPEGImages', segment_name + '.jpg')
        segment_image = keep_image_size_open_rgb(segment_path)
        image = keep_image_size_open_rgb(image_path)
        return transform(image), transform(segment_image)



if __name__ == '__main__':
    from torch.nn.functional import one_hot
    data = MyDataset('VOC2012/')
    print(data[0][0].shape)
    print(data[0][1].shape)
    out=one_hot(data[0][1].long())
    print(out.shape)
