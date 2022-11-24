import os

import tqdm
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from data import *
from net import *
from torchvision.utils import save_image
import time

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# mac m1芯片 使用的gpu 是 mps 后端处理，在拼接图的时候会有问题，故依然采用cpu
# device = torch.device('mps' if torch.has_mps else 'cpu')
device = torch.device('cpu')
weight_path = 'params/unet.pth'
data_path = 'VOC2012/'
save_path = 'train_image'
if __name__ == '__main__':
    num_classes = 3
    data_loader = DataLoader(MyDataset(data_path), batch_size=4, shuffle=True)
    net = UNet(num_classes).to(device)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print('successful load weight！')
    else:
        print('not successful load weight')

    opt = optim.Adam(net.parameters())
    loss_fun = nn.CrossEntropyLoss()

    start = time.time()
    epoch = 1
    while epoch < 2:
        for i, (image, segment_image) in enumerate(data_loader):
            image, segment_image = image.to(device), segment_image.to(device)
            out_image = net(image)
            train_loss = loss_fun(out_image, segment_image)
            opt.zero_grad()  # 梯度清零，如果不清零，梯度会累加
            train_loss.backward()  # 反向计算
            opt.step()  # 更新梯度

            if i % 5 == 0:  # 5张图片输出一次损失
                print(f'{epoch}-{i}-train_loss===>>{train_loss.item()}')

            if i % 50 == 0:
                torch.save(net.state_dict(), weight_path)
                print('save successfully!')

            # 将 原图-标签图-输出图 拼接并保存
            _image = image[0]
            _segment_image = segment_image[0]
            _out_image = out_image[0]

            img = torch.stack([_image, _segment_image, _out_image], dim=0)
            save_image(img, f'{save_path}/{i}.png')
        epoch += 1
    end = time.time()
    print('time_cost:', end - start)
