from torch.utils.data.dataset import Dataset
import torchvision.transforms as transform
import torch
import numpy as np
import cv2


class CreateDatasets(Dataset):
    def __init__(self, ori_imglist, img_size):
        self.ori_imglist = ori_imglist
        self.transform = transform.Compose([
            transform.ToTensor(),
            transform.Resize((img_size, img_size)),
            transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.ori_imglist)

    def __getitem__(self, item):
        ori_img = cv2.imread(self.ori_imglist[item])
        ori_img = ori_img[:, :, ::-1]
        ori_img = self.transform(ori_img.copy())
        ori_img2 = cv2.imread(self.ori_imglist[item].replace('b.png', 'c.png').replace('Band13_15_az', 'Band10_11_16'))
        ori_img2 = ori_img2[:, :, ::-1]
        ori_img2 = self.transform(ori_img2.copy())
        ori_img3 = cv2.imread(self.ori_imglist[item].replace('b.png', 'd.png').replace('Band13_15_az', 'Band08_09_sza'))
        ori_img3 = ori_img3[:, :, ::-1]
        ori_img3 = self.transform(ori_img3.copy())
        ori_img4 = cv2.imread(self.ori_imglist[item].replace('b.png', 'e.png').replace('Band13_15_az', 'Band_sataz_satza_map'))
        ori_img4 = ori_img4[:, :, ::-1]
        ori_img4 = self.transform(ori_img4.copy())
        # to change to 2 png files, we need to read the png as 3 channel files
        ori_img = torch.cat([ori_img, ori_img2, ori_img3, ori_img4], dim=0)
        real_img = cv2.imread(self.ori_imglist[item].replace('b.png', 'a.png').replace('Band13_15_az', 'Band03'))
        real_img = real_img[:, :, ::-1]
        real_img = self.transform(real_img.copy())
        return ori_img, real_img
