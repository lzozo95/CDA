from PIL import Image
from datasets.transforms import joint_transforms
#import os
#import random
import torch
# import torch.nn as nn
import numpy as np
# import torchvision.transforms.functional as TF
from datasets.transforms import *
import torchvision.transforms as standard_transforms
from datasets.transforms.transforms import MaskToTensor
from datasets.transforms.joint_transforms import Resize, RandomCrop, RandomHorizontallyFlip
from torch.utils.data import Dataset

class Day2NightDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.D_image_paths, self.D_label_paths, self.N_image_paths, self.N_label_paths = self.get_paths(opt)
        self.dataset_size = self.__len__()
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17:5, 19:6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27:14, 28:15, 31:16, 32: 17, 33: 18}
        # Transform definition
        self.target_transform = MaskToTensor()
        self.img_transform = standard_transforms.Compose([standard_transforms.ToTensor(),
                                                          standard_transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        self.day_joint_transform = joint_transforms.Compose([Resize(size=(2048, 1024))])
        self.night_joint_transform = joint_transforms.Compose([Resize(size=(640, 360)),
                                                             RandomCrop(size=(512, 256))])


    def get_paths(self, opt):
        opt.image_root_D = opt.image_root_D.replace('phase', opt.phase)
        opt.image_list_D = opt.image_list_D.replace('phase', opt.phase)
        opt.label_root_D = opt.label_root_D.replace('phase', opt.phase)
        opt.label_list_D = opt.label_list_D.replace('phase', opt.phase)

        D_image_paths, D_label_paths = [], []

        D_image_list = open(opt.image_list_D, 'r')
        for file in sorted(D_image_list):
            D_image_paths.append(file[:-1])
        
        D_label_list = open(opt.label_list_D, 'r')
        for file in sorted(D_label_list):
            D_label_paths.append(file[:-1])
        
        return D_image_paths, D_label_paths


    def __getitem__(self, index):
        # day image
        D_image_path = self.D_image_paths[index]
        D_image = Image.open(D_image_path).convert('RGB') # shape = (2048, 1024)

        # day label
        D_label_path = self.D_label_paths[index]
        D_label = self.create_label(Image.open(D_label_path))

        # Data Processing
        D_image, D_label = self.day_joint_transform(D_image, D_label)

        D_image= self.img_transform(D_image)
        D_label = self.target_transform(D_label)
        D_onehot_label = self.one_hot_label(D_label)

        data = {
            'D_image': D_image,
            'D_label': D_label,
            'D_onehot_label': D_onehot_label,
            'D_image_path': D_image_path
        }
        
        return data


    def __len__(self):
        return len(self.D_image_paths)


    def create_label(self, label):
        label = np.asarray(label, np.uint8)
        label_copy = 255 * np.ones(label.shape, dtype=np.uint8)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        label_copy[label_copy == 255] = 19
        label = Image.fromarray(label_copy.astype(np.uint8))
        return label


    def one_hot_label(self, GT):
        label_map = GT.unsqueeze(0)
        _, h, w = label_map.size()
        nc = 20
        input_label = torch.zeros((nc, h, w))
        input_semantics = input_label.scatter_(0, label_map, 1.0)

        return input_semantics