from PIL import Image
from datasets.transforms import joint_transforms
import os
import random
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms.functional as TF
from datasets.transforms import *
import torchvision.transforms as standard_transforms
from datasets.transforms.transforms import MaskToTensor
from datasets.transforms.joint_transforms import Resize, RandomCrop, RandomHorizontallyFlip
from torch.utils.data import Dataset
from deeplab import Deeplab
import network

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

    def __init__(self, opt):
        self.opt = opt
        
        if 'no_label' in opt.label_root_D:
            self.no_label_mode = True
        else: 
            self.no_label_mode = False
        if self.no_label_mode:
            self.D_image_paths= self.get_paths(opt)
        else:
            self.D_image_paths, self.D_label_paths = self.get_paths(opt)

        self.dataset_size = self.__len__()
        # self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17:5, 19:6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
        #                       26: 13, 27:14, 28:15, 31:16, 32: 17, 33: 18}
        self.id_to_trainid = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5:5, 6:6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12,
                              13: 13, 14:14, 15:15, 16:16, 17: 17, 18: 18}
        # Transform definition
        self.target_transform = MaskToTensor()
        # self.img_transform = standard_transforms.Compose([standard_transforms.Resize(512),
        #     standard_transforms.ToTensor(),
        #                                                   standard_transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        self.img_transform = standard_transforms.Compose([standard_transforms.ToTensor(),
                                                          standard_transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        # self.day_joint_transform = joint_transforms.Compose([Resize(size=(256, 512))])
        self.day_joint_transform = joint_transforms.Compose([Resize(size=(2048, 1024))])
        # self.night_joint_transform = joint_transforms.Compose([Resize(size=(640, 360)),
        #                                                      RandomCrop(size=(512, 256))])
        # self.model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
        # init_weights = './cyclegan_sem_model.pth'
        # self.model = Deeplab(num_classes=19)
        # if init_weights is not None:
        #     saved_state_dict = torch.load(init_weights, map_location=lambda storage, loc: storage)
        #     self.model.load_state_dict(saved_state_dict)
        # # model.eval()
        # # self.model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
        # self.model.eval()
        # self.model.cuda()
        # self.model=network.modeling.__dict__['deeplabv3plus_resnet101'](num_classes=19, output_stride=16)
        
        
        
        # self.model = network.modeling.__dict__['deeplabv3plus_resnet101'](num_classes=19, output_stride=16)
        # self.model.load_state_dict( torch.load( './models/best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar' )['model_state']  )
        # self.model = self.model.cuda()
        # self.model.eval()


    def get_paths(self, opt):
        if self.no_label_mode:
            opt.image_root_D = opt.image_root_D.replace('phase', opt.phase)
        
            D_img_list_unrefined = os.listdir(opt.image_root_D)
            D_img_list_refined = [x for x in D_img_list_unrefined if not 'json' in x]
            D_img_list_refined = [x for x in D_img_list_refined if not 'blend' in x]
            D_img_list_refined = [x for x in D_img_list_refined if not 'mask' in x]


            D_image_paths = []

            for file in sorted(D_img_list_refined):
                D_image_paths.append(os.path.join(opt.image_root_D, file))
            
        else:
            opt.image_root_D = opt.image_root_D.replace('phase', opt.phase)
        
            D_img_list_unrefined = os.listdir(opt.image_root_D)
            D_img_list_refined = [x for x in D_img_list_unrefined if not 'json' in x]
            D_img_list_refined = [x for x in D_img_list_refined if not 'blend' in x]
            D_img_list_refined = [x for x in D_img_list_refined if not 'mask' in x]


            D_image_paths, D_label_paths = [], []

            for file in sorted(D_img_list_refined):
                D_image_paths.append(os.path.join(opt.image_root_D, file))

            D_label_paths = [x.replace('.png','_mask.png').replace('/mnt/hdd2/lyj/Data/KIAPI/','/home/lyj/Projects/Semantic-Segment-Anything/output_kiapi/') for x in D_image_paths] # SSA mode
            # opt.image_root_D = opt.image_root_D.replace('phase', opt.phase)
            # opt.image_list_D = opt.image_list_D.replace('phase', opt.phase)
            
            # D_image_paths, D_label_paths = [], []

            # D_image_list = open(opt.image_list_D, 'r')
            # for file in sorted(D_image_list):
            #     D_image_paths.append(file[:-1])
            

            # D_label_list = open(opt.label_list_D, 'r')
            # for file in sorted(D_label_list):
            #     D_label_paths.append(file[:-1])
        
        if self.no_label_mode:
            return D_image_paths
        else:
            return D_image_paths, D_label_paths


    def __getitem__(self, index):
        if self.no_label_mode:
            # day image
            D_image_path = self.D_image_paths[index]
            D_image = Image.open(D_image_path).convert('RGB') # shape = (2048, 1024)

            # day label
            # D_label_path = self.D_label_paths[index]
            # D_label = self.create_label(Image.open(D_label_path))


            # Data Processing
            # D_image, D_label = self.day_joint_transform(D_image, D_label)

            D_image= self.img_transform(D_image)
            D_image_for_compute = D_image.unsqueeze(0)
            D_image_for_compute = D_image_for_compute.cuda()
            with torch.no_grad():
                # D_onehot_pred = self.model(D_image_for_compute)['out'][0]
                D_onehot_pred = self.model(D_image_for_compute)[0].cpu()
            D_label_pred = D_onehot_pred.argmax(0)
            D_label = self.create_label(D_label_pred)
            # D_label = self.target_transform(D_label_pred)
            D_label = self.target_transform(D_label)
            D_onehot_label = self.one_hot_label(D_label)

        else:
            # day image
            D_image_path = self.D_image_paths[index]
            D_image = Image.open(D_image_path).convert('RGB') # shape = (2048, 1024)

            # day label
            D_label_path = self.D_label_paths[index]
            D_label = self.create_label(Image.open(D_label_path))

            # index_N = random.randint(0, len(self.N_image_paths) - 1)
            # # night image
            # N_image_path = self.N_image_paths[index_N]
            # N_image = Image.open(N_image_path).convert('RGB') # shape = (1920, 1080)

            # # night label
            # N_label_path = self.N_label_paths[index_N]
            # N_label = self.create_label(Image.open(N_label_path))

            # Data Processing
            # D_image, D_label = self.day_joint_transform(D_image, D_label) # no transform for the inference images
            
            
            # N_image, N_label = self.night_joint_transform(N_image, N_label)
            
            # D_image, N_image = self.img_transform(D_image), self.img_transform(N_image)
            # D_label, N_label = self.target_transform(D_label), self.target_transform(N_label)
            # D_onehot_label, N_onehot_label = self.one_hot_label(D_label), self.one_hot_label(N_label)
            D_image= self.img_transform(D_image)
            D_label = self.target_transform(D_label)
            D_onehot_label = self.one_hot_label(D_label)

            # print(D_image_path)
            # print(N_image_path)

        data = {
            'D_image': D_image,
            'D_label': D_label,
            'D_onehot_label': D_onehot_label,
            'D_image_path': D_image_path,
            # 'N_image': N_image,
            # 'N_label': N_label,
            # 'N_onehot_label': N_onehot_label,
            # 'N_image_path': N_image_path
        }
        
        return data


    def __len__(self):
        return len(self.D_image_paths)#, len(self.N_image_paths))


    def create_label(self, label):
        if self.no_label_mode:
            label = np.asarray(label, np.uint8)
            label_copy = 255 * np.ones(label.shape, dtype=np.uint8)
            for i in range(19):
                label_copy[label == i] = i
            label_copy[label_copy == 255] = 19
            label = Image.fromarray(label_copy.astype(np.uint8))
        else:
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