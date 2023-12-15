"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
from utils import get_config, get_data_loader_folder, pytorch03_to_pytorch04, load_inception
from trainer_mod import MUNIT_Trainer, UNIT_Trainer
from torch import nn
from scipy.stats import entropy
import torch.nn.functional as F
import argparse
from torch.autograd import Variable
from data import ImageFolder
import numpy as np
import torchvision.utils as vutils
import time
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import sys
import torch
import os
from torch.utils.data import DataLoader, dataset
from datasets.dataset import Day2NightDataset
import pdb
# docker run -it --gpus '"device=0,1,2,3"' \
# nvcr.io/nvidia/tensorflow:20.12-tf1-py3

# CUDA_VISIBLE_DEVICES=1 python3 test_batch.py --config ./outputs/cityscapes2acdcsnow/config.yaml --name cityscapes2snow --weather snow --output_folder ./results/cityscapes2acdc_snow --checkpoint ./outputs/cityscapes2acdcsnow/checkpoints/gen_00050000.pt --output_only
# CUDA_VISIBLE_DEVICES=0 python3 test_batch_smtl.py --config ./outputs/citysaceps2acdcsnow_loss/config.yaml --name cityscapes2snow_loss --weather snow --output_folder ./results/cityscapes2acdcsnow_loss --checkpoint ./outputs/citysaceps2acdcsnow_loss/checkpoints/gen_00050000.pt --output_only
# CUDA_VISIBLE_DEVICES=1 python3 test_batch_smtl.py --config ./outputs/citysaceps2acdcsnow_loss/config.yaml --name cityscapes2snow_loss --weather snow --output_folder ./results/cityscapes2acdcsnow_loss --checkpoint ./outputs/citysaceps2acdcsnow_loss/checkpoints/gen_00050000.pt --output_only
# CUDA_VISIBLE_DEVICES=1 python3 test_batch.py --config ./outputs/cityscapes2acdcfog/config.yaml --name cityscapes2fog --weather fog --output_folder ./results/cityscapes2acdc_fog --checkpoint ./outputs/cityscapes2acdcfog/checkpoints/gen_00050000.pt --output_only
# CUDA_VISIBLE_DEVICES=0 python3 test_batch_smtl.py --config ./outputs/citysaceps2acdcfog_loss/config.yaml --name cityscapes2fog_loss --weather fog --output_folder ./results/cityscapes2acdcfog_loss --checkpoint ./outputs/citysaceps2acdcfog_loss/checkpoints/gen_00050000.pt --output_only
# CUDA_VISIBLE_DEVICES=1 python3 test_batch.py --config ./outputs/cityscapes2acdcnight/config.yaml --name cityscapes2night --weather night --output_folder ./results/cityscapes2acdc_night --checkpoint ./outputs/cityscapes2acdcnight/checkpoints/gen_00050000.pt --output_only
# CUDA_VISIBLE_DEVICES=0 python3 test_batch_smtl.py --config ./outputs/citysaceps2acdcnight_loss/config.yaml --name cityscapes2night_loss --weather night --output_folder ./results/cityscapes2acdcnight_loss --checkpoint ./outputs/citysaceps2acdcnight_loss/checkpoints/gen_00050000.pt --output_only

# python3 test_batch_smtl.py --config ./outputs/citysaceps2acdcsnow_loss/config.yaml --name kiapi2snow_loss --weather snow --output_folder ./results/kiapi2acdcsnow_loss --checkpoint ./outputs/citysaceps2acdcsnow_loss/checkpoints/gen_00050000.pt --output_only

# parser.add_argument('--image_root_D', type=str, default='/mnt/hdd2/lyj/Data/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/phase')
# parser.add_argument('--image_list_D', type=str, default='/mnt/hdd2/lyj/Data/cityscapes/cityscapes_phase_image.txt')
# parser.add_argument('--label_root_D', type=str, default='/mnt/hdd2/lyj/Data/cityscapes/gtFine_trainvaltest/gtFine/phase')
# parser.add_argument('--label_list_D', type=str, default='/mnt/hdd2/lyj/Data/cityscapes/cityscapes_phase_label.txt')
# parser.add_argument('--image_root_N', type=str, default='/mnt/hdd2/lyj/Data/acdc/rgb_anon/weather/train')
# parser.add_argument('--image_list_N', type=str, default='/mnt/hdd2/lyj/Data/acdc/acdc_weather_phase_image.txt')
# parser.add_argument('--label_root_N', type=str, default='/mnt/hdd2/lyj/Data/acdc/gt/weather/train')
# parser.add_argument('--label_list_N', type=str, default='/mnt/hdd2/lyj/Data/acdc/acdc_weather_phase_label.txt')

parser = argparse.ArgumentParser()
parser.add_argument('--concat', action='store_true')
parser.add_argument('--config', type=str, default='configs/cityscapes2acdc.yaml', help='Path to the config file.')
parser.add_argument("--resume", action="store_true")
parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")
parser.add_argument('--name', type=str, default='baseline')
parser.add_argument('--weather', type=str, required=True)
# parser.add_argument('--image_root_D', type=str, default='/mnt/ssd1/ssb/dataset/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/val')
# parser.add_argument('--image_list_D', type=str, default='/mnt/ssd1/ssb/dataset/cityscapes/cityscapes_val_image.txt')
# parser.add_argument('--label_root_D', type=str, default='/mnt/ssd1/ssb/dataset/cityscapes/gtFine_trainvaltest/gtFine/val')
# parser.add_argument('--label_list_D', type=str, default='/mnt/ssd1/ssb/dataset/cityscapes/cityscapes_val_label.txt')
# parser.add_argument('--image_root_N', type=str, default='/mnt/ssd1/ssb/dataset/acdc/rgb_anon/weather/val')
# parser.add_argument('--image_list_N', type=str, default='/mnt/ssd1/ssb/dataset/acdc/acdc_weather_phase_image.txt')
# parser.add_argument('--label_root_N', type=str, default='/mnt/ssd1/ssb/dataset/acdc/gt/weather/val')
# parser.add_argument('--label_list_N', type=str, default='/mnt/ssd1/ssb/dataset/acdc/acdc_weather_phase_label.txt')


# '/mnt/hdd2/lyj/Data/KIAPI/'

parser.add_argument('--image_root_D', type=str, default='/mnt/hdd2/lyj/Data/KIAPI/')
parser.add_argument('--image_list_D', type=str, default='/mnt/hdd2/lyj/Data/cityscapes/cityscapes_phase_image.txt')
parser.add_argument('--label_root_D', type=str, default='/mnt/hdd2/lyj/Data/cityscapes/gtFine_trainvaltest/gtFine/phase')
parser.add_argument('--label_list_D', type=str, default='/mnt/hdd2/lyj/Data/cityscapes/cityscapes_phase_label.txt')

# parser.add_argument('--image_root_N', type=str, default='/mnt/hdd2/lyj/Data/acdc/rgb_anon/weather/train')
# parser.add_argument('--image_list_N', type=str, default='/mnt/hdd2/lyj/Data/acdc/acdc_weather_phase_image.txt')
# parser.add_argument('--label_root_N', type=str, default='/mnt/hdd2/lyj/Data/acdc/gt/weather/train')
# parser.add_argument('--label_list_N', type=str, default='/mnt/hdd2/lyj/Data/acdc/acdc_weather_phase_label.txt')

parser.add_argument('--styleIN', action='store_true')
parser.add_argument('--phase', type=str, default='train')

parser.add_argument('--input_folder', type=str, help="input image folder")
parser.add_argument('--output_folder', type=str, default='./results/baseline_day2rain', help="output image folder")
parser.add_argument('--checkpoint', type=str, required=True, help="checkpoint of autoencoders")
parser.add_argument('--a2b', type=int, help="1 for a2b and 0 for b2a", default=1)
parser.add_argument('--seed', type=int, default=1, help="random seed")
parser.add_argument('--num_style',type=int, default=1, help="number of styles to sample")
parser.add_argument('--synchronized', action='store_true', help="whether use synchronized style code or not")
parser.add_argument('--output_only', action='store_true', help="whether only save the output images or also save the input images")
parser.add_argument('--output_path', type=str, default='.', help="path for logs, checkpoints, and VGG model weight")
parser.add_argument('--compute_IS', action='store_true', help="whether to compute Inception Score or not")
parser.add_argument('--compute_CIS', action='store_true', help="whether to compute Conditional Inception Score or not")
parser.add_argument('--inception_a', type=str, default='.', help="path to the pretrained inception network for domain A")
parser.add_argument('--inception_b', type=str, default='.', help="path to the pretrained inception network for domain B")

opts = parser.parse_args()

# opts.image_root_N = opts.image_root_N.replace('weather', opts.weather)
# opts.image_list_N = opts.image_list_N.replace('weather', opts.weather)
# opts.label_root_N = opts.label_root_N.replace('weather', opts.weather)
# opts.label_list_N = opts.label_list_N.replace('weather', opts.weather)

torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)

# Load experiment setting
config = get_config(opts.config)
input_dim = config['input_dim_a'] if opts.a2b else config['input_dim_b']

# Load the inception networks if we need to compute IS or CIIS
if opts.compute_IS or opts.compute_IS:
    inception = load_inception(opts.inception_b) if opts.a2b else load_inception(opts.inception_a)
    # freeze the inception models and set eval mode
    inception.eval()
    for param in inception.parameters():
        param.requires_grad = False
    inception_up = nn.Upsample(size=(299, 299), mode='bilinear')

# Setup model and data loader
traindataset = Day2NightDataset(opts)
train_loader = DataLoader(dataset=traindataset,
                          batch_size=1,
                          shuffle=False,
                          num_workers=2,
                          pin_memory=False)

config['vgg_model_path'] = opts.output_path
if opts.trainer == 'MUNIT':
    style_dim = config['gen']['style_dim']
    trainer = MUNIT_Trainer(config)
elif opts.trainer == 'UNIT':
    trainer = UNIT_Trainer(config)
else:
    sys.exit("Only support MUNIT|UNIT")

try:
    state_dict = torch.load(opts.checkpoint)
    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])
except:
    state_dict = pytorch03_to_pytorch04(torch.load(opts.checkpoint), opts.trainer)
    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])

trainer.cuda()
trainer.eval()

#encode_a, read_a, decode_a = trainer.gen_a.encode, trainer.gen_a.read, trainer.gen_a.decode
#encode_b, read_b, decode_b = trainer.gen_b.encode, trainer.gen_b.read, trainer.gen_b.decode

#encode = trainer.gen_a.encode if opts.a2b else trainer.gen_b.encode # encode function
#decode = trainer.gen_b.decode if opts.a2b else trainer.gen_a.decode # decode function
#read = trainer.gen_a.read if opts.a2b else trainer.gen_b.read 

if opts.compute_IS:
    IS = []
    all_preds = []
if opts.compute_CIS:
    CIS = []

# pdb.set_trace()

if opts.trainer == 'MUNIT':
    # Start testing
    style_fixed = Variable(torch.randn(opts.num_style, style_dim, 1, 1).cuda(), volatile=True)
    for i, (data) in enumerate(train_loader):
        if opts.compute_CIS:
            cur_preds = []
        
        # print(data['D_image_path'])
        x_a, l_a = Variable(data['D_image'].cuda(), volatile=True), Variable(data['D_label'].cuda(), volatile=True)
        ol_a = Variable(data['D_onehot_label'].cuda().detach(), volatile=True)
        # pdb.set_trace()
        with torch.no_grad():
            c_a, s_a = trainer.gen_a.encode(x_a, ol_a)
            ms_b = trainer.gen_a.read(c_a, l_a)
            
            # style normalization
            s_a = F.normalize(s_a, dim=1)
            ms_b = F.normalize(ms_b, dim=1)

            outputs = trainer.gen_b.decode(c_a, s_a, ms_b, l_a)
            if i > 0:
                print(time.time() - start)
            if opts.concat:
                outputs_source_only = trainer.gen_b.decode(c_a, s_a, None, l_a)
                outputs_target_only = trainer.gen_b.decode(c_a, None, ms_b, l_a)
                outputs_concat = (torch.cat((x_a, outputs_source_only, outputs, outputs_target_only), dim=3) + 1) / 2.
            outputs = (outputs + 1) / 2.
        # outputs = F.interpolate(outputs, size=(1024,2048))
        # path = os.path.join(opts.output_folder, 'input{:03d}_output{:03d}.jpg'.format(i, j))
        basename = os.path.basename(data['D_image_path'][0])
        city = basename.split('_')[0]
        path = os.path.join(opts.output_folder, city)
        os.makedirs(path, exist_ok=True)
        image_path = os.path.join(path, basename)
        # pdb.set_trace()
        vutils.save_image(outputs.data, image_path, padding=0, normalize=True)
        if opts.concat:
            path_concat = os.path.join(opts.output_folder+'_concat', city)
            os.makedirs(path_concat, exist_ok=True)
            image_path_concat = os.path.join(path_concat, basename)
            vutils.save_image(outputs_concat.data, image_path_concat, padding=0, normalize=True)
        start = time.time()