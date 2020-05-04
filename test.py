import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from model import *
from loss import *
from dataloader import *

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num-workers', type=int, default = 4)
    parser.add_argument('-e', '--epoch', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default = 1)
    parser.add_argument('-d', '--display-step', type=int, default = 1)
    opt = parser.parse_args()
    return opt

def imsave(result,path):
    img = result[0] * 255
    img = img.cpu().clamp(0,255)
    img = img.detach().numpy().astype('uint8')
    Image.fromarray(img).save(path)

def test(opt):
    # Init Model
    generator = Generator().cuda()
    generator.load_state_dict(torch.load('checkpoint.pt'))
    generator.train()

    # Test
    z = Variable(torch.randn(1000, 100)).cuda()
    label = np.array([[i//10, i%10] for _ in range(10) for i in range(100)])
    label = Variable(torch.LongTensor(label)).cuda()
    print(z.shape, label.shape)
    sample_images = generator(z, label)
    grid = save_image(sample_images.data, 'result.png', nrow=100, normalize=True)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    opt = get_opt()
    test(opt)