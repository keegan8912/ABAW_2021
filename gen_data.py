from __future__ import print_function
# %matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from wild_dataset import wildSet
from torchvision.utils import save_image
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader
from warnings import filterwarnings
from torch.utils.tensorboard import SummaryWriter

filterwarnings("ignore")
# manualSeed = 999


with open("/var/storage/cube-data/others/df_collection_clean3.pickle", "rb") as input_file:
    df2 = pickle.load(input_file)
device = torch.device("cuda:2")

# Number of workers for dataloader
workers = 2
# Batch size during training
batch_size = 128
# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64
# Number of channels in the training images. For color images this is 3
nc = 3
# Size of z latent vector (i.e. size of generator input)
nz = 100
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64
# Number of training epochs
num_epochs = 50
# Learning rate for optimizers
lr = 0.0002
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


# Create the generator
netG = Generator(ngpu).to(device)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# Create the Discriminator
netD = Discriminator(ngpu).to(device)

g_path = '/var/storage/cube-data/others/Code/dc_torch_G_150_cls'
d_path = '/var/storage/cube-data/others/Code/dc_torch_D_150_cls'
b_size = 128
cls = [1, 2, 3, 5, 6]
num_img_req = {
    1: 195556,
    2: 205892,
    3: 207517,
    5: 116986,
    6: 178011,
}
num_img_128 = {
    1: 1527,
    2: 1608,
    3: 1621,
    4: 526,
    5: 913,
    6: 1390
}

for i in cls:
    print(f"Generating image of class {i}")
    g_checkpoint = torch.load(g_path + str(i) + '.pt')
    netG.load_state_dict(g_checkpoint['state_dict'])
    d_checkpoint = torch.load(d_path + str(i) + '.pt')
    netD.load_state_dict(d_checkpoint['state_dict'])
    # Generate fake image batch with G
    for runs in tqdm(range(num_img_req[i])):
        manualSeed = random.randint(1, 10000)  # use if you want new results
        # print("Random Seed: ", manualSeed)
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake = netG(noise)
        # for j in range(len(fake)):
        path = "/var/storage/cube-data/others/Code/images/cls"+str(i)+"/"
        save_image(fake[0], path+"images"+str(runs)+".png", nrow=1,
                       normalize=True)

#
# 0    218364
# 4    150932
# 5    101378
# 6     40353
# 1     22808
# 2     12472
# 3     10847
