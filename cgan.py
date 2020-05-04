import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from tensorboardX import SummaryWriter

import tqdm
import random
from resnet import resnet18

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument("--id", type=str, required=True, help="program id")
parser.add_argument("--dataset", type=str, default='mnist',  choices=['mnist', 'f_mnist'], help="which dataset")

opt = parser.parse_args()
print(opt)

writer = SummaryWriter(f"logs/{opt.id}")
os.makedirs(f"images/{opt.id}", exist_ok=True)

img_shape = (opt.channels, opt.img_size, opt.img_size*2)

cuda = True if torch.cuda.is_available() else False


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(opt.n_classes*2, opt.n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim + opt.n_classes*2, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            *block(1024, 2048),
            nn.Linear(2048, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        emb = self.label_emb(labels).reshape(labels.shape[0], -1)
        gen_input = torch.cat((emb, noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(opt.n_classes*2, opt.n_classes)

        self.model = nn.Sequential(
            nn.Linear(opt.n_classes*2 + int(np.prod(img_shape)), 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 2),
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        emb = self.label_embedding(labels).reshape(labels.shape[0], -1)
        d_in = torch.cat((img.view(img.size(0), -1), emb), -1)        
        validity = self.model(d_in)
        return validity


# Loss functions
adversarial_loss = torch.nn.MSELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader
os.makedirs(f"../../data/{opt.dataset}", exist_ok=True)
if opt.dataset == 'mnist':
    dataset = datasets.MNIST(
        f"../../data/{opt.dataset}",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    )
elif opt.dataset == 'f_mnist':
    dataset = datasets.FashionMNIST(
        f"../../data/{opt.dataset}",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    )
else:
    raise NotImplementedError('??? no such dataset name')

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True,)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row * 100, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([[i//10, i%10] for _ in range(10) for i in range(100)])
    print(labels[:10,:])
    print(labels[100:110,:])
    print(labels.shape)
    raise NotImplementedError
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)

    save_image(gen_imgs.data, f"images/{opt.id}/{batches_done:#012}.png", nrow=100, normalize=True)


# ----------
#  Training
# ----------
pbar = tqdm.tqdm(range(opt.n_epochs))
iter_step = 0
for epoch in pbar:
    for i, (imgs, labels) in enumerate(dataloader):
        iter_step += 1
        # b, c, h, w - > b/2, c, h, 2*w
        assert(imgs.shape[0] % 2 == 0)
        batch_size = imgs.shape[0]//2
        real_rand = np.random.random(batch_size)
        batch0 = imgs[:batch_size,:,:,:]
        batch1 = imgs[batch_size:,:,:,:]
        imgs = torch.cat((batch0, batch1), dim=3)
        labels[:batch_size][real_rand<0.5] = 10
        labels[batch_size:][real_rand>=0.5] = 10
        labels = labels.unsqueeze(1)
        labels = torch.cat((labels[:batch_size,:], labels[batch_size:,:]), dim=1)

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, (batch_size, 2))))
        fake_rand = np.random.random(batch_size)
        gen_labels[fake_rand<0.5,0] = 10
        gen_labels[fake_rand>=0.5,1] = 10
        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs, gen_labels)
        g_loss_l = adversarial_loss(validity[fake_rand<0.5,0], valid[fake_rand<0.5])
        g_loss_r = adversarial_loss(validity[fake_rand>=0.5,1], valid[fake_rand>=0.5])
        if random.random() < 0.5:
            g_loss_l.backward(retain_graph=True)
            g_loss_r.backward()
        else:
            g_loss_r.backward(retain_graph=True)
            g_loss_l.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        validity_real = discriminator(real_imgs, labels)
        d_real_loss_l = adversarial_loss(validity_real[real_rand<0.5,0], valid[real_rand<0.5])
        d_real_loss_r = adversarial_loss(validity_real[real_rand>=0.5,1], valid[real_rand>=0.5])

        # Loss for fake images
        validity_fake = discriminator(gen_imgs.detach(), gen_labels)
        d_fake_loss_l = adversarial_loss(validity_fake[fake_rand<0.5,0], fake[fake_rand<0.5])
        d_fake_loss_r = adversarial_loss(validity_fake[fake_rand>=0.5,1], fake[fake_rand>=0.5])

        # Total discriminator loss
        d_loss_l = (d_real_loss_l + d_fake_loss_l) / 2
        d_loss_r = (d_real_loss_r + d_fake_loss_r) / 2

        if random.random() < 0.5:
            d_loss_l.backward(retain_graph=True)
            d_loss_r.backward()
        else:
            d_loss_r.backward(retain_graph=True)
            d_loss_l.backward()
        optimizer_D.step()

        state_msg = \
            "[Epoch %d/%d] [Batch %d/%d] [D loss_l: %f] [G loss_l: %f] [D loss_r: %f] [G loss_r: %f]" \
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss_l.item(), g_loss_l.item(), d_loss_r.item(), g_loss_r.item())

        pbar.set_description(state_msg)
        if (iter_step+1) % 10 == 0:
            writer.add_scalars('loss/loss_l', {'D_l':d_loss_l.item(), 'G_l':g_loss_l.item()}, iter_step)
            writer.add_scalars('loss/loss_r', {'D_r':d_loss_r.item(), 'G_r':g_loss_r.item()}, iter_step)

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            sample_image(n_row=10, batches_done=batches_done)