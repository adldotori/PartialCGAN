import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(20, 10)

        self.model = nn.Sequential(
            nn.Linear(120, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 2048),
            nn.Tanh()
        ) 

    def forward(self, z, y):
        z = z.view(z.size(0), 100)
        y = self.embedding(y)
        y = y.view(y.size(0), -1)
        ret = torch.cat([z, y], 1)
        ret = self.model(ret)
        return ret.view(z.size(0), -1, 32, 64)
         
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(20, 10)

        self.model = nn.Sequential(
            nn.Linear(2068, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 2),
        )

    def forward(self, x, y):
        x = x.view(x.size(0), 2048)
        y = self.embedding(y)
        y = y.view(y.size(0), -1)
        ret = torch.cat([x, y], 1)
        ret = self.model(ret)
        return ret

if __name__ == '__main__':
    batch_size = 100

    generator = Generator().cuda()
    noise = torch.randn(batch_size, 100).cuda()
    label1 = np.random.randint(0, 10,(batch_size,))
    label2 = np.random.randint(0, 10,(batch_size,))
    rand = np.random.random(batch_size)
    label1[rand<0.5] = -1
    label2[rand>=0.5] = -1
    label = np.stack([label1, label2], 1)
    label = torch.from_numpy(label).cuda()
    print(noise.shape, label.shape)
    gen = generator(noise, label)
    print(gen.shape)

    discriminator = Discriminator()
    discriminator.cuda()
    image = torch.rand(batch_size, 1, 32, 64).cuda()
    dis = discriminator(image, label)
    print(dis.shape)