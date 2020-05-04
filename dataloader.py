from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms
import random

class MNISTDataset(data.Dataset):
    def __init__(self, mode):
        super(MNISTDataset, self).__init__()
        if mode == 'train':
            self.dataset = datasets.MNIST('../data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.Resize(32),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=(0.5,), std=(0.5,))
                        ]))
        else:
            self.dataset = datasets.MNIST('../data', train=False, download=True,
                        transform=transforms.Compose([
                            transforms.Resize(32),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=(0.5,), std=(0.5,))
                        ])) 

    def name(self):
        return "MNISTDataset"

    def __getitem__(self, index):
        # rand = random.randint(0, len(self.dataset)-1)
        return self.dataset[index] # , self.dataset[rand]
    
    def __len__(self):
        return len(self.dataset)

class MNISTDataloader(object):
    def __init__(self, mode, opt, dataset):
        super(MNISTDataloader, self).__init__()
        use_cuda = not torch.cuda.is_available()
        kwargs = {'num_workers': opt.num_workers} if use_cuda else {}

        self.data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=opt.batch_size, shuffle=True, **kwargs)

        self.data_iter = self.data_loader.__iter__()

    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num-workers', type=int, default = 4)
    parser.add_argument('-e', '--epoch', type=int, default=40)
    parser.add_argument('-b', '--batch-size', type=int, default = 100)
    parser.add_argument('-d', '--display-step', type=int, default = 600)
    opt = parser.parse_args()

    train_dataset = MNISTDataset('train')
    train_data_loader = MNISTDataloader('train', opt, train_dataset)

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num-workers', type=int, default = 4)
    parser.add_argument('-e', '--epoch', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default = 1)
    parser.add_argument('-d', '--display-step', type=int, default = 600)
    opt = parser.parse_args()

    test_dataset = MNISTDataset('test')
    test_data_loader = MNISTDataloader('test', opt, test_dataset)

    print('[+] Size of the train dataset: %05d, train dataloader: %03d' \
        % (len(train_dataset), len(train_data_loader.data_loader)))   
    print('[+] Size of the test dataset: %05d, test dataloader: %03d' \
        % (len(test_dataset), len(test_data_loader.data_loader)))