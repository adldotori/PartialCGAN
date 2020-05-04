import torch
import torch.nn as nn

class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, x, y):
        return self.loss(x,y)

if __name__ == '__main__':
    batch_size = 3

    x = torch.rand(batch_size,).cuda()
    y = torch.rand(batch_size,).cuda()

    criterion = Loss()
    loss = criterion(x, y)

    print(loss.shape)
    print(loss)
