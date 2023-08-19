import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torch.nn as nn
import numpy as np
from torchvision.datasets import MNIST
device = torch.device('cuda')

BATCH_SIZE = 128
dataset = MNIST(root='data', download=False, transform=ToTensor())


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding='same')
        self.norm1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding='same')
        self.norm2 = nn.BatchNorm2d(8)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding='same')
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.up1 = nn.Upsample((7, 7))
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding='same')
        self.up2 = nn.Upsample((14, 14))
        self.conv5 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding='same')
        self.up3 = nn.Upsample((28, 28))
        self.conv6 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding='same')
        self.conv7 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding='same')
        self.norm3 = nn.BatchNorm2d(1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.relu(self.conv1(input))
        x = self.norm1(x)
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.norm2(x)
        x = self.pool2(x)
        x = self.relu(self.conv3(x))
        x = self.norm2(x)
        x = self.pool3(x)

        x = self.up1(x)
        x = self.relu(self.conv4(x))
        x = self.norm2(x)
        x = self.up2(x)
        x = self.relu(self.conv5(x))
        x = self.norm2(x)
        x = self.up3(x)
        x = self.relu(self.conv6(x))
        x = self.norm1(x)
        x = self.sigmoid(self.conv7(x))
        x = self.norm3(x)

        return x


if __name__ == '__main__':

    model = Autoencoder().to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    n_epoch = 20
    trainset = torch.utils.data.Subset(dataset, range(0, 50000))
    dataloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(n_epoch):
        for img, _ in dataloader:
            img = img.to(device)
            generated = model(img)
            loss = criterion(generated, img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('epoch:', epoch, 'loss:', loss.item())

    torch.save(model.state_dict(), 'autoencoder3.pth')





