import random

import torch
from autoencoder2 import Autoencoder, dataset
import matplotlib.pyplot as plt
import numpy as np

model = Autoencoder().to(torch.device('cuda'))
sd = torch.load('autoencoder3.pth')
model.load_state_dict(sd)

testset = torch.utils.data.Subset(dataset, range(50000, 60000))
with torch.no_grad():
    img, _ = testset[1005]
    img = img.to(torch.device('cuda'))
    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    ax.imshow(img[0].cpu(), cmap="gray_r")
    ax = fig.add_subplot(2, 1, 2)
    ax.imshow(model(img.unsqueeze(dim=0))[0][0].cpu(), cmap="gray_r")
    plt.show()


