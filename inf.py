import random

import torch
from autoencoder2 import Autoencoder, dataset
import matplotlib.pyplot as plt
import numpy as np



def select_images(images, labels, num_images=10):
    sample_images_index = np.random.choice(range(len(images)), num_images)
    sample_images = images[sample_images_index]
    sample_labels = labels[sample_images_index]
    return sample_images, sample_labels


def plot_reconstructed_images(images, reconstructed_images):
    fig = plt.figure(figsize=(15, 3))
    num_images = len(images)
    for i, (image, reconstructed_image) in enumerate(zip(images, reconstructed_images)):
        image = image.squeeze()
        ax = fig.add_subplot(2, num_images, i + 1)
        ax.axis("off")
        ax.imshow(image, cmap="gray_r")
        reconstructed_image = reconstructed_image.squeeze()
        ax = fig.add_subplot(2, num_images, i + num_images + 1)
        ax.axis("off")
        ax.imshow(reconstructed_image, cmap="gray_r")
    plt.show()


model = Autoencoder().to(torch.device('cuda'))
sd = torch.load('autoencoder3.pth')
model.load_state_dict(sd)



# num_sample_images_to_show = 8
# sample_images, _ = select_images(dataset.data, dataset.targets, num_sample_images_to_show)
# reconstructed_images = model(sample_images)
# plot_reconstructed_images(sample_images, reconstructed_images)
#

imgs = []
gen = []

# for i in range(8):
#     idx = random.randint(0, len(dataset))
#     imgs.append(dataset[idx])
#     gen.append(model(dataset[idx]))
#     plot_reconstructed_images(imgs, gen)

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


