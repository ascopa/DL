import os

import numpy as np
import torch
from matplotlib import pyplot as plt, pyplot
import torchvision.utils as vutils

from tut_dcgan.Train import Generator

dataroot = os.path.join("F:", os.sep, "backup", "Facultad", "Tesis", "DL", "datasets", "CelebA")
models_dir = os.path.join("F:", os.sep, "backup", "Facultad", "Tesis", "DL", "tut_dcgan", "models")

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
nz = 100
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

model = Generator(1)
model.load_state_dict(torch.load(os.path.join(models_dir, 'netG-dict-1500-4.pth')))
model.eval()
fake = model(fixed_noise).detach().cpu()


# # Plot the real images
# plt.figure(figsize=(15, 15))
# plt.subplot(1, 2, 1)
# plt.axis("off")
# plt.title("Real Images")
# plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

def save_plot(examples, n):
    # plot images
    plot_size = 3
    for i in range(n):
        # define subplot
        # pyplot.subplot(plot_size, plot_size, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(examples[i, :, :], cmap='gray_r')
        pyplot.savefig(models_dir + os.sep + "generated_image_" + str(i) + ".png")
        pyplot.close()


# save_plot(np.transpose(fake[-1], (1, 2, 0)), 64)
def print_imgs(imgs):
    for i in range(imgs):
        plt.subplot(1, 3, i)
        plt.axis("off")
        plt.title("Fake Images")
        plt.plot(np.transpose(fake[-1], (1, 2, 0)))
        plt.show()
