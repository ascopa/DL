import os

import numpy
import numpy as np
from PIL import Image
from matplotlib import pyplot
from numpy import expand_dims

image_size = 128
noise_size = 400

img_dir_source = os.path.join("F:", os.sep, "backup", "Facultad", "Tesis", "DL", "datasets", "Drive", "augmented", "drive_data_" + str(image_size) + ".npy")


def get_real_data():
    return numpy.load(img_dir_source)


def load_real_data():
    train_data = get_real_data()
    # pyplot.imshow(train_data[2])
    # add channel dimension
    x = expand_dims(train_data, axis=-1)
    return x


def get_noise_data(batch_size):
    return np.random.normal(0, 0.001, size=[batch_size, noise_size]).astype(np.float32)


def save_plot(examples, epoch, n=10):
    # plot images
    for i in range(n * n):
        # define subplot
        pyplot.subplot(n, n, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
    # save plot to file
    filename = 'generated_plot_e%03d.png' % (epoch + 1)
    pyplot.savefig(filename)
    pyplot.close()
