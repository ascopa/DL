import numpy
import numpy as np
from matplotlib import pyplot
from numpy import expand_dims


def get_real_data():
    return numpy.load("../datasets/benchmark net/real_images.npy")



def get_segmentation_data():
    return numpy.load("../datasets/benchmark net/segmentation_images.npy")

def load_real_data():
    train_data = get_real_data()
    # add channel dimension
    x = expand_dims(train_data, axis=-1)
    return x


def load_segmentation_data():
    train_data = get_segmentation_data()
    # add channel dimension
    x = expand_dims(train_data, axis=-1)
    return x


def get_noise_data(batch_size, z_size):
    return np.random.normal(0, 0.001, size=[batch_size, z_size]).astype(np.float32)


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
