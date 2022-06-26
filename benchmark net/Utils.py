import numpy
import numpy as np
from matplotlib import pyplot
from numpy import expand_dims

def get_real_data():
    return numpy.load("real_images.npy")

def get_segmentation_data():
    return numpy.load("segmentation_images.npy")

# load and prepare training images
def load_real_samples():
    train_data = get_real_data()
    # expand to 3d, e.g. add channels dimension
    x = expand_dims(train_data, axis=-1)
    return x

def get_generator_input(batch_size, z_size):
    seg_imgs = get_segmentation_data()
    noise = np.random.normal(0, 0.001, size=[batch_size, z_size]).astype(np.float32)
    return seg_imgs, noise

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