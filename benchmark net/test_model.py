import os
from random import random

import numpy
from PIL import Image
from keras.models import load_model
from matplotlib import pyplot
from numpy import zeros

import Utils

img_dir_save = os.path.join("F:", os.sep, "backup", "Facultad", "Tesis", "DL", "benchmark net", "generated images")


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
        pyplot.savefig(img_dir_save + os.sep + "generated_image_" + str(i) + ".png")
        pyplot.close()


samples = 50
# load model
model = load_model('gan_model_100.h5')
# generate images
seg_imgs = Utils.load_segmentation_data()
noise = Utils.get_noise_data(samples, 400)
generated_images = model.predict([seg_imgs[0:samples], noise])
numpy.save("generated_images", generated_images)


# plot the result
save_plot(numpy.load("generated_images.npy"), samples)
