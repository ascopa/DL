import os

import numpy
import numpy as np
from keras.datasets.fashion_mnist import load_data
from keras_preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from numpy import expand_dims
from numpy import ones
from numpy.random import randint
from numpy.random import randn

image_size = 224
noise_size = 400
dataset_labels = 7
# img_dir_source = os.path.join("F:", os.sep, "backup", "Facultad", "Tesis", "DL", "datasets", "Drive", "augmented",
#                               "drive_data_" + str(image_size) + ".npy")
#
#
# def get_real_data():
#     return numpy.load(img_dir_source)

# Define datagen. Here we can define any transformations we want to apply to images
datagen = ImageDataGenerator()

# define training directory that contains subfolders
train_dir = os.path.join("F:", os.sep, "backup", "Facultad", "Tesis", "DL", "datasets", "HAM10000", "data",
                         "reorganized")


# USe flow_from_directory


def load_real_data_old():
    # load dataset
    (trainX, trainy), (_, _) = load_data()
    # expand to 3d, e.g. add channels
    X = expand_dims(trainX, axis=-1)
    # convert from ints to floats
    X = X.astype('float32')
    # scale from [0,255] to [-1,1]
    X = (X - 127.5) / 127.5
    return [X, trainy]


def load_real_data():
    # emulation dataset loading
    train_data_keras = datagen.flow_from_directory(directory=train_dir,
                                                   class_mode='categorical',
                                                   batch_size=1,  # 16 images at a time
                                                   target_size=(image_size, image_size),
                                                   color_mode='grayscale')  # Resize images
    # split into images and labels
    images, labels = next(train_data_keras)
    size = train_data_keras.samples
    images = numpy.zeros([size, images[0][0].size, images[0][0].size, 1])
    return [images, labels]


def get_images_and_labels(n_samples):
    train_data_keras = datagen.flow_from_directory(directory=train_dir,
                                                   class_mode='sparse',
                                                   batch_size=n_samples,  # 16 images at a time
                                                   target_size=(image_size, image_size),
                                                   color_mode='grayscale')  # Resize images
    # split into images and labels
    images, labels = next(train_data_keras)
    # labels = numpy.argmax(labels, axis=-1)
    # convert from ints to floats
    images = images.astype('float32')
    # scale from [0,255] to [-1,1]
    images = (images - 127.5) / 127.5
    # generate class labels
    return images, labels.astype(int)


def generate_real_samples(n_samples):
    images, labels = get_images_and_labels(n_samples)
    # generate class labels
    y = -ones((n_samples, 1))
    return [images, labels.astype(int)], y


def get_noise(latent_dim, n_samples):
    return randn(latent_dim * n_samples).reshape(n_samples, latent_dim)


# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
    img_input, labels_input = get_images_and_labels(n_samples)
    z_input = get_noise(latent_dim, n_samples)
    # predict outputs
    images = generator.predict([img_input, labels_input, z_input])
    # create class labels
    y = ones((n_samples, 1))
    return [images, labels_input], y


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


from enum import IntEnum


class FashionLabel(IntEnum):
    Tshirt = 0
    Trouser = 1
    Pullover = 2
    Dress = 3
    Coat = 4
    Sandal = 5
    Shirt = 6
    Sneaker = 7
    Bag = 8
    Ankle_boot = 9


class CancerLabel(IntEnum):
    akiec = 0
    bcc = 1
    bkl = 2
    df = 3
    mel = 4
    nv = 5
    vasc = 6


