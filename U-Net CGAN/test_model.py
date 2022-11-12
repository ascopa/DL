import os
import shutil
from datetime import datetime
from os import listdir

import numpy
from keras.models import load_model
from matplotlib import pyplot
from numpy import ones, zeros

import Utils

# Params to modify before testing
model_to_test_dir = os.path.join("F:", os.sep, "backup", "Facultad", "Tesis", "DL", "U-Net CGAN", "models_to_test")

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
        pyplot.savefig(test_model_directory + os.sep + "generated_image_" + str(i) + ".png")
        pyplot.close()


for model_to_test in listdir(model_to_test_dir):

    model_path = os.path.join(model_to_test_dir, model_to_test)
    item_label = Utils.CancerLabel.nv

    # Create and save files of test execution
    now = datetime.now()
    parent_dir = os.path.join(os.getcwd(), "generated images")
    directory = str(item_label) + " - " + now.strftime("%d%m%Y-%H%M%S")
    test_model_directory = os.path.join(parent_dir, directory)
    os.mkdir(test_model_directory)

    generator_model = load_model(model_path, compile=False)
    # Save generator model

    shutil.copyfile(model_path, test_model_directory + os.sep + model_to_test)


    samples = 100
    channel = 1
    latent_dim = 100

    # generate images

    z_input = Utils.get_noise(latent_dim, samples)

    real_images, _ = Utils.get_images_and_labels(samples, 64)

    labels_input = ones((samples, 1)) * item_label

    fake_images = generator_model.predict([real_images, labels_input, z_input])

    # save generated images
    numpy.save("generated_images", fake_images)
    # plot the result
    save_plot(numpy.load("generated_images.npy"), samples)
