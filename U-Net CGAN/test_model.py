import os
import shutil
from datetime import datetime

import numpy
from keras.models import load_model
from matplotlib import pyplot
from numpy import ones, zeros

import Utils

# Params to modify before testing
model_to_test = 'gen_model_46800.h5'
item_label = Utils.FashionLabel.Coat

# Create and save files of test execution
now = datetime.now()
# TODO poner bien el path del parent dir
parent_dir = os.path.join(os.getcwd(), "generated images")
directory = str(item_label) + " - " + now.strftime("%d%m%Y-%H%M%S")
test_model_directory = os.path.join(parent_dir, directory)
os.mkdir(test_model_directory)

generator_model = load_model(model_to_test)
# Save generator model

shutil.copyfile(model_to_test, test_model_directory + os.sep + model_to_test)


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


samples = 100
channel = 1
labels = 10
latent_dim = 100

# generate images
real_images = Utils.load_real_data()
[z_input, _] = Utils.generate_latent_points(latent_dim, samples)
labels_input = ones((samples, 1)) * item_label
fake_images = generator_model.predict([z_input, labels_input], labels)
y_real = ones((samples, 1))
y_fake = zeros((samples, 1))

# #predict
# _, real_images_acc = discriminator_model.evaluate(real_images[0:samples], y_real)
# _, fake_images_acc = discriminator_model.evaluate(fake_images, y_fake)
numpy.save("generated_images", fake_images)

# print("real acc:" + str(real_images_acc))
# print("fake acc:" + str(fake_images_acc))


# plot the result
save_plot(numpy.load("generated_images.npy"), samples)
