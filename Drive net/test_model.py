import os

import numpy
from numpy import ones, zeros

import Nets
from keras.models import load_model
from matplotlib import pyplot

import Utils

img_dir_save = os.path.join("F:", os.sep, "backup", "Facultad", "Tesis", "DL", "Drive net", "generated images")


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


samples = 100
channel = 1
# load model
const = Nets.ClipConstraint(0.01)
# discriminator_model = load_model('dis_model_070.h5', custom_objects={"kernel_constraint": const})
generator_model = load_model('gen_model_050.h5')
# generate images
real_images = Utils.load_real_data()
noise = Utils.get_noise_data(samples)
fake_images = generator_model.predict([real_images[0:samples], noise])
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
