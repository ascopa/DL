import random

import numpy as np
from PIL import Image
from matplotlib import pyplot
from numpy import ones, zeros, vstack

import Nets
import Utils


def generate_real_samples(dataset, n_samples):
    # choose random instances
    ix = random.sample(range(0, dataset.shape[0]), n_samples)
    # retrieve selected images
    x = dataset[ix]
    # pyplot.imshow(x[5], cmap="gray")
    # pyplot.show()
    # generate 'real' class labels (1)
    y = ones((n_samples, 1))
    return x, y


def generate_fake_samples(g_model, dataset, n_samples):
    # get n_samples random segmentation img for prediction
    seg_img = dataset[random.sample(range(0, dataset.shape[0]), n_samples)]
    noise = Utils.get_noise_data(n_samples)
    x = g_model.predict([seg_img, noise])
    # pyplot.imshow(x[3], cmap="gray")
    # pyplot.show()
    # create 'fake' class labels (0)
    y = zeros((n_samples, 1))
    return x, y


def train(g_model, d_model, gan_model, real_img_dataset, seg_img_dataset, image_size, n_batch, n_epochs=100):
    batch_per_epo = int(real_img_dataset.shape[0] / n_batch)
    print("Batchs per epoch:" + str(batch_per_epo))
    for i in range(n_epochs):
        for j in range(batch_per_epo):
            # get randomly selected 'real' samples
            X_real, y_real = generate_real_samples(real_img_dataset, n_batch)
            # train discriminator on real samples
            d_r_loss, d_r_acc = d_model.train_on_batch(X_real, y_real)
            # generate 'fake' examples
            X_fake, y_fake = generate_fake_samples(g_model, seg_img_dataset, n_batch)
            X, y = vstack((X_real, X_fake)), vstack((y_real, y_fake))
            # train discriminator on real samples
            d_f_loss, d_f_acc = d_model.train_on_batch(X_fake, y_fake)
            # create inverted labels for the fake samples so the discriminator thinks they are real
            y_gan = ones((n_batch, 1))
            # get gan input
            gan_img_input = seg_img_dataset[random.sample(range(0, seg_img_dataset.shape[0]), n_batch)].reshape((-1, image_size, image_size, channel))
            gan_noise_input = Utils.get_noise_data(n_batch)
            # update generator weights twice
            _ = gan_model.train_on_batch([gan_img_input, gan_noise_input], y_gan)
            g_loss = gan_model.train_on_batch([gan_img_input, gan_noise_input], y_gan)
            # summarize loss on this batch
            # print('>%d, %d/%d, dr=%.3f, ar=%.3f, df=%.3f, af=%.3f, dg=%.3f' % (i + 1, j + 1, batch_per_epo, d_r_loss, d_r_acc, d_f_loss, d_f_acc, g_loss[0]))
            print('>%d, %d/%d, dr=%.3f, ar=%.3f' % (i + 1, j + 1, batch_per_epo, d_r_loss, d_r_acc,  ))

            # evaluate the model performance, sometimes
        if (i + 1) % 10 == 0:
            summarize_performance(i, g_model, d_model, real_img_dataset, seg_img_dataset)


# evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, d_model, real_img_dataset, seg_img_dataset, n_samples=15):
    # prepare real samples
    X_real, y_real = generate_real_samples(real_img_dataset, n_samples)
    # evaluate discriminator on real examples
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = generate_fake_samples(g_model, seg_img_dataset, n_samples)
    # evaluate discriminator on fake examples
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    # summarize discriminator performance
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real * 100, acc_fake * 100))
    # save plot
    # Utils.save_plot(x_fake, epoch)
    # save the generator model tile file
    filename = 'dis_model_%03d.h5' % (epoch + 1)
    d_model.save(filename)
    filename = 'gen_model_%03d.h5' % (epoch + 1)
    g_model.save(filename)


batch_size = 10
channel = 1
# load data
real_img_dataset = Utils.load_real_data()
# create the discriminator
d_model = Nets.discriminator([Utils.image_size, Utils.image_size, channel])
d_model.summary()
# create the generator
g_model = Nets.generator([Utils.image_size, Utils.image_size, channel], [Utils.noise_size])
g_model.summary()
# create the gan
gan_model = Nets.gan(g_model, d_model)

model_json = g_model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# train model
train(g_model, d_model, gan_model, real_img_dataset, real_img_dataset, Utils.image_size, batch_size)
