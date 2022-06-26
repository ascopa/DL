import random

from numpy import ones, zeros, vstack

import Nets
import Utils


def generate_real_samples(dataset, n_samples):
    # choose random instances
    ix = random.sample(range(0, dataset.shape[0]), n_samples)
    # retrieve selected images
    x = dataset[ix]
    # generate 'real' class labels (1)
    y = ones((n_samples, 1))
    return x, y

def generate_fake_samples(g_model, gen_dataset, n_samples):
    rnd = random.sample(range(0, dataset.shape[0]), n_samples)
    # get one random segmentation img for prediction
    x = g_model.predict([gen_dataset[0][rnd], gen_dataset[1]], n_samples)
    # create 'fake' class labels (0)
    y = zeros((n_samples, 1))
    return x, y


def train(g_model, d_model, gan_model, dataset, gen_dataset, noise_size, image_size, n_epochs=100, n_batch=1):
    batch_per_epo = dataset.shape[0]
    print("Batchs per epoch:" + str(batch_per_epo))
    for i in range(n_epochs):
        for j in range(batch_per_epo):
            # get randomly selected 'real' samples
            X_real, y_real = generate_real_samples(dataset, n_batch)
            # generate 'fake' examples
            X_fake, y_fake = generate_fake_samples(g_model, gen_dataset, n_batch)
            # create training set for the discriminator
            X, y = vstack((X_real, X_fake)), vstack((y_real, y_fake))
            # print(X.shape, y.shape)
            # update discriminator model weights
            d_loss, _ = d_model.train_on_batch(X, y)
            # create inverted labels for the fake samples
            y_gan = ones((n_batch, 1))
            # update the generator via the discriminator's error
            gan_img_input = gen_dataset[0][j].reshape((-1, image_size, image_size))
            gan_noise_input = gen_dataset[1].reshape((-1, noise_size))
            _ = gan_model.train_on_batch([gan_img_input, gan_noise_input], y_gan)
            g_loss = gan_model.train_on_batch([gan_img_input, gan_noise_input], y_gan)
            # summarize loss on this batch
            print('>%d, %d/%d, d=%.3f, g_0=%.3f, g_1=%.3f' % (i + 1, j + 1, batch_per_epo, d_loss, g_loss[0], g_loss[1]))
            # evaluate the model performance, sometimes
        if (i + 1) % 10 == 0:
            summarize_performance(i, g_model, d_model, dataset, gen_dataset)

# evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, d_model, dataset, gen_dataset, n_samples=1):
    # prepare real samples
    X_real, y_real = generate_real_samples(dataset, n_samples)
    # evaluate discriminator on real examples
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = generate_fake_samples(g_model, gen_dataset, n_samples)
    # evaluate discriminator on fake examples
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    # summarize discriminator performance
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real * 100, acc_fake * 100))
    # save plot
    Utils.save_plot(x_fake, epoch)
    # save the generator model tile file
    filename = 'gan_model_%03d.h5' % (epoch + 1)
    g_model.save(filename)


noise_size = 400
image_size = 512
batch_size = 1

# load image data
dataset = Utils.load_real_samples()
channel = dataset.shape[3]
gen_dataset = Utils.get_generator_input(batch_size, noise_size)
# create the discriminator
d_model = Nets.Discriminator([image_size, image_size, channel]).model
# d_model.summary()
# create the generator
g_model = Nets.Generator([image_size, image_size, channel], [noise_size]).model
# g_model.summary()
# create the gan
gan_model = Nets.Gan(g_model, d_model).model
# train model
train(g_model, d_model, gan_model, dataset, gen_dataset, noise_size, image_size)