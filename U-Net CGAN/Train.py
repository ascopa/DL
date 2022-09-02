# example of training an conditional gan on the fashion mnist dataset
from matplotlib import pyplot
from numpy import ones, mean

import Nets
import Utils
from matplotlib import pyplot as plt


# create a line plot of loss for the gan and save to file
def plot_history(d1_hist, d2_hist, g_hist):
    # plot history
    pyplot.plot(d1_hist, label='crit_real')
    pyplot.plot(d2_hist, label='crit_fake')
    pyplot.plot(g_hist, label='gen')
    pyplot.legend()
    pyplot.savefig('plot_line_plot_loss.png')
    pyplot.close()


# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=128):
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    n_critic = 5
    # lists for keeping track of loss
    c1_hist, c2_hist, g_hist = list(), list(), list()
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    for i in range(n_steps):
        # update the critic more than the generator
        c1_tmp, c2_tmp = list(), list()
        for _ in range(n_critic):
            # get randomly selected 'real' samples
            [X_real, labels_real], y_real = Utils.generate_real_samples(dataset, half_batch)
            # update critic model weights
            c_loss1 = d_model.train_on_batch([X_real, labels_real], y_real)
            c1_tmp.append(c_loss1)
            # generate 'fake' examples
            [X_fake, labels_fake], y_fake = Utils.generate_fake_samples(g_model, latent_dim, half_batch)
            # update critic model weights
            c_loss2 = d_model.train_on_batch([X_fake, labels_fake], y_fake)
            c2_tmp.append(c_loss2)
        # store critic loss
        c1_hist.append(mean(c1_tmp))
        c2_hist.append(mean(c2_tmp))
        # prepare points in latent space as input for the generator
        [z_input, labels_input] = Utils.generate_latent_points(latent_dim, n_batch)
        # create inverted labels for the fake samples
        y_gan = -ones((n_batch, 1))
        # update the generator via the critic's error
        g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
        g_hist.append(g_loss)
        # summarize loss on this batch
        print('>%d/%d, c1=%.3f, c2=%.3f g=%.3f' % (i + 1, n_steps, c1_hist[-1], c2_hist[-1], g_loss))
        # evaluate the model performance every 'epoch'
        if (i + 1) % bat_per_epo == 0:
            summarize_performance(i, g_model, d_model, dataset)
    # line plots of loss
    plot_history(c1_hist, c2_hist, g_hist)


# evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, d_model, dataset, n_samples=15):
    # prepare real samples
    # X_real, y_real = Utils.generate_real_samples(dataset, n_samples)
    # # evaluate discriminator on real examples
    # _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    # # prepare fake examples
    # x_fake, y_fake = Utils.generate_fake_samples(g_model, latent_dim, n_samples)
    # # evaluate discriminator on fake examples
    # _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    # # summarize discriminator performance
    # print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real * 100, acc_fake * 100))
    # save plot
    # Utils.save_plot(x_fake, epoch)
    # save the generator model tile file
    filename = 'dis_model_%03d.h5' % (epoch + 1)
    d_model.save(filename)
    filename = 'gen_model_%03d.h5' % (epoch + 1)
    g_model.save(filename)


# size of the latent space
latent_dim = 100
# load image data
dataset = Utils.load_real_data()
# create the discriminator
d_model = Nets.define_discriminator(dataset[0][0].shape, Utils.dataset_labels)
d_model.summary()
# create the generator
g_model = Nets.define_generator(latent_dim, Utils.dataset_labels)
g_model.summary()
# create the gan
gan_model = Nets.define_gan(g_model, d_model)
# train model
train(g_model, d_model, gan_model, dataset, latent_dim)
