from numpy import mean
import Nets
import Utils
import tensorflow as tf
import wandb

# from google.colab import drive
# drive.mount("/content/gdrive")

wandb.login()

hyperparams = {
    "epochs": 100,
    "batch_size": 128,
    "image_size": 64,
    "latent_dim": 100,
    "dataset_labels": 7,
    "clip_value": 0.01,
    "lr": 0.001,
    "dropout": 0.5,
    "beta_1": 0.9,
    "beta_2": 0.99
}

wandb.init(
    project="WCGAN",
    config=hyperparams)

# Copy your config
config = wandb.config

# Create net
dataset = Utils.load_real_data(config.image_size)
image_shape = dataset[0][0].shape
d_model = Nets.define_discriminator(image_shape, config.dataset_labels)
g_model = Nets.define_generator(config.latent_dim, image_shape, config.dataset_labels)
gan_model = Nets.define_gan(g_model, d_model)

save_dir = Utils.create_save_dir()
# save_hyperparams(save_dir)

# Start training
bat_per_epo = int(dataset[0].shape[0] / config.batch_size)
half_batch = int(config.batch_size / 2)
n_critic = 5
# lists for keeping track of loss
c1_hist, c2_hist, g_hist = list(), list(), list()
# calculate the number of training iterations
n_steps = bat_per_epo * config.epochs
X_real, labels_real, y_real = [], [], []
for i in range(n_steps):
    # update the critic more than the generator
    c1_tmp, c2_tmp = list(), list()
    for _ in range(n_critic):
        # get randomly selected 'real' samples
        [X_real, labels_real], y_real = Utils.generate_real_samples(half_batch)
        # update critic model weights
        c_loss1 = d_model.train_on_batch([X_real, labels_real], y_real)
        real_images_logits = d_model([X_real, labels_real])
        c1_tmp.append(c_loss1)
        # generate 'fake' examples
        [X_fake, labels_fake], y_fake = Utils.generate_fake_samples(g_model, config.latent_dim, half_batch)
        # update critic model weights
        c_loss2 = d_model.train_on_batch([X_fake, labels_fake], y_fake)
        c2_tmp.append(c_loss2)
        # store critic loss
        c1_hist.append(mean(c1_tmp))
        c2_hist.append(mean(c2_tmp))

        for l in d_model.layers:
            weights = l.get_weights()
            weights = [tf.clip_by_value(w, -config.clip_value, config.clip_value) for w in weights]
            l.set_weights(weights)

    # prepare points in latent space as input for the generator
    img_input, labels_input, y_gan = X_real, labels_real, y_real
    # img_input, labels_input, y_gan = Utils.generate_real_samples(n_batch)
    z_input = Utils.get_noise(config.latent_dim, half_batch)
    # update the generator via the critic's error
    g_loss = gan_model.train_on_batch([img_input, labels_input, z_input], y_gan)
    g_hist.append(g_loss)
    # summarize loss on this batch
    print('>%d/%d, c1=%.3f, c2=%.3f g=%.3f' % (i + 1, n_steps, c1_hist[-1], c2_hist[-1], g_loss))

    metrics = {
        "real_loss": c1_hist[-1],
        "fake_loss": c2_hist[-1],
        "generator_loss": g_loss,
        "step": i,
        "total_steps": n_steps
    }
    # Log train metrics to wandb
    wandb.log(metrics)
    # evaluate the model performance every 'epoch'
    if (i + 1) % bat_per_epo == 0:
        Utils.save_models(i, g_model, d_model, save_dir)
wandb.finish()
