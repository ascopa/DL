import tensorflow as tf
from keras import backend
from keras.constraints import Constraint
from keras.initializers.initializers_v1 import RandomNormal
from keras.layers import Input, Dense, Concatenate, ReLU, Conv2D, Conv2DTranspose, Reshape, BatchNormalization, Flatten, \
    Activation, Embedding, LeakyReLU, Dropout
from keras.models import Model
from keras.optimizer_v1 import RMSprop
from tensorflow import Tensor, keras


def relu_bn(inputs: Tensor) -> Tensor:
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn


w_initializer = tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.02)


def wasserstein_loss(y_true, y_pred):
    return backend.mean(y_true * y_pred)


# clip model weights to a given hypercube
class ClipConstraint(Constraint):
    # set clip value when initialized
    def __init__(self, clip_value):
        self.clip_value = clip_value

    # clip model weights to hypercube
    def __call__(self, weights):
        return backend.clip(weights, -self.clip_value, self.clip_value)

    # get the config
    def get_config(self):
        return {'clip_value': self.clip_value}


# define the standalone generator model
def define_generator(latent_dim, n_classes):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # label input
    in_label = Input(shape=(1,))
    # embedding for categorical input
    li = Embedding(n_classes, 50)(in_label)
    # linear multiplication
    n_nodes = 7 * 7
    li = Dense(n_nodes)(li)
    # reshape to additional channel
    li = Reshape((7, 7, 1))(li)
    # image generator input
    in_lat = Input(shape=(latent_dim,))
    # foundation for 7x7 image
    n_nodes = 128 * 7 * 7
    gen = Dense(n_nodes)(in_lat)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Reshape((7, 7, 128))(gen)
    # merge image gen and label input
    merge = Concatenate()([gen, li])
    # upsample to 14x14
    gen = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(merge)
    gen = LeakyReLU(alpha=0.2)(gen)
    # upsample to 28x28
    gen = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    # output
    out_layer = Conv2D(1, (7, 7), activation='tanh', padding='same', kernel_initializer=init)(gen)
    # define model
    model = Model([in_lat, in_label], out_layer)
    return model

def define_discriminator(in_shape, n_classes):
    # label input
    filters = 128
    kernel = 3
    stride = 2
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # weight constraint
    const = ClipConstraint(0.01)
    in_label = Input(shape=(1,))
    # embedding for categorical input
    li = Embedding(n_classes, 50)(in_label)
    # scale up to image dimensions with linear activation
    n_nodes = in_shape[0] * in_shape[1]
    li = Dense(n_nodes)(li)
    # reshape to additional channel
    li = Reshape((in_shape[0], in_shape[1], 1))(li)
    # image input
    in_image = Input(shape=in_shape)
    # concat label as a channel
    merge = Concatenate()([in_image, li])
    # downsample
    fe = Conv2D(filters, (kernel, kernel), strides=(stride, stride), padding='same', kernel_constraint=const, kernel_initializer=init)(merge)
    fe = LeakyReLU(alpha=0.2)(fe)
    # downsample
    fe = Conv2D(filters, (kernel, kernel), strides=(stride, stride), padding='same', kernel_constraint=const, kernel_initializer=init)(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    # flatten feature maps
    fe = Flatten()(fe)
    # dropout
    fe = Dropout(0.4)(fe)
    # output
    out_layer = Dense(1, activation='linear')(fe)
    # define model
    model = Model([in_image, in_label], out_layer)
    # compile model
    opt = keras.optimizers.RMSprop(learning_rate=0.00005)
    model.compile(loss=wasserstein_loss, optimizer=opt)
    return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
    # d_model.trainable = False
    # get noise and label inputs from generator model
    gen_noise, gen_label = g_model.input
    # get image output from the generator model
    gen_output = g_model.output
    # connect image output and label input from generator as inputs to discriminator
    gan_output = d_model([gen_output, gen_label])
    # define gan model as taking noise and label and outputting a classification
    model = Model([gen_noise, gen_label], gan_output)
    # compile model
    opt = keras.optimizers.RMSprop(learning_rate=0.00005)
    model.compile(loss=wasserstein_loss, optimizer=opt)
    return model


# def gan(gen, dis):
#     # make weights in the discriminator not trainable
#     dis.trainable = False
#     model = Model(inputs=gen.input, outputs=dis(gen.output))
#     opt = keras.optimizers.RMSprop(learning_rate=0.00005)
#     model.compile(optimizer=opt, loss=wasserstein_loss, metrics=['accuracy'])
#     return model
