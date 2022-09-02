import tensorflow as tf
from keras import backend
from keras.constraints import Constraint
from keras.initializers.initializers_v1 import RandomNormal
from keras.layers import Input, Dense, Concatenate, ReLU, Conv2D, Conv2DTranspose, Reshape, BatchNormalization, Flatten, \
    Activation, Embedding, LeakyReLU, Dropout
from keras.models import Model
from keras.optimizer_v1 import RMSprop
from tensorflow import Tensor, keras
import densenet121


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

def up_scaling_layer(x, n_filters):
    x = Conv2DTranspose(n_filters, (1, 1), strides=(0.5, 0.5), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout()(x)
    return x


def down_scaling_layer(x, n_filters):
    x = Conv2D(n_filters, (1, 1), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout()(x)
    return x


def resNet_layer(x, n_filters, scaling):
    x = Conv2D(n_filters, (1, 1), strides=(3, 3), padding='same')(x)
    x = Conv2D(n_filters, (1, 1), strides=(3, 3), padding='same')(x)
    x = Conv2D(n_filters, (1, 1), strides=(3, 3), padding='same')(x)
    x = Conv2D(n_filters, (1, 1), strides=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    if scaling == 'up':
        x = up_scaling_layer(x, n_filters)
    else:
        x = down_scaling_layer(x, n_filters)
    return x


def define_generator(latent_dim, input_shape, n_classes):
    n_filters = 3
    input_layer = Input(shape=input_shape)

    x = resNet_layer(input_layer, n_filters, 'down')
    x = resNet_layer(input_layer, n_filters, 'down')
    x = resNet_layer(input_layer, n_filters, 'down')
    x = resNet_layer(input_layer, n_filters, 'down')
    x = resNet_layer(input_layer, n_filters, 'up')
    x = resNet_layer(input_layer, n_filters, 'up')
    x = resNet_layer(input_layer, n_filters, 'up')
    x = resNet_layer(input_layer, n_filters, 'up')

    model = Model([in_lat, in_label], out_layer)
    return model

def define_discriminator(in_shape, n_classes):
    model = densenet121.DenseNet(reduction=0.5, classes=7)

    sgd = keras.optimizers.SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
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
