import tensorflow as tf
from keras import backend
from keras.constraints import Constraint
from keras.layers import Input, Dense, Concatenate, ReLU, Conv2D, Conv2DTranspose, Reshape, BatchNormalization, \
    Activation, Embedding, LeakyReLU, Dropout
from keras.models import Model
from tensorflow import Tensor, keras

import densenet121


def relu_bn(inputs: Tensor) -> Tensor:
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn


drop_out_rate = 0.5
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
    kernel = 1
    stride = 2
    x = Conv2DTranspose(n_filters, kernel_size=kernel, strides=stride, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(drop_out_rate)(x)
    return x


def down_scaling_layer(x, n_filters):
    kernel = 1
    stride = 2
    x = Conv2D(n_filters, kernel_size=kernel, strides=stride, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(drop_out_rate)(x)
    return x


def resNet_block(x, n_filters, scaling):
    kernel = 3
    stride = 1
    x = Conv2D(n_filters, kernel_size=kernel, strides=stride, padding='same')(x)
    x = Conv2D(n_filters, kernel_size=kernel, strides=stride, padding='same')(x)
    x = Conv2D(n_filters, kernel_size=kernel, strides=stride, padding='same')(x)
    x = Conv2D(n_filters, kernel_size=kernel, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    if scaling == 'up':
        x = up_scaling_layer(x, n_filters)
    else:
        x = down_scaling_layer(x, n_filters)
    print("x shape: ", x.shape)
    return x


def representation_layer(layer, noise_input):
    filter = 3
    stride = 2
    kernel = 4

    dense = Dense(7 * 7 * filter, kernel_initializer=w_initializer)(noise_input)

    reshape = Reshape((7, 7, filter))(dense)

    noise_conv = Conv2DTranspose(kernel_size=kernel,
                                 strides=stride,
                                 filters=2 * filter,
                                 padding="same",
                                 kernel_initializer=w_initializer)(reshape)
    concat_layer = Concatenate()([layer, noise_conv])
    print("concat_layer shape: ", concat_layer.shape)
    return concat_layer


def img_n_label_layer(input_shape, n_classes):
    # label input
    label_input_layer = Input(shape=(1,))
    # embedding for categorical input
    li = Embedding(n_classes, 50)(label_input_layer)
    # scale up to image dimensions with linear activation
    n_nodes = input_shape[0] * input_shape[1]
    li = Dense(n_nodes)(li)
    # reshape to additional channel
    li = Reshape((input_shape[0], input_shape[1], 1))(li)
    # image input
    img_input_layer = Input(shape=input_shape)
    # concat label as a channel
    img_n_label_input_layer = Concatenate()([img_input_layer, li])
    return img_input_layer, img_n_label_input_layer, label_input_layer


def define_generator(latent_dim, input_shape, n_classes):
    n_filters = 3

    img_input_layer, img_n_label_input_layer, label_input_layer = img_n_label_layer(input_shape, n_classes)

    x1 = resNet_block(img_n_label_input_layer, n_filters, 'down')
    x2 = resNet_block(x1, 2 * n_filters, 'down')
    x3 = resNet_block(x2, 4 * n_filters, 'down')
    x4 = resNet_block(x3, 8 * n_filters, 'down')

    noise_input_layer = Input(shape=latent_dim)
    decoded_img_and_noise = representation_layer(x4, noise_input_layer)

    x5 = resNet_block(decoded_img_and_noise, 8 * n_filters, 'up')
    skip_connection_3_5 = Concatenate(axis=-1)([x3, x5])
    x6 = resNet_block(skip_connection_3_5, 4 * n_filters, 'up')
    skip_connection_2_6 = Concatenate(axis=-1)([x2, x6])
    x7 = resNet_block(skip_connection_2_6, 2 * n_filters, 'up')
    skip_connection_1_7 = Concatenate(axis=-1)([x1, x7])
    x8 = resNet_block(skip_connection_1_7, 1, 'up')

    gen_output = Activation('tanh')(x8)

    model = Model(inputs=[img_input_layer, label_input_layer, noise_input_layer], outputs=gen_output)
    return model


def define_discriminator(input_shape, n_classes):
    img_input_layer, img_n_label_input_layer, label_input_layer = img_n_label_layer(input_shape, n_classes)

    model = densenet121.DenseNet(img_input_layer, img_n_label_input_layer, label_input_layer, reduction=0.5, classes=1)

    sgd = keras.optimizers.SGD(learning_rate=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
    # d_model.trainable = False
    # get img, label and noise inputs from generator model
    gen_img, gen_label, gen_noise = g_model.input
    # get image output from the generator model
    gen_output = g_model.output
    # connect image output and label input from generator as inputs to discriminator
    gan_output = d_model([gen_output, gen_label])
    # define gan model as taking img, noise and label and outputting a classification
    model = Model([gen_img, gen_label, gen_noise], gan_output)
    # compile model
    opt = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0, beta_2=0.9)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

# def gan(gen, dis):
#     # make weights in the discriminator not trainable
#     dis.trainable = False
#     model = Model(inputs=gen.input, outputs=dis(gen.output))
#     opt = keras.optimizers.RMSprop(learning_rate=0.00005)
#     model.compile(optimizer=opt, loss=wasserstein_loss, metrics=['accuracy'])
#     return model
