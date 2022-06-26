import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Concatenate, ReLU, Conv2D, Conv2DTranspose, Reshape, BatchNormalization, Flatten, \
    Activation
from keras.models import Model, Sequential
from matplotlib.pyplot import plot
from tensorflow import Tensor, keras


def relu_bn(inputs: Tensor) -> Tensor:
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn


w_initializer = tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.02)


class Generator:
    def __init__(self, img_shape, noise_shape):
        self.filter = 64
        self.kernel = 4
        self.stride = 2
        l1_input = Input(shape=img_shape)

        # 64 filter
        l1_conv = Conv2D(kernel_size=self.kernel,
                         strides=self.stride,
                         filters=self.filter,
                         padding="same",
                         kernel_initializer=w_initializer)(l1_input)
        l1 = relu_bn(l1_conv)
        # 128 filter
        l2_conv = Conv2D(kernel_size=self.kernel,
                         strides=self.stride,
                         filters=2 * self.filter,
                         padding="same",
                         kernel_initializer=w_initializer)(l1)
        l2 = relu_bn(l2_conv)
        # 264 filter
        l3_conv = Conv2D(kernel_size=self.kernel,
                         strides=self.stride,
                         filters=4 * self.filter,
                         padding="same",
                         kernel_initializer=w_initializer)(l2)
        l3 = relu_bn(l3_conv)
        # 512 filter
        l4_conv = Conv2D(kernel_size=self.kernel,
                         strides=self.stride,
                         filters=8 * self.filter,
                         padding="same",
                         kernel_initializer=w_initializer)(l3)
        l4 = relu_bn(l4_conv)
        # 512 filter
        l5_conv = Conv2D(kernel_size=self.kernel,
                         strides=self.stride,
                         filters=8 * self.filter,
                         padding="same",
                         kernel_initializer=w_initializer)(l4)
        l5 = relu_bn(l5_conv)
        # 512 filter
        l6_conv = Conv2D(kernel_size=self.kernel,
                         strides=self.stride,
                         filters=8 * self.filter,
                         padding="same",
                         kernel_initializer=w_initializer)(l5)
        l6 = relu_bn(l6_conv)

        ######################## Noise ######################################

        l1_noise_input = Input(shape=noise_shape)

        # ruido = tf.keras.backend.random_normal

        l1_dense = Dense(4 * 4 * self.filter, kernel_initializer=w_initializer)(l1_noise_input)
        print(l1_dense.shape)

        l1_reshape = Reshape((4, 4, self.filter))(l1_dense)
        print(l1_reshape.shape)

        # 128 filter
        l1_noise_conv = Conv2DTranspose(kernel_size=self.kernel,
                                        strides=self.stride,
                                        filters=2 * self.filter,
                                        padding="same",
                                        kernel_initializer=w_initializer)(l1_reshape)
        l1_noise = relu_bn(l1_noise_conv)
        print(l1_noise.shape)
        """# 256 filter
        l2_noise_conv = Conv2DTranspose(kernel_size=self.kernel,
                         strides=self.stride,
                         filters=4 * self.filter,
                         padding="same")(l1_noise)
        l2_noise = relu_bn(l2_noise_conv)"""

        noisy_img = Concatenate()([l6, l1_noise])

        # 512 filter
        l7_conv = Conv2DTranspose(kernel_size=self.kernel,
                                  strides=self.stride,
                                  filters=8 * self.filter,
                                  padding="same",
                                  kernel_initializer=w_initializer)(noisy_img)
        l7 = relu_bn(l7_conv)

        l7_output = Concatenate(axis=-1)([l7, l5])

        # 512 filter
        l8_conv = Conv2DTranspose(kernel_size=self.kernel,
                                  strides=self.stride,
                                  filters=8 * self.filter,
                                  padding="same",
                                  kernel_initializer=w_initializer)(l7_output)
        l8 = relu_bn(l8_conv)

        l8_output = Concatenate(axis=-1)([l8, l4])

        # 256 filter
        l9_conv = Conv2DTranspose(kernel_size=self.kernel,
                                  strides=self.stride,
                                  filters=4 * self.filter,
                                  padding="same",
                                  kernel_initializer=w_initializer)(l8_output)
        l9 = relu_bn(l9_conv)

        l9_output = Concatenate(axis=-1)([l9, l3])

        # 128 filter
        l10_conv = Conv2DTranspose(kernel_size=self.kernel,
                                   strides=self.stride,
                                   filters=2 * self.filter,
                                   padding="same",
                                   kernel_initializer=w_initializer)(l9_output)
        l10 = relu_bn(l10_conv)

        l10_output = Concatenate(axis=-1)([l10, l2])

        # 64 filter
        l11_conv = Conv2DTranspose(kernel_size=self.kernel,
                                   strides=self.stride,
                                   filters=self.filter,
                                   padding="same",
                                   kernel_initializer=w_initializer)(l10_output)
        l11 = relu_bn(l11_conv)

        l11_output = Concatenate(axis=-1)([l11, l1])

        l12_conv = Conv2DTranspose(kernel_size=self.kernel,
                                   strides=self.stride,
                                   filters=1,
                                   padding="same",
                                   kernel_initializer=w_initializer)(l11_output)

        gen_output = Activation('tanh')(l12_conv)

        self.model = Model(inputs=[l1_input, l1_noise_input], outputs=gen_output)
        # opt = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        # self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


class Discriminator:
    def __init__(self, img_shape):
        self.filter = 32
        self.kernel = 4
        self.stride = 2
        input_layer = Input(shape=img_shape)

        # 32 filter
        l1_conv = Conv2D(kernel_size=self.kernel,
                         strides=self.stride,
                         filters=self.filter,
                         padding="same",
                         kernel_initializer=w_initializer)(input_layer)
        l1 = relu_bn(l1_conv)
        print(l1.shape)

        # 64 filter
        l2_conv = Conv2D(kernel_size=self.kernel,
                         strides=self.stride,
                         filters=2 * self.filter,
                         padding="same",
                         kernel_initializer=w_initializer)(l1)
        l2 = relu_bn(l2_conv)
        print(l2.shape)

        # 128 filter
        l3_conv = Conv2D(kernel_size=self.kernel,
                         strides=self.stride,
                         filters=4 * self.filter,
                         padding="same",
                         kernel_initializer=w_initializer)(l2)
        l3 = relu_bn(l3_conv)
        print(l3.shape)

        # 256 filter
        l4_conv = Conv2D(kernel_size=self.kernel,
                         strides=self.stride,
                         filters=8 * self.filter,
                         padding="same",
                         kernel_initializer=w_initializer)(l3)
        l4 = relu_bn(l4_conv)
        print(l4.shape)

        # 512 filter
        l5_conv = Conv2D(kernel_size=self.kernel,
                         strides=self.stride,
                         filters=16 * self.filter,
                         padding="same",
                         kernel_initializer=w_initializer)(l4)
        l5 = relu_bn(l5_conv)
        print(l5.shape)

        flatten_layer = Flatten()(l5)
        print(flatten_layer.shape)
        dis_output = Dense(1, activation='sigmoid', kernel_initializer=w_initializer)(flatten_layer)

        self.model = Model(inputs=input_layer, outputs=dis_output)
        opt = keras.optimizers.SGD(learning_rate=0.0001)
        self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


class Gan:
    def __init__(self, gen, dis):
        # make weights in the discriminator not trainable
        dis.trainable = False
        self.model = Model(inputs=gen.input, outputs=dis(gen.output))
        opt = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
