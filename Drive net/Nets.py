import tensorflow as tf
from keras.layers import Input, Dense, Concatenate, ReLU, Conv2D, Conv2DTranspose, Reshape, BatchNormalization, Flatten, \
    Activation
from keras.models import Model
from tensorflow import Tensor, keras


def relu_bn(inputs: Tensor) -> Tensor:
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn


w_initializer = tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.02)


def generator(img_shape, noise_shape):
    filter = 64
    kernel = 4
    stride = 2
    l1_input = Input(shape=img_shape)

    # 64 filter
    l1_conv = Conv2D(kernel_size=kernel,
                     strides=stride,
                     filters=filter,
                     padding="same",
                     kernel_initializer=w_initializer)(l1_input)
    l1 = relu_bn(l1_conv)
    # 128 filter
    l2_conv = Conv2D(kernel_size=kernel,
                     strides=stride,
                     filters=2 * filter,
                     padding="same",
                     kernel_initializer=w_initializer)(l1)
    l2 = relu_bn(l2_conv)
    # 264 filter
    l3_conv = Conv2D(kernel_size=kernel,
                     strides=stride,
                     filters=4 * filter,
                     padding="same",
                     kernel_initializer=w_initializer)(l2)
    l3 = relu_bn(l3_conv)
    # 512 filter
    l4_conv = Conv2D(kernel_size=kernel,
                     strides=stride,
                     filters=8 * filter,
                     padding="same",
                     kernel_initializer=w_initializer)(l3)
    l4 = relu_bn(l4_conv)
    # 512 filter
    l5_conv = Conv2D(kernel_size=kernel,
                     strides=stride,
                     filters=8 * filter,
                     padding="same",
                     kernel_initializer=w_initializer)(l4)
    l5 = relu_bn(l5_conv)

    ######################## Noise ######################################

    l1_noise_input = Input(shape=noise_shape)

    # ruido = tf.keras.backend.random_normal

    l1_dense = Dense(4 * 4 * filter, kernel_initializer=w_initializer)(l1_noise_input)

    l1_reshape = Reshape((4, 4, filter))(l1_dense)

    # 128 filter
    l1_noise_conv = Conv2DTranspose(kernel_size=kernel,
                                    strides=stride,
                                    filters=2 * filter,
                                    padding="same",
                                    kernel_initializer=w_initializer)(l1_reshape)
    l1_noise = relu_bn(l1_noise_conv)

    noisy_img = Concatenate()([l5, l1_reshape])

    # 512 filter
    l7_conv = Conv2DTranspose(kernel_size=kernel,
                              strides=stride,
                              filters=8 * filter,
                              padding="same",
                              kernel_initializer=w_initializer)(noisy_img)
    l7 = relu_bn(l7_conv)

    l7_output = Concatenate(axis=-1)([l7, l4])

    # 512 filter
    l8_conv = Conv2DTranspose(kernel_size=kernel,
                              strides=stride,
                              filters=8 * filter,
                              padding="same",
                              kernel_initializer=w_initializer)(l7_output)
    l8 = relu_bn(l8_conv)

    l8_output = Concatenate(axis=-1)([l8, l3])

    # 256 filter
    l9_conv = Conv2DTranspose(kernel_size=kernel,
                              strides=stride,
                              filters=4 * filter,
                              padding="same",
                              kernel_initializer=w_initializer)(l8_output)
    l9 = relu_bn(l9_conv)

    l9_output = Concatenate(axis=-1)([l9, l2])

    # 128 filter
    l10_conv = Conv2DTranspose(kernel_size=kernel,
                               strides=stride,
                               filters=2 * filter,
                               padding="same",
                               kernel_initializer=w_initializer)(l9_output)
    l10 = relu_bn(l10_conv)

    l10_output = Concatenate(axis=-1)([l10, l1])

    # 64 filter
    l11_conv = Conv2DTranspose(kernel_size=kernel,
                               strides=stride,
                               filters=1,
                               padding="same",
                               kernel_initializer=w_initializer)(l10_output)

    gen_output = Activation('tanh')(l11_conv)

    model = Model(inputs=[l1_input, l1_noise_input], outputs=gen_output)
    return model


def discriminator(img_shape):
    filter = 32
    kernel = 4
    stride = 2
    input_layer = Input(shape=img_shape)

    # 32 filter
    l1_conv = Conv2D(kernel_size=kernel,
                     strides=stride,
                     filters=filter,
                     padding="same",
                     kernel_initializer=w_initializer)(input_layer)
    l1 = relu_bn(l1_conv)

    # 64 filter
    l2_conv = Conv2D(kernel_size=kernel,
                     strides=stride,
                     filters=2 * filter,
                     padding="same",
                     kernel_initializer=w_initializer)(l1)
    l2 = relu_bn(l2_conv)

    # 128 filter
    l3_conv = Conv2D(kernel_size=kernel,
                     strides=stride,
                     filters=4 * filter,
                     padding="same",
                     kernel_initializer=w_initializer)(l2)
    l3 = relu_bn(l3_conv)

    # 256 filter
    l4_conv = Conv2D(kernel_size=kernel,
                     strides=stride,
                     filters=8 * filter,
                     padding="same",
                     kernel_initializer=w_initializer)(l3)
    l4 = relu_bn(l4_conv)

    # 512 filter
    l5_conv = Conv2D(kernel_size=kernel,
                     strides=stride,
                     filters=16 * filter,
                     padding="same",
                     kernel_initializer=w_initializer)(l4)
    l5 = relu_bn(l5_conv)

    flatten_layer = Flatten()(l5)
    dis_output = Dense(1, activation='sigmoid', kernel_initializer=w_initializer)(flatten_layer)

    model = Model(inputs=input_layer, outputs=dis_output)
    opt = keras.optimizers.SGD(learning_rate=0.0001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def gan(gen, dis):
    # make weights in the discriminator not trainable
    dis.trainable = False
    model = Model(inputs=gen.input, outputs=dis(gen.output))
    opt = keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.5)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
