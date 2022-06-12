import tensorflow as tf

from keras.layers import Input, Dense, Concatenate, ReLU
from keras.models import Model

from keras.layers import Dropout
from keras.layers import Flatten
from matplotlib import pyplot
from numpy import zeros, ones, expand_dims, vstack
from numpy.matlib import randn
from tensorflow import keras

class Generator():
    def __init__(self, input_img, input_noise):
        self.filter = 64
        self.kernel = 4
        self.stride = 2
        l1_input = Input(shape=input_img.shape)

        #64 filter
        l1_conv = Conv2D(kernel_size= self.kernel,
                   strides=self.stride,
                   filters=self.filter,
                   padding="same")(l1_input)
        l1 = relu_bn(l1_conv)
        # 128 filter
        l2_conv = Conv2D(kernel_size=self.kernel,
                         strides=self.stride,
                         filters=2 * self.filter,
                         padding="same")(l1)
        l2 = relu_bn(l2_conv)
        # 264 filter
        l3_conv = Conv2D(kernel_size=self.kernel,
                         strides=self.stride,
                         filters=4 * self.filter,
                         padding="same")(l2)
        l3 = relu_bn(l3_conv)
        # 512 filter
        l4_conv = Conv2D(kernel_size=self.kernel,
                         strides=self.stride,
                         filters=8 * self.filter,
                         padding="same")(l3)
        l4 = relu_bn(l4_conv)
        # 512 filter
        l5_conv = Conv2D(kernel_size=self.kernel,
                         strides=self.stride,
                         filters=8 * self.filter,
                         padding="same")(l4)
        l5 = relu_bn(l5_conv)
        # 512 filter
        l6_conv = Conv2D(kernel_size=self.kernel,
                         strides=self.stride,
                         filters=8 * self.filter,
                         padding="same")(l5)
        l6 = relu_bn(l6_conv)

        # Noise

        l1_noise_input = Input(shape=input_noise.shape)

        l1_dense = Dense(4 * 4 * n)(l1_noise_input)

        l1_reshape = Reshape((4, 4, n))(l1_dense)

        # 128 filter
        l1_noise_conv = Conv2DTranspose(kernel_size=self.kernel,
                         strides=self.stride,
                         filters=2 * self.filter,
                         padding="same")(l1_reshape)
        l1_noise = relu_bn(l1_noise_conv)
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
                               padding="same")(noisy_img)
        l7 = relu_bn(l7_conv)

        l7_output = Concatenate()([l7, l5])

        # 512 filter
        l8_conv = Conv2DTranspose(kernel_size=self.kernel,
                         strides=self.stride,
                         filters=8 * self.filter,
                         padding="same")(l7_output)
        l8 = relu_bn(l8_conv)

        l8_output = Concatenate()([l8, l4])

        # 256 filter
        l9_conv = Conv2DTranspose(kernel_size=self.kernel,
                         strides=self.stride,
                         filters=4 * self.filter,
                         padding="same")(l8_output)
        l9 = relu_bn(l9_conv)

        l9_output = Concatenate()([l9, l3])

        # 128 filter
        l10_conv = Conv2DTranspose(kernel_size=self.kernel,
                         strides=self.stride,
                         filters=2 * self.filter,
                         padding="same")(l9_output)
        l10 = relu_bn(l10_conv)

        l10_output = Concatenate()([l10, l2])

        # 64 filter
        l11_conv = Conv2DTranspose(kernel_size=self.kernel,
                         strides=self.stride,
                         filters=self.filter,
                         padding="same")(l10_output)
        l11 = relu_bn(l11_conv)

        l11_output = Concatenate()([l11, l1])

        gen_output = Activation('tanh')(l11_output)

        self.model = Model(inputs=[input_img,input_noise], outputs=gen_output)
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
#       model.fit(data, labels)



def relu_bn(inputs: Tensor) -> Tensor:
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn

class Discriminator():
    def __init__(self,in_shape):
        self.model = Sequential()
        self.filter = 32
        self.kernel = 4
        self.stride = 2
        # 32 filter,
        model.add(Conv2D(filter, (kernel, kernel), strides=(stride, stride), padding='same', input_shape=in_shape))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        # 64 filter
        model.add(Conv2D(2 * filter, (kernel, kernel), strides=(stride, stride), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        # 128 filter
        model.add(Conv2D(4 * filter, (kernel, kernel), strides=(stride, stride), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        # 256 filter
        model.add(Conv2D(8 * filter, (kernel, kernel), strides=(stride, stride), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        # 512 filter
        model.add(Conv2D(16 * filter, (kernel, kernel), strides=(stride, stride), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        # Output
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        opt = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt)


image = np.zeros((512, 512))
noise = np.zeros((400, 1))
gen = Generator(image, noise)
gen.summary()
dis = Discriminator(image)
dis.summary()






