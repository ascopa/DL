import numpy as np
from keras.utils.version_utils import callbacks
from matplotlib import pyplot as plt
from tensorflow import keras
from keras import layers

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# the data, split between train and test sets
(data_train, label_train), (data_test, label_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
data_train = data_train.astype("float32") / 255
data_test = data_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
data_train = np.expand_dims(data_train, -1) #reshape con -1 al final
data_test = np.expand_dims(data_test, -1)
print("x_train shape:", data_train.shape)
print(data_train.shape[0], "train samples")
print(data_test.shape[0], "test samples")


# convert class vectors to binary class matrices
label_train = keras.utils.to_categorical(label_train, num_classes)
label_test = keras.utils.to_categorical(label_test, num_classes)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

batch_size = 128
epochs = 200

opt = keras.optimizers.Adam(learning_rate=0.0005)

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

early_stopping = callbacks.EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=25, # how many epochs to wait before stopping
    restore_best_weights=True,
)
history_seq = model.fit(data_train, label_train, batch_size=batch_size, epochs=epochs,callbacks=[early_stopping], validation_split=0.1)

score = model.evaluate(data_test, label_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])


# summarize history for accuracy
plt.plot(history_seq.history['accuracy'])
plt.plot(history_seq.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history_seq.history['loss'])
plt.plot(history_seq.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

