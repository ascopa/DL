import math

import matplotlib.pyplot as plt
import tensorflow
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.regularizers import l2
from keras.utils.version_utils import callbacks
from keras.utils.vis_utils import plot_model
from numpy import loadtxt
from tensorflow import keras

plt.close('all')

print("Num GPUs Available: ", len(tensorflow.config.list_physical_devices('GPU')))

# load the dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
# split into input (X) and output (y) variables
dataset_end = len(dataset[:, 0])
dataset_mid = math.ceil(dataset_end/2)

data_train = dataset[0:dataset_mid, 0:8]
label_train = dataset[0:dataset_mid, 8]

data_test = dataset[dataset_mid+1:dataset_end, 0:8]
label_test = dataset[dataset_mid+1:dataset_end, 8]

# define the keras model using Sequential model API

initializer = keras.initializers.HeNormal()

model = Sequential()
model.add(Dense(64, input_dim=8, activation='relu',kernel_initializer=initializer))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.05)))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())
plot_model(model, to_file='model_plot_seq.png', show_shapes=True, show_layer_names=True)

# define the keras model using Keras functional API
# visible = Input(shape=(8,))
# hidden1 = Dense(12, activation='relu')(visible)
# hidden2 = Dense(8, activation='relu')(hidden1)
# output = Dense(1, activation='sigmoid')(hidden2)
# model2 = Model(inputs=visible, outputs=output)
# print(model2.summary())
# plot_model(model2, to_file='model_plot_func.png', show_shapes=True, show_layer_names=True)

early_stopping = callbacks.EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=25, # how many epochs to wait before stopping
    restore_best_weights=True,
)

# compile the keras model
# lr_schedule = keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=1e-2,
#     decay_steps=1000,
#     decay_rate=0.9)

opt = keras.optimizers.Adam(learning_rate=0.001)

metric = keras.metrics.BinaryAccuracy(
    name="binary_accuracy", dtype=None, threshold=0.5
)

model.compile(loss='binary_crossentropy', optimizer=opt, metrics=metric)

# fit the keras model on the dataset
history_seq = model.fit(data_train, label_train, validation_split=0.2, epochs=500, batch_size=10, callbacks=[early_stopping], verbose=1)

# make class predictions with the model
predictions = (model.predict(data_test) > 0.5).astype(int)
_, accuracy = model.evaluate(data_test, label_test)

print('Accuracy: %.2f' % (accuracy * 100))
# summarize the first 5 cases
for i in range(5):
    print('%s => %d (expected %d)' % (data_test[i].tolist(), predictions[i], label_test[i]))

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")


# list all data in history
print(history_seq.history.keys())
# summarize history for accuracy
plt.plot(history_seq.history['binary_accuracy'])
plt.plot(history_seq.history['val_binary_accuracy'])
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


