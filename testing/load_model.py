import math

from numpy import loadtxt
import tensorflow as tf
import keras
from keras.layers import Dense
from keras.models import Sequential, model_from_json

# load the dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
# split into input (X) and output (y) variables
dataset_end = len(dataset[:, 0])
dataset_mid = math.ceil(dataset_end/2)

data_train = dataset[0:dataset_mid, 0:8]
label_train = dataset[0:dataset_mid, 8]

data_test = dataset[dataset_mid+1:dataset_end, 0:8]
label_test = dataset[dataset_mid+1:dataset_end, 8]


# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
score = loaded_model.evaluate(data_test, label_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))
