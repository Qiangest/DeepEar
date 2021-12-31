# -*- coding: utf-8 -*-
"""
DeepEar Demo Code for INFOCOM 2022

For reference only, the full version may vary.

@author: Qiang Yang
"""
# %% import

import numpy as np
import mat73
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler, StandardScaler, scale
from sklearn.model_selection import train_test_split
import itertools
from scipy.io import loadmat
import matplotlib.pyplot as plt
from time import time
from timeit import default_timer as timer

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Activation, Dense, LSTM, Subtract, Dropout, Lambda, GRU, RepeatVector, TimeDistributed, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import binary_crossentropy, RootMeanSquaredError
from tensorflow.keras.losses import mse
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.python.framework.ops import disable_eager_execution, enable_eager_execution

# enable_eager_execution()
disable_eager_execution()

# %% Check GPU

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# %% Functions


def sampling(distribution):
    distribution_mean, distribution_variance = distribution
    batchsize = tf.shape(distribution_variance)[0]
    random = K.random_normal(
        shape=(batchsize, tf.shape(distribution_variance)[1]),  mean=0., stddev=1.)
    return distribution_mean + tf.exp(0.5 * distribution_variance) * random

# %% Read data


# Read the mat data
matpath = './trainData.mat'
data = mat73.loadmat(matpath)
data = data['entry']
data = np.array(data)

x1, x2, x3, y = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

# Formulate input
x1 = np.array(x1.tolist())
x1 = x1.transpose(0, 2, 1)

x2 = np.array(x2.tolist())
x2 = x2.transpose(0, 2, 1)

x3 = np.array(x3.tolist())

y = np.array(y.tolist())
y = y.reshape(np.size(x1, 0), -1)

# Split data to train and test set
x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y_train, y_test = train_test_split(
    x1, x2, x3, y, test_size=0.2, random_state=2)

# Formulate labels
y1_train = y_train[:, 0]
y2_train = y_train[:, 1]
y3_train = y_train[:, 2:7]

y4_train = y_train[:, 7]
y5_train = y_train[:, 8]
y6_train = y_train[:, 9:14]

y7_train = y_train[:, 14]
y8_train = y_train[:, 15]
y9_train = y_train[:, 16:21]

y10_train = y_train[:, 21]
y11_train = y_train[:, 22]
y12_train = y_train[:, 23:28]

y13_train = y_train[:, 28]
y14_train = y_train[:, 29]
y15_train = y_train[:, 30:35]

y16_train = y_train[:, 35]
y17_train = y_train[:, 36]
y18_train = y_train[:, 37:42]

y19_train = y_train[:, 42]
y20_train = y_train[:, 43]
y21_train = y_train[:, 44:49]

y22_train = y_train[:, 49]
y23_train = y_train[:, 50]
y24_train = y_train[:, 51:56]

y1_test = y_test[:, 0:1]
y2_test = y_test[:, 1:2]
y3_test = y_test[:, 2:7]

y4_test = y_test[:, 7:8]
y5_test = y_test[:, 8:9]
y6_test = y_test[:, 9:14]

y7_test = y_test[:, 14:15]
y8_test = y_test[:, 15:16]
y9_test = y_test[:, 16:21]

y10_test = y_test[:, 21:22]
y11_test = y_test[:, 22:23]
y12_test = y_test[:, 23:28]

y13_test = y_test[:, 28:29]
y14_test = y_test[:, 29:30]
y15_test = y_test[:, 30:35]

y16_test = y_test[:, 35:36]
y17_test = y_test[:, 36:37]
y18_test = y_test[:, 37:42]

y19_test = y_test[:, 42:43]
y20_test = y_test[:, 43:44]
y21_test = y_test[:, 44:49]

y22_test = y_test[:, 49:50]
y23_test = y_test[:, 50:51]
y24_test = y_test[:, 51:56]

# %% Build model

data_dim = 100
timesteps = 19
latent_dim = 100


# Encoder can be trained independently
# encoder1
encoder1_input = Input(shape=(timesteps, data_dim))
encode1 = GRU(200, return_sequences=True, activation='relu')(encoder1_input)
encode1 = GRU(100, return_sequences=False, activation='relu')(encode1)
z_mean = Dense(latent_dim)(encode1)
z_log_variance = Dense(latent_dim)(encode1)
encoder1_output = Lambda(sampling)([z_mean, z_log_variance])
encoder1 = Model(encoder1_input, encoder1_output)

# encoder2
encoder2_input = Input(shape=(timesteps, data_dim))
encode2 = GRU(200, return_sequences=True, activation='relu')(encoder2_input)
encode2 = GRU(100, return_sequences=False, activation='relu')(encode2)
z_mean = Dense(latent_dim)(encode2)
z_log_variance = Dense(latent_dim)(encode2)
encoder2_ouput = Lambda(sampling)([z_mean, z_log_variance])
encoder2 = Model(encoder2_input, encoder2_ouput)

# Subtract
Subed = Subtract()([encoder1_output, encoder2_ouput])

# Cross-correlation
cc_input = Input(shape=(data_dim,))

# Concatenate features
features = Concatenate(
    axis=1)([cc_input, encoder1_output, encoder2_ouput, Subed])

body = Dense(512)(features)
body = Dropout(0.2)(body)
body = Dense(400)(body)
body = Dropout(0.2)(body)
body = Dense(200)(body)
body = Dropout(0.2)(body)

subNet1 = Dense(100, activation='relu')(body)
subNet1 = Dropout(0.2)(subNet1)
soundNet1 = Dense(50, activation='relu')(subNet1)
soundNet1 = Dense(10, activation='relu')(soundNet1)
soundNet1 = Dense(1, activation='sigmoid', name='soundNet1')(soundNet1)
AoANet1 = Dense(50, activation='relu')(subNet1)
AoANet1 = Dense(10, activation='relu')(AoANet1)
AoANet1 = Dense(1, activation='sigmoid', name='AoANet1')(AoANet1)
disNet1 = Dense(50, activation='relu')(subNet1)
disNet1 = Dense(10, activation='relu')(disNet1)
disNet1 = Dense(5, activation='softmax', name='disNet1')(disNet1)


subNet2 = Dense(100, activation='relu')(body)
subNet2 = Dropout(0.2)(subNet2)
soundNet2 = Dense(50, activation='relu')(subNet2)
soundNet2 = Dense(10, activation='relu')(soundNet2)
soundNet2 = Dense(1, activation='sigmoid', name='soundNet2')(soundNet2)
AoANet2 = Dense(50, activation='relu')(subNet2)
AoANet2 = Dense(10, activation='relu')(AoANet2)
AoANet2 = Dense(1, activation='sigmoid', name='AoANet2')(AoANet2)
disNet2 = Dense(50, activation='relu')(subNet2)
disNet2 = Dense(10, activation='relu')(disNet2)
disNet2 = Dense(5, activation='softmax', name='disNet2')(disNet2)


subNet3 = Dense(100, activation='relu')(body)
subNet3 = Dropout(0.2)(subNet3)
soundNet3 = Dense(50, activation='relu')(subNet3)
soundNet3 = Dense(10, activation='relu')(soundNet3)
soundNet3 = Dense(1, activation='sigmoid', name='soundNet3')(soundNet3)
AoANet3 = Dense(50, activation='relu')(subNet3)
AoANet3 = Dense(10, activation='relu')(AoANet3)
AoANet3 = Dense(1, activation='sigmoid', name='AoANet3')(AoANet3)
disNet3 = Dense(50, activation='relu')(subNet3)
disNet3 = Dense(10, activation='relu')(disNet3)
disNet3 = Dense(5, activation='softmax', name='disNet3')(disNet3)


subNet4 = Dense(100, activation='relu')(body)
subNet4 = Dropout(0.2)(subNet4)
soundNet4 = Dense(50, activation='relu')(subNet4)
soundNet4 = Dense(10, activation='relu')(soundNet4)
soundNet4 = Dense(1, activation='sigmoid', name='soundNet4')(soundNet4)
AoANet4 = Dense(50, activation='relu')(subNet4)
AoANet4 = Dense(10, activation='relu')(AoANet4)
AoANet4 = Dense(1, activation='sigmoid', name='AoANet4')(AoANet4)
disNet4 = Dense(50, activation='relu')(subNet4)
disNet4 = Dense(10, activation='relu')(disNet4)
disNet4 = Dense(5, activation='softmax', name='disNet4')(disNet4)

subNet5 = Dense(100, activation='relu')(body)
subNet5 = Dropout(0.2)(subNet5)
soundNet5 = Dense(50, activation='relu')(subNet5)
soundNet5 = Dense(10, activation='relu')(soundNet5)
soundNet5 = Dense(1, activation='sigmoid', name='soundNet5')(soundNet5)
AoANet5 = Dense(50, activation='relu')(subNet5)
AoANet5 = Dense(10, activation='relu')(AoANet5)
AoANet5 = Dense(1, activation='sigmoid', name='AoANet5')(AoANet5)
disNet5 = Dense(50, activation='relu')(subNet5)
disNet5 = Dense(10, activation='relu')(disNet5)
disNet5 = Dense(5, activation='softmax', name='disNet5')(disNet5)


subNet6 = Dense(100, activation='relu')(body)
subNet6 = Dropout(0.2)(subNet6)
soundNet6 = Dense(50, activation='relu')(subNet6)
soundNet6 = Dense(10, activation='relu')(soundNet6)
soundNet6 = Dense(1, activation='sigmoid', name='soundNet6')(soundNet6)
AoANet6 = Dense(50, activation='relu')(subNet6)
AoANet6 = Dense(10, activation='relu')(AoANet6)
AoANet6 = Dense(1, activation='sigmoid', name='AoANet6')(AoANet6)
disNet6 = Dense(50, activation='relu')(subNet6)
disNet6 = Dense(10, activation='relu')(disNet6)
disNet6 = Dense(5, activation='softmax', name='disNet6')(disNet6)

subNet7 = Dense(100, activation='relu')(body)
subNet7 = Dropout(0.2)(subNet7)
soundNet7 = Dense(50, activation='relu')(subNet7)
soundNet7 = Dense(10, activation='relu')(soundNet7)
soundNet7 = Dense(1, activation='sigmoid', name='soundNet7')(soundNet7)
AoANet7 = Dense(50, activation='relu')(subNet7)
AoANet7 = Dense(10, activation='relu')(AoANet7)
AoANet7 = Dense(1, activation='sigmoid', name='AoANet7')(AoANet7)
disNet7 = Dense(50, activation='relu')(subNet7)
disNet7 = Dense(10, activation='relu')(disNet7)
disNet7 = Dense(5, activation='softmax', name='disNet7')(disNet7)

subNet8 = Dense(100, activation='relu')(body)
subNet8 = Dropout(0.2)(subNet8)
soundNet8 = Dense(50, activation='relu')(subNet8)
soundNet8 = Dense(10, activation='relu')(soundNet8)
soundNet8 = Dense(1, activation='sigmoid', name='soundNet8')(soundNet8)
AoANet8 = Dense(50, activation='relu')(subNet8)
AoANet8 = Dense(10, activation='relu')(AoANet8)
AoANet8 = Dense(1, activation='sigmoid', name='AoANet8')(AoANet8)
disNet8 = Dense(50, activation='relu')(subNet8)
disNet8 = Dense(10, activation='relu')(disNet8)
disNet8 = Dense(5, activation='softmax', name='disNet8')(disNet8)


model = Model(inputs=[encoder1_input, encoder2_input, cc_input],
              outputs=[soundNet1, AoANet1, disNet1,
                       soundNet2, AoANet2, disNet2,
                       soundNet3, AoANet3, disNet3,
                       soundNet4, AoANet4, disNet4,
                       soundNet5, AoANet5, disNet5,
                       soundNet6, AoANet6, disNet6,
                       soundNet7, AoANet7, disNet7,
                       soundNet8, AoANet8, disNet8,
                       ])


# model.load_weights('DeepEar_weights.h5') # load the pre-trained model weights


# %% Compile model


model.compile(optimizer='Adam',
              loss=['binary_crossentropy', 'mean_squared_error', 'categorical_crossentropy',
                    'binary_crossentropy', 'mean_squared_error', 'categorical_crossentropy',
                    'binary_crossentropy', 'mean_squared_error', 'categorical_crossentropy',
                    'binary_crossentropy', 'mean_squared_error', 'categorical_crossentropy',
                    'binary_crossentropy', 'mean_squared_error', 'categorical_crossentropy',
                    'binary_crossentropy', 'mean_squared_error', 'categorical_crossentropy',
                    'binary_crossentropy', 'mean_squared_error', 'categorical_crossentropy',
                    'binary_crossentropy', 'mean_squared_error', 'categorical_crossentropy'],
              loss_weights=[0.4, 0.35, 0.25,
                            0.4, 0.35, 0.25,
                            0.4, 0.35, 0.25,
                            0.4, 0.35, 0.25,
                            0.4, 0.35, 0.25,
                            0.4, 0.35, 0.25,
                            0.4, 0.35, 0.25,
                            0.4, 0.35, 0.25],
              metrics=[['accuracy'], ['mae'], ['accuracy'],
                       ['accuracy'], ['mae'], ['accuracy'],
                       ['accuracy'], ['mae'], ['accuracy'],
                       ['accuracy'], ['mae'], ['accuracy'],
                       ['accuracy'], ['mae'], ['accuracy'],
                       ['accuracy'], ['mae'], ['accuracy'],
                       ['accuracy'], ['mae'], ['accuracy'],
                       ['accuracy'], ['mae'], ['accuracy']])

# %% Train the model

EarlyStopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=3, verbose=2)


history = model.fit([x1_train, x2_train, x3_train], [y1_train, y2_train, y3_train,
                                                     y4_train, y5_train, y6_train,
                                                     y7_train, y8_train, y9_train,
                                                     y10_train, y11_train, y12_train,
                                                     y13_train, y14_train, y15_train,
                                                     y16_train, y17_train, y18_train,
                                                     y19_train, y20_train, y21_train,
                                                     y22_train, y23_train, y24_train,
                                                     ],
                    callbacks=[EarlyStopping],
                    batch_size=200, epochs=50, verbose=1, shuffle=True, validation_split=0.1)

# %% Test

y_test_all = [y1_test, y2_test, y3_test, y4_test, y5_test, y6_test,
              y7_test, y8_test, y9_test, y10_test, y11_test, y12_test,
              y13_test, y14_test, y15_test, y16_test, y17_test, y18_test,
              y19_test, y20_test, y21_test, y22_test, y23_test, y24_test]

y_pred = model.predict([x1_test, x2_test, x3_test], batch_size=200)
