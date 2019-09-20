from tkinter import *
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.animation as animation
from matplotlib import pyplot as plt
from sklearn import preprocessing
import threading
import os

from keras import optimizers, regularizers, callbacks, models, backend

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPool1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.animation as animation
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)

modelName = "model.h5"


Preprocessing_Min = 0.0
Preprocessing_Max = 1.0
TestSizePercent = 0.2
BatchMod = 0.05
MaxBatchSize = 300

layersNames = np.array(["conv1d", "dense", "max_pooling1d", "flatten"])
layersShortNames = np.array(["c1d", "d", "mp1d", "fl"])


def initModel():
    # model
    kernel_init = 'glorot_uniform'
    bias_init = 'zeros'
    kernel_reg = regularizers.l1_l2(l1=0.000, l2=0.000)
    bias_reg = regularizers.l1_l2(l1=0.000, l2=0.000)

    droprate=0.0
    learning_rate=0.00001
    kernel_size = 10
    filters = 5
    model = Sequential()

    model.add(Dense(50, activation='tanh',input_shape=(100,1),
                    kernel_initializer=kernel_init,
                    bias_initializer=bias_init,
                    bias_regularizer=bias_reg,
                    kernel_regularizer=kernel_reg,
                    # activity_regularizer=activity_reg
                    ))
    model.add(Flatten())
    model.add(Dropout(droprate))
    model.add(Dense(50, activation='tanh',
                    kernel_initializer=kernel_init,
                    bias_initializer=bias_init,
                    bias_regularizer=bias_reg,
                    kernel_regularizer=kernel_reg,
                    # activity_regularizer=activity_reg
                    ))
    model.add(Dropout(droprate))

    model.add(Dense(2,activation='softmax'))

    optimizer = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,
                                amsgrad=False);

    # optimizer = optimizers.Nadam(lr=self.settings['ls'], beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    model.compile(
        #loss='mean_squared_error',
         loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy'])
    print(model.summary())

    sName = ""
    for i in model.layers:
        for j in range(0, layersNames.size - 1):
            if (i.name.find(layersNames[j]) != -1):
                for k in i.input_shape:
                    if (k != None):
                        sName += str(k)
                        sName += "."
                    else:
                        sName += "_."
                sName += layersShortNames[j]
                sName += "."
    for i in model.layers[model.layers.__len__() - 1].output_shape:
        if (i != None):
            sName += str(i)
            sName += "."
        else:
            sName += "_."
    if (os.path.isfile(sName)):
        model.load_weights(sName)
    return model


model = initModel()

data0=np.random.random_sample(100,)
data0=np.reshape(data0,[1,100,1])
prediction_1=model.predict(data0)
target=np.array([[0,1]])
model.fit(x=data0,y=target,epochs=1000)
prediction_2=model.predict(data0)
print(prediction_1)
print(prediction_2)
