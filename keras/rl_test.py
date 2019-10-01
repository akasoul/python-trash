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

def readFile(fileName):
        completed=False
        while(completed==False):
            while not os.path.isfile(fileName):
                pass
            while not completed:
                try:
                    data=np.genfromtxt(fileName)
                except:
                    pass
                else:
                    completed=True
        return data

def writeFile(fileName,data):
    output = ""
    for i in data:
        output += str(i)
        output += " "
    file = open(fileName, 'w')
    file.write(output)
    file.close()

def initModel(inputSize):
    # model
    kernel_init = 'glorot_uniform'
    bias_init = 'zeros'
    kernel_reg = regularizers.l1_l2(l1=0.01, l2=0.01)
    bias_reg = regularizers.l1_l2(l1=0.01, l2=0.01)

    droprate=0.0
    learning_rate=0.001
    kernel_size = 10
    filters = 5

    kernel_size = 10
    filters = 5
    model = Sequential()
    model.add(
        Conv1D(kernel_size=kernel_size, filters=filters, activation='relu', input_shape=(inputSize, 1),
               padding="same",
               kernel_initializer=kernel_init,
               bias_initializer=bias_init,
               bias_regularizer=bias_reg,
               kernel_regularizer=kernel_reg,
               # activity_regularizer=activity_reg
               ))
    model.add(MaxPool1D(pool_size=(3)))  # , strides=(1)))
    model.add(Conv1D(kernel_size=kernel_size, filters=filters, activation='relu', padding="same",
                     kernel_initializer=kernel_init,
                     bias_initializer=bias_init,
                     bias_regularizer=bias_reg,
                     kernel_regularizer=kernel_reg,
                     # activity_regularizer=activity_reg
                     ))
    model.add(MaxPool1D(pool_size=(3)))  # , strides=(1)))
    model.add(Conv1D(kernel_size=kernel_size, filters=filters, activation='relu', padding="same",
                     kernel_initializer=kernel_init,
                     bias_initializer=bias_init,
                     bias_regularizer=bias_reg,
                     kernel_regularizer=kernel_reg,
                     # activity_regularizer=activity_reg
                     ))
    model.add(MaxPool1D(pool_size=(3)))  # , strides=(1)))
    model.add(Flatten())
    #model.add(Dropout(droprate))
    model.add(Dense(50, activation='relu',
                    kernel_initializer=kernel_init,
                    bias_initializer=bias_init,
                    bias_regularizer=bias_reg,
                    kernel_regularizer=kernel_reg,
                    # activity_regularizer=activity_reg
                    ))
    #model.add(Dropout(droprate))
    model.add(Dense(50, activation='relu',
                    kernel_initializer=kernel_init,
                    bias_initializer=bias_init,
                    bias_regularizer=bias_reg,
                    kernel_regularizer=kernel_reg,
                    # activity_regularizer=activity_reg
                    ))
    #model.add(Dropout(droprate))
    model.add(Dense(50, activation='relu',
                    kernel_initializer=kernel_init,
                    bias_initializer=bias_init,
                    bias_regularizer=bias_reg,
                    kernel_regularizer=kernel_reg,
                    # activity_regularizer=activity_reg
                    ))
    #model.add(Dropout(droprate))

    model.add(Dense(4,activation='softmax',
              bias_initializer='glorot_uniform'))

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


model = None
stateFname="state.txt"
rewardFname="reward.txt"
actionFname="action.txt"

states = None
actions = None
rewards = None

examples=0

while(True):

    state=None
    action=None
    reward=None

    state=readFile(stateFname)

    shape=state.shape[0]

    if(model==None):
        model=initModel(shape)
        if(os.path.isfile("model.h5")):
            try:
                model.load_weights("model.h5")
            except:
                pass

    state=np.reshape(state,[1,shape,1])

    action=model.predict(state)

    writeFile(actionFname,action[0])

    for i in range(0,15):
        fnameState="state"+str(i)+".txt"
        fnameReward="reward"+str(i)+".txt"
        if(os.path.isfile(fnameState)):
            if(os.path.isfile(rewardFname)):
                examples+=1
                state=readFile(fnameState)
                reward=readFile(rewardFname)
                states = np.append([states], [state])
                states = np.reshape(states, [examples,shape,1])
                rewards = np.append([rewards], [reward])
                rewards = np.reshape(rewards, [examples, 2])

    if examples>0:
        #test_model=model.predict(states)
        model.fit(states, rewards,steps_per_epoch=1, epochs=1)
        model.save_weights("model.h5")
        print(rewards.shape)

