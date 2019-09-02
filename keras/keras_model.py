import numpy as np
import tensorflow as tf
from keras import optimizers, regularizers, callbacks, models

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPool1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.animation as animation
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib import pyplot as plt


TestSizePercent=0.1
s_indatapath='in_data.txt'
s_outdatapath='out_data.txt'

X = np.genfromtxt(s_indatapath)
Y = np.genfromtxt(s_outdatapath)
X = np.float32(X)
Y = np.float32(Y)

n_inputs = X.shape[1]
try:
    n_outputs = Y.shape[1]
except:
    n_outputs = 1
n_datasize = X.shape[0]
test_outputs = Y
# self.test_outputs.fill(0)

scaler = preprocessing.MinMaxScaler(feature_range=(0.0, 100.0))
X = scaler.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=TestSizePercent,
                                                                        shuffle=True)

train_size = x_train.shape[0]
test_size = x_test.shape[0]

x_train = np.reshape(x_train, [train_size, n_inputs,1])
y_train = np.reshape(y_train, [train_size, n_outputs])
x_test  = np.reshape(x_test, [test_size, n_inputs,1])
y_test  = np.reshape(y_test, [test_size, n_outputs])

X = np.reshape(X, [n_datasize, n_inputs,1])
Y = np.reshape(Y, [n_datasize, n_outputs])

#x_train=np.expand_dims(x_train, axis=2)
#y_train=np.expand_dims(y_train, axis=1)
#x_test=np.expand_dims(x_test, axis=2)
#y_test=np.expand_dims(y_test, axis=1)
# Generate dummy data
#x_train = np.random.random((1000, 20))
#y_train = np.random.randint(2, size=(1000, 1))
#x_test = np.random.random((100, 20))
#y_test = np.random.randint(2, size=(100, 1))


regL1=0.05
regL2=0.05


model = Sequential()
model.add(Conv1D(kernel_size = 10, filters = 20, activation='relu',input_shape=(n_inputs,1),padding="same",
                 kernel_initializer='glorot_normal',
                 bias_initializer='glorot_normal',
                 bias_regularizer=regularizers.l1_l2(l1=regL1, l2=regL2),
                 kernel_regularizer=regularizers.l1_l2(l1=regL1, l2=regL2),
                 activity_regularizer=regularizers.l1_l2(l1=regL1, l2=regL2)
                 ))
model.add(Conv1D(kernel_size = 5, filters = 30, activation='relu',padding="same",
                 kernel_initializer='glorot_normal',
                 bias_initializer='glorot_normal',
                 bias_regularizer=regularizers.l1_l2(l1=regL1, l2=regL2),
                 kernel_regularizer=regularizers.l1_l2(l1=regL1, l2=regL2),
                 activity_regularizer=regularizers.l1_l2(l1=regL1, l2=regL2)
                 ))
model.add(Conv1D(kernel_size = 3, filters = 30, activation='relu',padding="same",
                 kernel_initializer='glorot_normal',
                 bias_initializer='glorot_normal',
                 bias_regularizer=regularizers.l1_l2(l1=regL1, l2=regL2),
                 kernel_regularizer=regularizers.l1_l2(l1=regL1, l2=regL2),
                 activity_regularizer=regularizers.l1_l2(l1=regL1, l2=regL2)
                 ))
model.add(MaxPool1D(pool_size=(50)))#, strides=(1)))
model.add(Flatten())
model.add(Dense(50, activation='elu',
                kernel_initializer='glorot_normal',
                bias_initializer='glorot_normal',
                bias_regularizer=regularizers.l1_l2(l1=regL1, l2=regL2),
                kernel_regularizer=regularizers.l1_l2(l1=regL1, l2=regL2),
                activity_regularizer=regularizers.l1_l2(l1=regL1, l2=regL2)
                ))
model.add(Dense(25, activation='elu',
                kernel_initializer='glorot_normal',
                bias_initializer='glorot_normal',
                bias_regularizer=regularizers.l1_l2(l1=regL1, l2=regL2),
                kernel_regularizer=regularizers.l1_l2(l1=regL1, l2=regL2),
                activity_regularizer=regularizers.l1_l2(l1=regL1, l2=regL2)
                ))
model.add(Dense(n_outputs))

callbacks = [
  # Прерывает обучение если потери при проверке `val_loss` перестают
  # уменьшаться после 2 эпох
  callbacks.EarlyStopping(patience=50, monitor='val_loss'),
    #tf.keras.callbacks.LearningRateScheduler(schedule, verbose=0),
  # Записываем логи TensorBoard в папку `./logs`
  #tf.keras.callbacks.TensorBoard(log_dir='./logs')
    callbacks.ModelCheckpoint("my_model.h5", monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False,
                              mode='min', period=1)
]

opt=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False);
#opt=optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='mean_squared_error',
              #optimizer='rmsprop',
                optimizer=opt,
              metrics=['accuracy'])
file = open('my_model.txt', 'w')
file.write(str(model.summary()))
file.close()
structure=model.summary()
print(model.summary())
try:
    model = models.load_model('my_model.h5')
except:
    pass

model.fit(x_train, y_train,epochs=5000,batch_size=500,callbacks=callbacks,validation_data=(x_test, y_test))
score = model.evaluate(X, Y, batch_size=500)
inp=X[0]
inp = np.reshape(inp, [1, n_inputs,1])

#arr=np.reshape(x_test,(1,x_test.shape[0]) )
prediction = model.predict(x=X)#,batch_size=n_datasize)


#testingfig = plt.figure(figsize=(10, 7), dpi=80, num='Testing plot')
#testingplot = testingfig.add_subplot(111)
#testingplot.plot(prediction, linewidth=1.0, label="outputs", color='r')
#testingplot.plot(y_test, linewidth=1.0, label="targets", color='b')
#testingfig.show()
model.save('my_model.h5')
#models.save_model(filepath="my_model.h5")
plt.plot(prediction,linewidth=0.5)
plt.plot(Y,linewidth=0.5)
#plt.grid()
plt.show()
print(score)