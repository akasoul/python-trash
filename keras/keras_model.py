import numpy as np
import tensorflow as tf
from keras import optimizers
from keras import regularizers
from keras.models import Sequential
#from keras import layers
from keras.layers import Dense, Dropout, Conv1D, MaxPool1D
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
#x = np.reshape(X, [n_datasize, n_inputs])
#y = np.reshape(Y, [n_datasize, n_outputs])
x = np.reshape(X, [n_datasize, n_inputs, 1])
y = np.reshape(Y, [n_datasize, n_outputs,1])
test_outputs = Y
# self.test_outputs.fill(0)

scaler = preprocessing.MinMaxScaler(feature_range=(-1.0, 1.0))
X = scaler.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=TestSizePercent,
                                                                        shuffle=True)

train_size = x_train.shape[0]
test_size = x_test.shape[0]

x_train = np.reshape(x_train, [train_size, n_inputs, 1])
y_train = np.reshape(y_train, [train_size, n_outputs,1])
x_test = np.reshape(x_test, [test_size, n_inputs, 1])
y_test = np.reshape(y_test, [test_size, n_outputs,1])

# Generate dummy data
#x_train = np.random.random((1000, 20))
#y_train = np.random.randint(2, size=(1000, 1))
#x_test = np.random.random((100, 20))
#y_test = np.random.randint(2, size=(100, 1))

model = Sequential()
model.add(Conv1D(kernel_size = 50, filters = 20, activation='relu',input_shape=(n_inputs,1)))
#model.add(Dense(256, input_dim=n_inputs, activation='elu',
#                kernel_initializer='glorot_normal',
#                bias_initializer='glorot_normal',
#                activity_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
model.add(MaxPool1D(pool_size=(20), strides=(10)))
model.add(Dropout(0.5))
model.add(Dense(256, activation='elu',
                kernel_initializer='glorot_normal',
                bias_initializer='glorot_normal',
                activity_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
model.add(Dropout(0.5))
model.add(Dense(256, activation='elu',
                kernel_initializer='glorot_normal',
                bias_initializer='glorot_normal',
                activity_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
model.add(Dropout(0.5))
model.add(Dense(256, activation='elu',
                kernel_initializer='glorot_normal',
                bias_initializer='glorot_normal',
                activity_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
model.add(Dropout(0.5))
model.add(Dense(256, activation='elu',
                kernel_initializer='glorot_normal',
                bias_initializer='glorot_normal',
                activity_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
model.add(Dropout(0.5))
model.add(Dense(n_outputs))

callbacks = [
  # Прерывает обучение если потери при проверке `val_loss` перестают
  # уменьшаться после 2 эпох
  tf.keras.callbacks.EarlyStopping(patience=20, monitor='val_loss'),
    #tf.keras.callbacks.LearningRateScheduler(schedule, verbose=0),
  # Записываем логи TensorBoard в папку `./logs`
  #tf.keras.callbacks.TensorBoard(log_dir='./logs')
]

opt=optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False);
#opt=optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='mean_squared_error',
              #optimizer='rmsprop',
                optimizer=opt,
              metrics=['accuracy'])
#print(model.summary())

model.fit(x_train, y_train,epochs=1000,batch_size=128,callbacks=callbacks,validation_data=(x_test, y_test))
score = model.evaluate(x, y, batch_size=128)
#arr=np.reshape(x_test,(1,x_test.shape[0]) )
prediction = model.predict(x=X,batch_size=n_datasize)


#testingfig = plt.figure(figsize=(10, 7), dpi=80, num='Testing plot')
#testingplot = testingfig.add_subplot(111)
#testingplot.plot(prediction, linewidth=1.0, label="outputs", color='r')
#testingplot.plot(y_test, linewidth=1.0, label="targets", color='b')
#testingfig.show()

plt.plot(prediction,linewidth=0.5)
plt.plot(y,linewidth=0.5)
#plt.grid()
plt.show()
print(score)