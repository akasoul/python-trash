from tkinter import *
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.animation as animation
from matplotlib import pyplot as plt
from sklearn import preprocessing
import threading
import os

from keras import optimizers, regularizers, callbacks, models

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPool1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.animation as animation
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)

modelName = "model.h5"

# defines
elu = tf.nn.elu
sig = tf.nn.sigmoid
tan = tf.nn.tanh
relu = tf.nn.relu
softsign = tf.nn.softsign

##nn structure
# f_l_f = elu
# neurons=100
# struct = np.array([[neurons,neurons,neurons,neurons,neurons,neurons,neurons,neurons],
#                   [f_l_f,f_l_f,f_l_f,f_l_f,f_l_f,f_l_f,f_l_f,f_l_f]])
# outputs_af = tan


# nn structure
f_l_f = elu
neurons = 1024
struct = np.array([[neurons, neurons, neurons],
                   [f_l_f, f_l_f, f_l_f]])
outputs_af = None

Preprocessing_Min = -1.0
Preprocessing_Max = 1.0
TestSizePercent = 0.1
BatchMod = 0.2
MaxBatchSize = 300


class historyCallback(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        try:
            self.loss = np.append(self.loss, logs.get('loss'))
        except:
            self.loss = np.array([logs.get('loss')], dtype=float)

            try:
                self.val_loss = np.append(self.val_loss, logs.get('val_loss'))
            except:
                self.val_loss = np.array([logs.get('val_loss')], dtype=float)


class app:

    def tryLoadData(self):
        self.data_is_loading = True;
        self.path_is_valid = False
        fpath = self.ed_indatapath.get(1.0, END)
        fpath = fpath.rstrip()
        fpath = fpath.lstrip()
        self.sDataInputPath = fpath

        fpath = self.ed_outdatapath.get(1.0, END)
        fpath = fpath.rstrip()
        fpath = fpath.lstrip()
        self.sDataOutputPath = fpath

        # while(self.path_is_valid==False):
        if (os.path.isfile(self.sDataInputPath)):
            self.ed_indatapath['bg'] = 'green'
            if (os.path.isfile(self.sDataOutputPath)):
                self.ed_outdatapath['bg'] = 'green'
                self.path_is_valid = True
                self.loadData()
                # init arrays for training

                try:
                    self.trainingdata_train_error = np.empty(shape=0)
                except:
                    self.trainingdata_train_error.reshape(0)
                else:
                    pass

                try:
                    self.trainingdata_test_error = np.empty(shape=0)
                except:
                    self.trainingdata_test_error.reshape(0)
                else:
                    pass
            else:
                self.ed_outdatapath['bg'] = 'red'

        else:
            self.ed_indatapath['bg'] = 'red'
            if (os.path.isfile(self.sDataOutputPath)):
                self.ed_outdatapath['bg'] = 'green'
            else:
                self.ed_outdatapath['bg'] = 'red'
            print('no data')
            self.data_is_loading = False
            return False

        self.data_is_loading = False

    def initModel(self):
        # create a placeholder to dynamically switch between batch sizes
        self.phBatchSize = tf.placeholder(tf.int64)
        self.phDropRate = tf.placeholder(tf.float32)
        self.phL1 = tf.placeholder(tf.float32)
        self.phL2 = tf.placeholder(tf.float32)
        self.phLearningRate = tf.placeholder(tf.float32)
        self.batch_normalization_active = tf.placeholder(tf.bool)

        # model
        self.model = Sequential()
        self.model.add(
            Conv1D(kernel_size=10, filters=20, activation='relu', input_shape=(self.nInputs, 1), padding="same",
                   kernel_initializer='glorot_normal',
                   bias_initializer='glorot_normal',
                   bias_regularizer=regularizers.l1_l2(l1=self.phL1, l2=self.phL2),
                   kernel_regularizer=regularizers.l1_l2(l1=self.phL1, l2=self.phL2),
                   activity_regularizer=regularizers.l1_l2(l1=self.phL1, l2=self.phL2)
                   ))
        self.model.add(Conv1D(kernel_size=5, filters=30, activation='relu', padding="same",
                              kernel_initializer='glorot_normal',
                              bias_initializer='glorot_normal',
                              bias_regularizer=regularizers.l1_l2(l1=self.phL1, l2=self.phL2),
                              kernel_regularizer=regularizers.l1_l2(l1=self.phL1, l2=self.phL2),
                              activity_regularizer=regularizers.l1_l2(l1=self.phL1, l2=self.phL2)
                              ))
        self.model.add(Conv1D(kernel_size=3, filters=30, activation='relu', padding="same",
                              kernel_initializer='glorot_normal',
                              bias_initializer='glorot_normal',
                              bias_regularizer=regularizers.l1_l2(l1=self.phL1, l2=self.phL2),
                              kernel_regularizer=regularizers.l1_l2(l1=self.phL1, l2=self.phL2),
                              activity_regularizer=regularizers.l1_l2(l1=self.phL1, l2=self.phL2)
                              ))
        self.model.add(MaxPool1D(pool_size=(50)))  # , strides=(1)))
        self.model.add(Flatten())
        self.model.add(Dense(50, activation='elu',
                             kernel_initializer='glorot_normal',
                             bias_initializer='glorot_normal',
                             bias_regularizer=regularizers.l1_l2(l1=self.phL1, l2=self.phL2),
                             kernel_regularizer=regularizers.l1_l2(l1=self.phL1, l2=self.phL2),
                             activity_regularizer=regularizers.l1_l2(l1=self.phL1, l2=self.phL2)
                             ))
        self.model.add(Dropout(self.phDropRate))
        self.model.add(Dense(25, activation='elu',
                             kernel_initializer='glorot_normal',
                             bias_initializer='glorot_normal',
                             bias_regularizer=regularizers.l1_l2(l1=self.phL1, l2=self.phL2),
                             kernel_regularizer=regularizers.l1_l2(l1=self.phL1, l2=self.phL2),
                             activity_regularizer=regularizers.l1_l2(l1=self.phL1, l2=self.phL2)
                             ))
        self.model.add(Dropout(self.phDropRate))
        self.model.add(Dense(self.nOutputs))

        self.optimizer = optimizers.Adam(lr=self.phLearningRate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,
                                         amsgrad=False);

        self.model.compile(loss='mean_squared_error',
                           optimizer=self.optimizer,
                           metrics=['accuracy'])
        if (os.path.isfile(modelName)):
            self.loadModel()

    def initModelName(self):
        self.model_name = modelName

    def loadModel(self):
        try:
            self.model = models.load_model('my_model.h5')
        except:
            print('no model')

    def setUiBlocking(self, type, state):
        if type == 'run':
            ui = [
                self.btnTrain,
            ]
        if type == 'train':
            ui = [
                self.btnShowTestingPlot,
                self.ed_indatapath,
                self.ed_outdatapath,
                self.ed_inputpath,
                self.btnRun,
                self.ed_trs_ls,
                self.ed_trs_l1,
                self.ed_trs_l2,
                self.ed_trs_droprate
            ]
        for i in ui:
            if (state == 'block'):
                i['state'] = 'disabled'
                i['bg'] = 'silver'
            if (state == 'unblock'):
                i['state'] = 'enabled'
                i['bg'] = 'white'

    def onClickRunBtn(self, event):
        if self.run_is_launched == False:
            tt = threading.Thread(target=self.theadRun)
            tt.daemon = True
            tt.start()
        else:
            self.stop_run_is_pressed = True

    def onClickTrainBtn(self, event):
        if self.training_is_launched == False:
            tt = threading.Thread(target=self.threadTrain)
            tt.daemon = True
            tt.start()
        else:
            self.stop_train_is_pressed = True

    def onClickReloadDataBtn(self, event):
        tt = threading.Thread(target=self.threadReloadData)
        tt.daemon = True
        tt.start()

    def onClickShowTrainingPlot(self, event):
        self.trainingfig = plt.figure(figsize=(10, 7), dpi=80, num='Training plot')
        self.trainingplot = self.trainingfig.add_subplot(111)
        self.trainingani = animation.FuncAnimation(self.trainingfig, self.threadDrawTrainingPlot, interval=1000)
        self.trainingfignum = self.trainingfig.number

        self.trainingfig.show()
        # if(plt.fignum_exists(self.trainingfignum)):
        #    pass
        # else:

    def onClickShowTestingPlot(self, event):
        prediction = self.model.predict(x=self.X)  # ,batch_size=n_datasize)
        plt.plot(prediction, linewidth=0.5)
        plt.plot(self.Y, linewidth=0.5)
        plt.show()

    def onChangePath(self, event):
        if self.data_is_loaded == False:
            self.tryLoadData()

    def onChangeSettings(self, event, ui_index, format):
        i = ui_index
        value = self.settingsUI[i].get(1.0, END)
        value = value.rstrip()
        try:
            format(value)
        except:
            self.settingsUI[i]['bg'] = 'red'
        else:
            self.settingsUI[i]['bg'] = 'white'
            self.settings[i] = format(value)
            if self.saving_is_launched == False:
                ts = threading.Thread(target=self.threadSaveSettings)
                ts.daemon = True
                ts.start()

    def threadSaveSettings(self):
        # while(self.training_is_launched):
        # pass
        # while(self.run_is_launched):
        # pass
        self.saving_is_launched = True
        self.saveSettings()
        self.saving_is_launched = False

    def threadTrain(self):
        self.training_is_launched = True
        self.stop_train_is_pressed = False
        self.btnTrain.config(text="Stop training")

        self.callbacks = [
            callbacks.EarlyStopping(patience=50, monitor='val_loss'),
            callbacks.ModelCheckpoint("my_model.h5", monitor='val_loss', verbose=0, save_best_only=False,
                                      save_weights_only=False,
                                      mode='min', period=1),
            self.historyCallback
        ]

        self.model.fit(self.X_train, self.Y_train, epochs=5, batch_size=500, callbacks=callbacks, validation_data=(self.X_test, self.Y_test))



        self.updateError()
        self.training_is_launched = False
        self.btnTrain.config(text="Train model")
        return

    def theadRun(self):
        self.run_is_launched = True
        self.stop_run_is_pressed = False
        self.loadModel()
        self.btnRun.config(text="Stop model")
        self.input_path_is_valid = False
        fpath = self.ed_inputpath.get(1.0, END)
        fpath = fpath.rstrip()
        fpath = fpath.lstrip()
        self.sDataInput1Path = fpath

        while True:
            if (os.path.isfile(self.sDataInput1Path)):
                self.ed_inputpath['bg'] = 'green'
                self.input_path_is_valid = True

                try:
                    X0 = np.genfromtxt(self.sDataInput1Path)
                except:
                    pass
                else:
                    X0 = np.float32(X0)
                    X0 = np.reshape(X0, [1, self.nInputs])
                    X0 = self.scaler.transform(X0)

                    Y0 = np.zeros(shape=[1, self.nOutputs])
                    Y0 = np.reshape(Y0, [1, self.nOutputs])

                    os.remove(self.sDataInput1Path)

                    self.sess.run(self.iter.initializer,
                                  feed_dict={self.x: X0, self.y: Y0, self.phBatchSize: 1, self.phDropRate: 0,
                                             self.batch_normalization_active: False})

                    p = self.sess.run(self.prediction)
                    output = ""
                    for i in range(self.nOutputs):
                        output += " "
                        output += str(p[0][i])
                        file = open('answer' + str(i) + '.txt', 'w')
                        file.write(str(p[0][i]))
                        file.close()
                    print(output)
            else:
                self.ed_inputpath['bg'] = 'red'

            if self.stop_run_is_pressed == True:
                break
        self.run_is_launched = False
        self.btnRun.config(text="Run model")

    def threadReloadData(self):
        self.btnReloadData['state'] = 'disabled'
        if (self.tryLoadData() == False):
            self.btnReloadData['bg'] = 'red'
        self.btnReloadData['state'] = 'normal'
        self.updateError()

    def threadDrawTrainingPlot(self, i):
        self.trainingplot.clear()
        if TestSizePercent > 0.0:
            try:
                min_index = np.argmin(self.trainingdata_test_error)
            except:
                return
            else:
                self.trainingplot.plot(self.trainingdata_train_error, color='b',
                                       label='train_loss=' + ("%.4f" % self.trainingdata_train_error[
                                           self.trainingdata_train_error.size - 1]))
                self.trainingplot.plot(self.trainingdata_test_error, color='darkorange',
                                       label='test_loss=' + ("%.4f" % self.trainingdata_test_error[
                                           self.trainingdata_test_error.size - 1]))
                self.trainingplot.axvline(x=self.p_epoch, color='k', linestyle='--',
                                          label='epoch=' + str(self.p_epoch)
                                                + '\ntrain_loss=' + (
                                                            "%.4f" % self.trainingdata_train_error[self.p_epoch])
                                                + '\ntest_loss=' + (
                                                            "%.4f" % self.trainingdata_test_error[self.p_epoch]))
                self.trainingplot.axhline(y=self.trainingdata_test_error[self.p_epoch], color='k', linestyle='--')
                self.trainingplot.legend(loc='upper right')
        else:
            try:
                min_index = np.argmin(self.trainingdata_train_error)
            except:
                return
            else:
                self.trainingplot.plot(self.trainingdata_train_error, color='b',
                                       label='train_loss=' + ("%.4f" % self.trainingdata_train_error[
                                           self.trainingdata_train_error.size - 1]))
                self.trainingplot.axvline(x=self.p_epoch, color='k', linestyle='--',
                                          label='epoch=' + str(self.p_epoch) + '\ntrain_loss=' + (
                                                      "%.4f" % self.trainingdata_train_error[self.p_epoch]))
                self.trainingplot.axhline(y=self.trainingdata_train_error[self.p_epoch], color='k', linestyle='--')
                self.trainingplot.legend(loc='upper right')

    def updateError(self):
        self.sess.run(self.iter.initializer,
                      feed_dict={self.x: self.X, self.y: self.Y, self.phBatchSize: self.nDataSize,
                                 self.phDropRate: 0,
                                 self.batch_normalization_active: False})
        self.losses = self.sess.run(self.loss)
        self.lbl_losses.config(text="Loss: " + str(self.losses))

    def saveSettings(self):
        fname = 'settings.txt'
        keys = self.settings.keys()
        values = self.settings.values()
        f = open(fname, 'w')
        for i in keys:
            s_value = str(i) + ":" + str(self.settings[i]) + "\n"
            f.write(s_value)

    def loadSettings(self):
        keys = self.settings.keys()
        values = self.settings.values()
        fname = 'settings.txt'
        if (os.path.isfile(fname)):
            f = open(fname, 'r')
            a = f.read()
            for i in keys:
                x = a.find(i)
                if (x != -1):
                    y = a.find(":", x)
                    if (y != -1):
                        z = a.find("\n", y)
                        if (z != -1):
                            self.settings[i] = self.settingsDtypes[i](a[y + 1:z])
                            self.settingsUI[i].insert(1.0, a[y + 1:z])
                        else:
                            pass
                    else:
                        pass
                else:
                    self.settings[i] = self.settingsDefault[i]

        else:
            pass

    def loadModelStructure(self):
        fname = 'model.txt'
        if (os.path.isfile(fname)):
            f = open(fname, 'r')
            a = f.read()
            n = a.split('\n')

    def initStructure(self):
        return

    def initSettings(self):
        self.settings = {
            'epochs': 0,
            'stop_error': 0,
            'ls': 0,
            'l1': 0,
            'l2': 0,
            'drop_rate': 0,
            'overfit_epochs': 0
        }
        self.settingsUI = {
            'epochs': self.ed_trs_epochs,
            'stop_error': self.ed_trs_stoperror,
            'ls': self.ed_trs_ls,
            'l1': self.ed_trs_l1,
            'l2': self.ed_trs_l2,
            'drop_rate': self.ed_trs_droprate,
            'overfit_epochs': self.ed_trs_ovf_epochs
        }
        self.settingsDtypes = {
            'epochs': int,
            'stop_error': np.float32,
            'ls': np.float32,
            'l1': np.float32,
            'l2': np.float32,
            'drop_rate': np.float32,
            'overfit_epochs': int
        }
        self.settingsDefault = {
            'epochs': 10000,
            'stop_error': 0.00000001,
            'ls': 0.0001,
            'l1': 0.001,
            'l2': 0.01,
            'drop_rate': 0.15,
            'overfit_epochs': 2000
        }
        self.loadSettings()

        self.training_is_launched = False
        self.run_is_launched = False
        self.saving_is_launched = False
        self.data_is_loaded = False
        self.savepng_is_launched = False
        self.test_model = False
        self.model_is_tested = True
        self.testing_model = False

    def initInterface(self):
        self.root = Tk()
        self.root.minsize(width=430, height=330)

        # filenames
        self.sDataInputPath = ''
        self.sDataOutputPath = ''
        self.sDataInput1Path = ''

        # buttons
        self.btnTrain = Button(self.root, height=2, width=20, text='Train model')
        self.btnRun = Button(self.root, height=2, width=20, text='Run model')
        self.btnReloadData = Button(self.root, height=2, width=20, text='Reload data')
        self.btnShowTrainingPlot = Button(self.root, height=2, width=20, text='Show training plot')
        self.btnShowTestingPlot = Button(self.root, height=2, width=20, text='Show testing plot')

        # file paths
        self.lbl_indatapath = Label(self.root, height=1, width=12, font='Arial 11', bg="white", fg="black",
                                    text='in_data fname :', anchor=W, justify=LEFT)
        self.lbl_outdatapath = Label(self.root, height=1, width=12, font='Arial 11', bg="white", fg="black",
                                     text='out_data fname:', anchor=W, justify=LEFT)
        self.lbl_inputpath = Label(self.root, height=1, width=12, font='Arial 11', bg="white", fg="black",
                                   text='input fname   :', anchor=W, justify=LEFT)
        self.ed_indatapath = Text(self.root, height=1, width=12, font='Arial 11', wrap=WORD)
        self.ed_outdatapath = Text(self.root, height=1, width=12, font='Arial 11', wrap=WORD)
        self.ed_inputpath = Text(self.root, height=1, width=12, font='Arial 11', wrap=WORD)
        self.ed_indatapath.insert(1.0, 'in_data.txt')
        self.ed_outdatapath.insert(1.0, 'out_data.txt')
        self.ed_inputpath.insert(1.0, 'input.txt')

        # training settings
        self.lbl_trs_epochs = Label(self.root, height=1, width=12, font='Arial 11', bg="white", fg="black",
                                    text='epochs:', anchor=W, justify=LEFT)
        self.lbl_trs_stoperror = Label(self.root, height=1, width=12, font='Arial 11', bg="white", fg="black",
                                       text='stop error:', anchor=W, justify=LEFT)
        self.lbl_trs_ls = Label(self.root, height=1, width=12, font='Arial 11', bg="white", fg="black",
                                text='training speed:', anchor=W, justify=LEFT)
        self.lbl_trs_l1 = Label(self.root, height=1, width=12, font='Arial 11', bg="white", fg="black", text='l1:',
                                anchor=W, justify=LEFT)
        self.lbl_trs_l2 = Label(self.root, height=1, width=12, font='Arial 11', bg="white", fg="black", text='l2:',
                                anchor=W, justify=LEFT)
        self.lbl_trs_droprate = Label(self.root, height=1, width=12, font='Arial 11', bg="white", fg="black",
                                      text='drop rate:', anchor=W, justify=LEFT)
        self.lbl_trs_ovf_epochs = Label(self.root, height=1, width=12, font='Arial 11', bg="white", fg="black",
                                        text='ovf epochs:', anchor=W, justify=LEFT)
        #
        self.ed_trs_epochs = Text(self.root, height=1, width=12, font='Arial 11', wrap=WORD)
        self.ed_trs_stoperror = Text(self.root, height=1, width=12, font='Arial 11', wrap=WORD)
        self.ed_trs_ls = Text(self.root, height=1, width=12, font='Arial 11', wrap=WORD)
        self.ed_trs_l1 = Text(self.root, height=1, width=12, font='Arial 11', wrap=WORD)
        self.ed_trs_l2 = Text(self.root, height=1, width=12, font='Arial 11', wrap=WORD)
        self.ed_trs_droprate = Text(self.root, height=1, width=12, font='Arial 11', wrap=WORD)
        self.ed_trs_ovf_epochs = Text(self.root, height=1, width=12, font='Arial 11', wrap=WORD)
        self.lbl_losses = Label(self.root, height=1, width=16, font='Arial 11', bg="white", fg="black", text='',
                                anchor=W, justify=LEFT)

        # binds
        self.ed_indatapath.bind('<KeyRelease>', self.onChangePath)
        self.ed_outdatapath.bind('<KeyRelease>', self.onChangePath)
        self.ed_inputpath.bind('<KeyRelease>', self.onChangePath)
        #
        self.ed_trs_epochs.bind('<KeyRelease>',
                                lambda event, u_index='epochs', format=int: self.onChangeSettings(event, u_index,
                                                                                                  format))
        self.ed_trs_stoperror.bind('<KeyRelease>',
                                   lambda event, u_index='stop_error', format=float: self.onChangeSettings(event,
                                                                                                           u_index,
                                                                                                           format))
        self.ed_trs_ls.bind('<KeyRelease>',
                            lambda event, u_index='ls', format=float: self.onChangeSettings(event, u_index, format))
        self.ed_trs_l1.bind('<KeyRelease>',
                            lambda event, u_index='l1', format=float: self.onChangeSettings(event, u_index, format))
        self.ed_trs_l2.bind('<KeyRelease>',
                            lambda event, u_index='l2', format=float: self.onChangeSettings(event, u_index, format))
        self.ed_trs_droprate.bind('<KeyRelease>',
                                  lambda event, u_index='drop_rate', format=float: self.onChangeSettings(event, u_index,
                                                                                                         format))
        self.ed_trs_ovf_epochs.bind('<KeyRelease>',
                                    lambda event, u_index='overfit_epochs', format=int: self.onChangeSettings(event,
                                                                                                              u_index,
                                                                                                              format))
        #
        self.ed_trs_epochs.bind('<FocusOut>',
                                lambda event, u_index='epochs', format=int: self.onChangeSettings(event, u_index,
                                                                                                  format))
        self.ed_trs_stoperror.bind('<FocusOut>',
                                   lambda event, u_index='stop_error', format=float: self.onChangeSettings(event,
                                                                                                           u_index,
                                                                                                           format))
        self.ed_trs_ls.bind('<FocusOut>',
                            lambda event, u_index='ls', format=float: self.onChangeSettings(event, u_index, format))
        self.ed_trs_l1.bind('<FocusOut>',
                            lambda event, u_index='l1', format=float: self.onChangeSettings(event, u_index, format))
        self.ed_trs_l2.bind('<FocusOut>',
                            lambda event, u_index='l2', format=float: self.onChangeSettings(event, u_index, format))
        self.ed_trs_droprate.bind('<FocusOut>',
                                  lambda event, u_index='drop_rate', format=float: self.onChangeSettings(event, u_index,
                                                                                                         format))
        self.ed_trs_ovf_epochs.bind('<FocusOut>',
                                    lambda event, u_index='overfit_epochs', format=int: self.onChangeSettings(event,
                                                                                                              u_index,
                                                                                                              format))
        #
        self.btnTrain.bind('<Button 1>', self.onClickTrainBtn)
        self.btnRun.bind('<Button 1>', self.onClickRunBtn)
        self.btnReloadData.bind('<Button 1>', self.onClickReloadDataBtn)
        self.btnShowTrainingPlot.bind('<Button 1>', self.onClickShowTrainingPlot)
        self.btnShowTestingPlot.bind('<Button 1>', self.onClickShowTestingPlot)

        # placement
        self.ed_indatapath.place(x=140, y=10)
        self.ed_outdatapath.place(x=140, y=40)
        self.ed_inputpath.place(x=140, y=70)

        self.btnRun.place(x=270, y=80)
        self.btnTrain.place(x=270, y=130)
        self.btnReloadData.place(x=270, y=180)
        self.btnShowTestingPlot.place(x=270, y=230)
        self.btnShowTrainingPlot.place(x=270, y=280)

        self.lbl_losses.place(x=270, y=10)
        self.lbl_indatapath.place(x=10, y=10)
        self.lbl_outdatapath.place(x=10, y=40)
        self.lbl_inputpath.place(x=10, y=70)

        self.lbl_trs_epochs.place(x=10, y=120)
        self.lbl_trs_stoperror.place(x=10, y=150)
        self.lbl_trs_ovf_epochs.place(x=10, y=180)
        self.lbl_trs_ls.place(x=10, y=210)
        self.lbl_trs_l1.place(x=10, y=240)
        self.lbl_trs_l2.place(x=10, y=270)
        self.lbl_trs_droprate.place(x=10, y=300)

        self.ed_trs_epochs.place(x=140, y=120)
        self.ed_trs_stoperror.place(x=140, y=150)
        self.ed_trs_ovf_epochs.place(x=140, y=180)
        self.ed_trs_ls.place(x=140, y=210)
        self.ed_trs_l1.place(x=140, y=240)
        self.ed_trs_l2.place(x=140, y=270)
        self.ed_trs_droprate.place(x=140, y=300)

    def loadData(self):
        _file = False
        while _file == False:
            if os.path.isfile(self.sDataInputPath):
                if os.path.isfile(self.sDataOutputPath):
                    _file = True
        # import data
        self.X = np.genfromtxt(self.sDataInputPath)
        self.Y = np.genfromtxt(self.sDataOutputPath)
        self.X = np.float32(self.X)
        self.Y = np.float32(self.Y)

        self.nInputs = self.X.shape[1]
        try:
            self.nOutputs = self.Y.shape[1]
        except:
            self.nOutputs = 1
        self.nDataSize = self.X.shape[0]
        self.X = np.reshape(self.X, [self.nDataSize, self.nInputs])
        self.Y = np.reshape(self.Y, [self.nDataSize, self.nOutputs])
        self.test_outputs = self.Y
        # self.test_outputs.fill(0)

        self.scaler = preprocessing.MinMaxScaler(feature_range=(Preprocessing_Min, Preprocessing_Max))

        self.X = self.scaler.fit_transform(self.X)
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y,
                                                                                test_size=TestSizePercent, shuffle=True)

        self.nTrainSize = self.X_train.shape[0]
        self.nTestSize = self.X_test.shape[0]

        self.data_is_loaded = True

        self.losses = 0

        self.n_batches_train = int(self.nTrainSize * BatchMod)

        self.X = np.reshape(self.X, [self.nDataSize, self.nInputs, 1])
        self.Y = np.reshape(self.Y, [self.nDataSize, self.nOutputs])
        self.X_train = np.reshape(self.X_train, [self.nTrainSize, self.nInputs, 1])
        self.Y_train = np.reshape(self.Y_train, [self.nTrainSize, self.nOutputs])
        self.X_test = np.reshape(self.X_test, [self.nTestSize, self.nInputs, 1])
        self.Y_test = np.reshape(self.Y_test, [self.nTestSize, self.nOutputs])

    def initPlots(self):
        # plot
        self.trainingfig = plt.figure(figsize=(10, 7), dpi=80, num='Training plot')
        self.trainingplot = self.trainingfig.add_subplot(111)
        self.trainingani = animation.FuncAnimation(self.trainingfig, self.threadDrawTrainingPlot, interval=1000)
        self.trainingfignum = self.trainingfig.number

        self.testingfig = plt.figure(figsize=(10, 7), dpi=80, num='Testing plot')
        self.testingplot = self.testingfig.add_subplot(111)
        self.testingani = animation.FuncAnimation(self.testingfig, self.theadDrawTestingPlot, interval=1000)
        self.testingfignum = self.testingfig.number

    def __init__(self):
        self.historyCallback = historyCallback()

        self.initInterface()
        self.initPlots()
        self.initSettings()
        self.saveSettings()

        # loadData
        self.tryLoadData()

        # tf && model
        self.initModelName()
        self.initModel()
        self.loadModel()
        self.updateError()

        # run
        self.root.mainloop()
        self.saveSettings()
        # self.root.withdraw()


z = app()
# z.ani = animation.FuncAnimation(z.fig, drawthread, interval=1000)
# z.root.mainloop()
