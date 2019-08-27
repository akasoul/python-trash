from tkinter import *
from tkinter import filedialog
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.animation as animation
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib import pyplot as plt

from sklearn import preprocessing
import threading
import os
import random
import time
import msvcrt


# defines
elu         = tf.nn.elu
sig         = tf.nn.sigmoid
tan         = tf.nn.tanh
relu        = tf.nn.relu
softsign    = tf.nn.softsign

##nn structure
#f_l_f = elu
#neurons=100
#struct = np.array([[neurons,neurons,neurons,neurons,neurons,neurons,neurons,neurons],
#                   [f_l_f,f_l_f,f_l_f,f_l_f,f_l_f,f_l_f,f_l_f,f_l_f]])
#outputs_af = tan


#nn structure
f_l_f = elu
neurons=1024
struct = np.array([[neurons,neurons,neurons],
                   [f_l_f,f_l_f,f_l_f]])
outputs_af = None




Preprocessing_Min=-1.0
Preprocessing_Max=1.0
TestSizePercent = 0.1
MaxBatchSize=300





class app:


    def tryLoadData(self):
        self.data_is_loading=True;
        self.path_is_valid=False
        fpath=self.ed_indatapath.get(1.0, END)
        fpath=fpath.rstrip()
        fpath=fpath.lstrip()
        self.s_indatapath = fpath

        fpath=self.ed_outdatapath.get(1.0, END)
        fpath=fpath.rstrip()
        fpath=fpath.lstrip()
        self.s_outdatapath = fpath


        #while(self.path_is_valid==False):
        if(os.path.isfile(self.s_indatapath)):
            self.ed_indatapath['bg']= 'green'
            if (os.path.isfile(self.s_outdatapath)):
                self.ed_outdatapath['bg'] = 'green'
                self.path_is_valid=True
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
            if (os.path.isfile(self.s_outdatapath)):
                self.ed_outdatapath['bg'] = 'green'
            else:
                self.ed_outdatapath['bg'] = 'red'
            print('no data')
            self.data_is_loading=False
            return False


        self.data_is_loading=False


    def model(self,x):
        net = tf.layers.dense(inputs=x, units=struct[0][0], activation=struct[1][0], kernel_regularizer=self.regularizer,
                              bias_regularizer=self.regularizer, kernel_initializer=self.initializer,
                              bias_initializer=self.initializer)

        for i in range(struct.shape[1] - 1):
            net = tf.layers.dense(inputs=net, units=struct[0][i + 1], activation=struct[1][i + 1],
                                  kernel_regularizer=self.regularizer,
                                  bias_regularizer=self.regularizer, kernel_initializer=self.initializer,
                                  bias_initializer=self.initializer)

            if (self.batch_normalization_active == True):
                net = tf.layers.batch_normalization(inputs=net)

            net = tf.layers.dropout(net, rate=self.drop_rate)

        net = tf.layers.dense(inputs=net, units=self.n_outputs, activation=outputs_af, kernel_regularizer=self.regularizer,
                              bias_regularizer=self.regularizer, kernel_initializer=self.initializer,
                              bias_initializer=self.initializer)

        return net

#    def model(self,x):
#        net = tf.keras.layers.Convolution1D(inputs=x,filters=10,activation=elu,kernel_size=10,
#                              kernel_regularizer=self.regularizer,bias_regularizer=self.regularizer,
#                               kernel_initializer=self.initializer,bias_initializer=self.initializer)
#
#        for i in range(struct.shape[1] - 1):
#            net = tf.keras.layers.Convolution1D(inputs=net, filters=10, activation=elu,kernel_size=10,
#            kernel_regularizer = self.regularizer, bias_regularizer = self.regularizer,
#                                                                      kernel_initializer = self.initializer, bias_initializer = self.initializer)
#
#            if (self.batch_normalization_active == True):
#                net = tf.layers.batch_normalization(inputs=net)
#
#            net = tf.layers.dropout(net, rate=self.drop_rate)
#
#        net = tf.layers.dense(inputs=net, units=self.n_outputs, activation=None, kernel_regularizer=self.regularizer,
#                              bias_regularizer=self.regularizer, kernel_initializer=self.initializer,
#                              bias_initializer=self.initializer)
#
#        return net

    def initModelVariables(self):
        # create a placeholder to dynamically switch between batch sizes
        self.batch_size = tf.placeholder(tf.int64)
        self.drop_rate = tf.placeholder(tf.float32)
        self.l1_ph=tf.placeholder(tf.float32)
        self.l2_ph=tf.placeholder(tf.float32)
        self.learning_rate_placeholder = tf.placeholder(tf.float32)
        self.batch_normalization_active = tf.placeholder(tf.bool)
        self.x, self.y = tf.placeholder(tf.float32, shape=[None, self.n_inputs]), tf.placeholder(tf.float32, shape=[None, self.n_outputs])
        self.dataset = tf.data.Dataset.from_tensor_slices((self.x, self.y)).batch(self.batch_size).shuffle(1).repeat()

        self.iter = self.dataset.make_initializable_iterator()
        self.features, self.labels = self.iter.get_next()

        self.regularizer = tf.contrib.layers.l1_l2_regularizer(scale_l1=self.settings['l1'], scale_l2=self.settings['l2'])
        #self.regularizer = tf.contrib.layers.l1_l2_regularizer(scale_l1=self.settings['l1'], scale_l2=self.settings['l2'])
        self.initializer = tf.contrib.layers.xavier_initializer()

        self.prediction = self.model(self.features)
        self.ans = tf.argmax(self.model(self.features), 1)
        self.wb = tf.trainable_variables()
        self.reg_l1_l2 = tf.contrib.layers.l1_l2_regularizer(scale_l1=self.settings['l1'], scale_l2=self.settings['l2'])
        #self.reg_l1_l2 = tf.contrib.layers.l1_l2_regularizer(scale_l1=self.settings['l1'], scale_l2=self.settings['l2'])
        self.rp_l1_l2 = tf.contrib.layers.apply_regularization(self.reg_l1_l2, self.wb)

        self.reg_l1 = tf.contrib.layers.l1_regularizer(scale=self.settings['l1'])
        #self.reg_l1 = tf.contrib.layers.l1_regularizer(scale=self.settings['l1'])
        self.rp_l1 = tf.contrib.layers.apply_regularization(self.reg_l1, self.wb)

        self.reg_l2 = tf.contrib.layers.l2_regularizer(scale=self.settings['l2'])
        #self.reg_l2 = tf.contrib.layers.l2_regularizer(scale=self.settings['l2'])
        self.rp_l2 = tf.contrib.layers.apply_regularization(self.reg_l2, self.wb)

        self.loss = tf.losses.mean_squared_error(predictions=self.prediction, labels=self.labels)
        #self.loss = tf.losses.absolute_difference(predictions=self.prediction, labels=self.labels)
        if self.settings['l1'] > 0 and self.settings['l2'] == 0:
            self.loss_reg = tf.losses.mean_squared_error(predictions=self.prediction, labels=self.labels) + self.rp_l1
        if self.settings['l2'] > 0 and self.settings['l1'] == 0:
            self.loss_reg = tf.losses.mean_squared_error(predictions=self.prediction, labels=self.labels) + self.rp_l2
        if self.settings['l1'] == 0 and self.settings['l2'] == 0:
            self.loss_reg = tf.losses.mean_squared_error(predictions=self.prediction, labels=self.labels)
        if self.settings['l1'] > 0 and self.settings['l2'] > 0:
            self.loss_reg = tf.losses.mean_squared_error(predictions=self.prediction, labels=self.labels) + self.rp_l1_l2

        # train_op = tf.train.RMSPropOptimizer(learning_rate=LearningRate).minimize(loss_reg)
        # train_op = tf.train.AdagradOptimizer(learning_rate=LearningRate).minimize(loss_reg)
        #self.train_op = tf.train.AdamOptimizer(learning_rate=self.settings['ls']).minimize(self.loss_reg)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate_placeholder).minimize(self.loss_reg)
        #self.train_op = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate_placeholder).minimize(self.loss_reg)

        self.saver = tf.train.Saver()
        self.sess=tf.Session()
        self.sess.run(tf.global_variables_initializer())




    def initModelName(self):
        self.model_path="models/"
        self.model_name = str(self.n_inputs)
        for index in range(struct.shape[1]):
            self.model_name += "_"
            self.model_name += str(struct[0][index])
        self.model_name += "_"
        self.model_name += str(self.n_outputs)
        self.model_path=self.model_path+self.model_name+'/'

        self.model_name='model'
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)


    def loadModel(self):
        printnomodel=False
        if os.path.isfile(self.model_path + self.model_name + '.skpt.meta'):
            if os.path.isfile(self.model_path + self.model_name + '.skpt.index'):
                if os.path.isfile(self.model_path + self.model_name + '_error.txt'):
                    self.save_path = self.saver.restore(self.sess, self.model_path + self.model_name + '.skpt')
                    self.restored_error = np.genfromtxt(self.model_path + self.model_name + '_error.txt')
                else:
                    printnomodel = True

            else:
                printnomodel = True

        else:
            printnomodel=True
        if printnomodel==True:
            print('no model')

    def setUiBlocking(self, type, state):
        if type=='run':
            ui = [
                  self.btnTrain,
                    ]
        if type=='train':
            ui = [self.ed_indatapath,
                  self.ed_outdatapath,
                  self.ed_inputpath,
                  self.btnRun,
                  self.ed_trs_ls,
                  self.ed_trs_l1,
                  self.ed_trs_l2,
                  self.ed_trs_droprate
                  ]
        for i in ui:
            if(state=='block'):
                i['state']='disabled'
                i['bg']='silver'
            if(state=='unblock'):
                i['state'] = 'enabled'
                i['bg'] = 'white'



    def onClickRunBtn(self, event):
        if self.run_is_launched==False:
            tt = threading.Thread(target=self.theadRun)
            tt.daemon = True
            tt.start()
        else:
            self.stop_run_is_pressed=True


    def onClickTrainBtn(self, event):
        if self.training_is_launched==False:
            tt = threading.Thread(target=self.thread_train)
            tt.daemon = True
            tt.start()
        else:
            self.stop_train_is_pressed=True

    def onClickReloadDataBtn(self, event):
        tt = threading.Thread(target=self.threadReloadData)
        tt.daemon = True
        tt.start()



    def onClickShowTrainingPlot(self,event):
        self.trainingfig = plt.figure(figsize=(10, 7), dpi=80,num='Training plot')
        self.trainingplot=self.trainingfig.add_subplot(111)
        self.trainingani=animation.FuncAnimation(self.trainingfig, self.threadDrawTrainingPlot, interval=1000)
        self.trainingfignum=self.trainingfig.number

        self.trainingfig.show()
        #if(plt.fignum_exists(self.trainingfignum)):
        #    pass
        #else:

    def onClickShowTestingPlot(self,event):
        self.testingfig = plt.figure(figsize=(10, 7), dpi=80,num='Testing plot')
        self.testingplot=self.testingfig.add_subplot(111)
        self.testingani=animation.FuncAnimation(self.testingfig, self.theadDrawTestingPlot, interval=1000)
        self.testingfignum=self.testingfig.number

        self.testingfig.show()
        #if(plt.fignum_exists(self.testingfignum)):
        #    pass
        #else:


    def onChangePath(self, event):
        if self.data_is_loaded==False:
            self.tryLoadData()

    def onChangeSettings(self, event, ui_index, format):
        i=ui_index
        value=self.settingsui[i].get(1.0,END)
        value = value.rstrip()
        try:
            format(value)
        except:
            self.settingsui[i]['bg']='red'
        else:
            self.settingsui[i]['bg']='white'
            self.settings[i]=format(value)
            if self.saving_is_launched == False:
                ts = threading.Thread(target=self.threadSaveSettings)
                ts.daemon = True
                ts.start()

    def threadSaveSettings(self):
        #while(self.training_is_launched):
            #pass
        #while(self.run_is_launched):
            #pass
        self.saving_is_launched=True
        self.saveSettings()
        self.saving_is_launched=False


    def thread_train(self):
        self.training_is_launched=True
        plt.fignum_exists(self.testingfig.number)
        self.stop_train_is_pressed=False
        self.btnTrain.config(text="Stop training")

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


        # initialise iterator with train data
        self.sess.run(self.iter.initializer, feed_dict={self.x: self.X_train, self.y: self.Y_train, self.batch_size: self.batchsize, self.drop_rate: self.settings['drop_rate'],
                                                   self.batch_normalization_active: True})
        print('Training...')

        save_path = self.saver.save(self.sess, self.model_path + self.model_name + '.skpt')

        self.sess.run(self.iter.initializer,
                      feed_dict={self.x: self.X_train, self.y: self.Y_train, self.batch_size: self.train_size,
                                 self.drop_rate: 0,
                                 self.batch_normalization_active: False})
        p_train_loss = self.sess.run(self.loss)
        self.trainingdata_train_error = np.append(self.trainingdata_train_error, p_train_loss)

        if (TestSizePercent > 0.0):
            self.sess.run(self.iter.initializer,
                          feed_dict={self.x: self.X_test, self.y: self.Y_test, self.batch_size: self.test_size,
                                     self.drop_rate: 0,
                                     self.batch_normalization_active: False})
            p_test_loss = self.sess.run(self.loss)
            self.trainingdata_test_error = np.append(self.trainingdata_test_error, p_test_loss)

        self.p_epoch = 0
        epoch = 0
        of_counter = 0
        for i in range(self.settings['epochs']):
            self.sess.run(self.iter.initializer,
                     feed_dict={self.x: self.X_train, self.y: self.Y_train, self.batch_size: self.batchsize, self.drop_rate: self.settings['drop_rate'],
                                self.batch_normalization_active: True})
            train_loss = 0
            for _ in range(self.n_batches_train):
                _, loss_value = self.sess.run([self.train_op, self.loss],feed_dict={self.learning_rate_placeholder: self.settings['ls']})
                train_loss += loss_value
            train_loss /= self.n_batches_train
            self.trainingdata_train_error = np.append(self.trainingdata_train_error, train_loss)
            if TestSizePercent > 0.0:
                self.sess.run(self.iter.initializer, feed_dict={self.x: self.X_test, self.y: self.Y_test, self.batch_size: self.test_size, self.drop_rate: 0,
                                                           self.batch_normalization_active: True})
                test_loss = self.sess.run(self.loss)
                print("Iter: {0:4d} TrainLoss: {1:.10f} TestLoss: {2:.10f}".format(i, train_loss, test_loss))
                self.trainingdata_test_error = np.append(self.trainingdata_test_error, test_loss)
            else:
                print("Iter: {0:4d} Loss: {1:.10f}".format(i, train_loss, ))

            epoch = epoch + 1
            if TestSizePercent>0.0:
                if (test_loss < p_test_loss and i > 0 and train_loss < p_train_loss):
                    self.p_epoch = i+1
                    p_test_loss = test_loss
                    p_train_loss = train_loss
                    save_path = self.saver.save(self.sess, self.model_path + self.model_name + '.skpt')
                    of_counter = 0
                    if (self.test_model == True):
                        self.model_is_tested=False
                        self.sess.run(self.iter.initializer,
                                      feed_dict={self.x: self.X, self.y: self.Y, self.batch_size: self.n_datasize,
                                                 self.drop_rate: 0,
                                                 self.batch_normalization_active: False})
                        self.test_outputs = self.sess.run(self.prediction)  # , feed_dict={ x: X, y: Y, batch_size: data_size})
                        self.model_is_tested=True
                else:
                    of_counter = of_counter + 1
            else:
                if (train_loss < p_train_loss and i > 0):
                    self.p_epoch = i+1
                    p_train_loss = train_loss
                    save_path = self.saver.save(self.sess, self.model_path + self.model_name + '.skpt')
                    of_counter = 0
                    if (self.test_model == True):
                        self.model_is_tested=False
                        self.sess.run(self.iter.initializer,
                                      feed_dict={self.x: self.X, self.y: self.Y, self.batch_size: self.n_datasize,
                                                 self.drop_rate: 0,
                                                 self.batch_normalization_active: False})
                        self.test_outputs = self.sess.run(self.prediction)  # , feed_dict={ x: X, y: Y, batch_size: data_size})
                        self.model_is_tested=True
                else:
                    of_counter = of_counter + 1

            if (of_counter > self.settings['overfit_epochs']):
                break

            if (test_loss < self.settings['stop_error']):
                break

            if (self.stop_train_is_pressed == True):
                break

        save_path = self.saver.restore(self.sess, self.model_path + self.model_name + '.skpt')
        self.sess.run(self.iter.initializer,
                 feed_dict={self.x: self.X, self.y: self.Y, self.batch_size: self.n_datasize, self.drop_rate: 0, self.batch_normalization_active: False})
        tot_loss = self.sess.run(self.loss)

        file = open(self.model_path + self.model_name + '_error.txt', 'w')
        file.write(str(tot_loss))
        file.close()

        if TestSizePercent>0.0:
            print("Selected Iter: {0:4d} TrainLoss: {1:.10f} TestLoss: {2:.10f}"
                    .format(self.p_epoch,self.trainingdata_train_error[self.p_epoch],self.trainingdata_test_error[self.p_epoch]))
        else:
            print("Selected Iter: {0:4d} TrainLoss: {1:.10f}"
                    .format(self.p_epoch,self.trainingdata_train_error[self.p_epoch]))
        self.updateError()
        self.training_is_launched=False
        self.btnTrain.config(text="Train model")
        return



    def theadRun(self):
        self.run_is_launched=True
        self.stop_run_is_pressed=False
        self.loadModel()
        self.btnRun.config(text="Stop model")
        self.input_path_is_valid=False
        fpath=self.ed_inputpath.get(1.0, END)
        fpath=fpath.rstrip()
        fpath=fpath.lstrip()
        self.s_inputpath = fpath

        while True:
            if (os.path.isfile(self.s_inputpath)):
                self.ed_inputpath['bg'] = 'green'
                self.input_path_is_valid=True

                try:
                    X0 = np.genfromtxt(self.s_inputpath)
                except:
                    pass
                else:
                    X0 = np.float32(X0)
                    X0 = np.reshape(X0, [1, self.n_inputs])
                    X0 = self.scaler.transform(X0)

                    Y0 = np.zeros(shape=[1, self.n_outputs])
                    Y0 = np.reshape(Y0, [1, self.n_outputs])

                    os.remove(self.s_inputpath)

                    self.sess.run(self.iter.initializer,
                             feed_dict={self.x: X0, self.y: Y0, self.batch_size: 1, self.drop_rate: 0, self.batch_normalization_active: False})

                    p = self.sess.run(self.prediction)
                    output = ""
                    for i in range(self.n_outputs):
                        output += " "
                        output += str(p[0][i])
                        file = open('answer' + str(i) + '.txt', 'w')
                        file.write(str(p[0][i]))
                        file.close()
                    print(output)
            else:
                self.ed_inputpath['bg'] = 'red'

            if self.stop_run_is_pressed==True:
                break
        self.run_is_launched=False
        self.btnRun.config(text="Run model")

    def threadReloadData(self):
        self.btnReloadData['state']= 'disabled'
        if(self.tryLoadData()==False):
            self.btnReloadData['bg']= 'red'
        self.btnReloadData['state'] = 'normal'
        self.updateError()

    def threadDrawTrainingPlot(self, i):
        self.trainingplot.clear()
        if TestSizePercent>0.0:
            try:
                min_index=np.argmin(self.trainingdata_test_error)
            except:
                return
            else:
                self.trainingplot.plot(self.trainingdata_train_error,color='b',
                                       label='train_loss='+("%.4f" % self.trainingdata_train_error[self.trainingdata_train_error.size-1]))
                self.trainingplot.plot(self.trainingdata_test_error,color='darkorange',
                                       label='test_loss='+("%.4f" % self.trainingdata_test_error[self.trainingdata_test_error.size-1]))
                self.trainingplot.axvline(x=self.p_epoch, color='k', linestyle='--',
                                          label='epoch='+str(self.p_epoch)
                                                +'\ntrain_loss=' + ("%.4f" % self.trainingdata_train_error[self.p_epoch])
                                                +'\ntest_loss=' + ("%.4f" % self.trainingdata_test_error[self.p_epoch]))
                self.trainingplot.axhline(y=self.trainingdata_test_error[self.p_epoch], color='k', linestyle='--')
                self.trainingplot.legend(loc='upper right')
        else:
            try:
                min_index=np.argmin(self.trainingdata_train_error)
            except:
                return
            else:
                self.trainingplot.plot(self.trainingdata_train_error,color='b',
                                       label='train_loss='+("%.4f" % self.trainingdata_train_error[self.trainingdata_train_error.size-1]))
                self.trainingplot.axvline(x=self.p_epoch, color='k', linestyle='--',label='epoch='+str(self.p_epoch)+'\ntrain_loss='+("%.4f" % self.trainingdata_train_error[self.p_epoch]))
                self.trainingplot.axhline(y=self.trainingdata_train_error[self.p_epoch], color='k', linestyle='--')
                self.trainingplot.legend(loc='upper right')

    def theadDrawTestingPlot(self, i):
        fnum=self.testingfig.number
        if(self.test_model==False):
            self.test_model = True
        if(self.data_is_loading==False):
            if(self.training_is_launched==False):
                self.sess.run(self.iter.initializer,
                              feed_dict={self.x: self.X, self.y: self.Y, self.batch_size: self.n_datasize, self.drop_rate: 0,
                                         self.batch_normalization_active: False})
                zout = self.sess.run(self.prediction)  # , feed_dict={ x: X, y: Y, batch_size: data_size})
                net_outputs = zout
                targets = self.Y
                if(len(net_outputs)!=len(targets)):
                    return


                self.testingplot.clear()
                self.testingplot.plot(net_outputs, linewidth=1.0, label="outputs", color='r')
                self.testingplot.plot(targets, linewidth=1.0, label="targets", color='b')
            if(self.training_is_launched==True):
                if(self.model_is_tested==True):
                    if (len(self.test_outputs) == len(self.Y)):
                        self.testingplot.clear()
                        self.testingplot.plot(self.test_outputs, linewidth=1.0, label="outputs", color='r')
                        self.testingplot.plot(self.Y, linewidth=1.0, label="targets", color='b')

    def updateError(self):
        self.sess.run(self.iter.initializer,
                      feed_dict={self.x: self.X, self.y: self.Y, self.batch_size: self.n_datasize,
                                 self.drop_rate: 0,
                                 self.batch_normalization_active: False})
        self.losses = self.sess.run(self.loss)
        self.lbl_losses.config(text="Loss: "+str(self.losses))

    def saveSettings(self):
        fname='settings.txt'
        keys=self.settings.keys()
        values=self.settings.values()
        f=open(fname,'w')
        for i in keys:
            s_value=str(i)+":"+str(self.settings[i])+"\n"
            f.write(s_value)


    def loadSettings(self):
        keys=self.settings.keys()
        values = self.settings.values()
        fname='settings.txt'
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
                            self.settings[i]=self.settingsdtypes[i](a[y + 1:z])
                            self.settingsui[i].insert(1.0, a[y + 1:z])
                        else:
                            return False
                    else:
                        return False
                else:
                    return False

        else:
            return False

    def loadModelStructure(self):
        fname='model.txt'
        if (os.path.isfile(fname)):
            f = open(fname, 'r')
            a = f.read()
            n=a.split('\n')

    def initStructure(self):
        return

    def initSettings(self):
        self.settings={
            'epochs':          0,
            'stop_error':       0,
            'ls':               0,
            'l1':               0,
            'l2':               0,
            'drop_rate':        0,
            'overfit_epochs':  0
        }
        self.settingsui={
            'epochs'          :self.ed_trs_epochs,
            'stop_error'       :self.ed_trs_stoperror,
            'ls'               :self.ed_trs_ls,
            'l1'               :self.ed_trs_l1,
            'l2'               :self.ed_trs_l2,
            'drop_rate'        :self.ed_trs_droprate,
            'overfit_epochs'  :self.ed_trs_ovf_epochs
        }
        self.settingsdtypes={
            'epochs'          :int,
            'stop_error'       :np.float32,
            'ls'               :np.float32,
            'l1'               :np.float32,
            'l2'               :np.float32,
            'drop_rate'        :np.float32,
            'overfit_epochs'  :int
        }
        if self.loadSettings()==False:
            self.settings = {
                'epochs': 10000,
                'stop_error': 0.00000001,
                'ls': 0.0001,
                'l1': 0.001,
                'l2': 0.01,
                'drop_rate': 0.15,
                'overfit_epochs':2000
            }
        self.training_is_launched=False
        self.run_is_launched=False
        self.saving_is_launched=False
        self.data_is_loaded=False
        self.savepng_is_launched=False
        self.test_model=False
        self.model_is_tested = True
        self.testing_model=False


    def initInterface(self):
        self.root = Tk()
        self.root.minsize(width=430,height=330)

        self.s_indatapath=''
        self.s_outdatapath=''
        self.s_inputpath=''

        self.new_path=True


        self.frm_training=      Frame(self.root, bg='white', bd=5, height=200, width=300)
        self.frm_testing=       Frame(self.root, bg='white', bd=5, height=200, width=300)

        self.btnTrain=                  Button(self.root, height=2, width=20, text='Train model')
        self.btnRun=                    Button(self.root, height=2, width=20, text='Run model')
        self.btnReloadData=             Button(self.root, height=2, width=20, text='Reload data')
        self.btnShowTrainingPlot=       Button(self.root, height=2, width=20, text='Show training plot')
        self.btnShowTestingPlot=        Button(self.root, height=2, width=20, text='Show testing plot')

        #file paths
        self.lbl_indatapath=    Label(self.root,height=1,width=12,font='Arial 11',bg="white", fg="black",text='in_data fname :',anchor=W, justify=LEFT)
        self.lbl_outdatapath=   Label(self.root,height=1,width=12,font='Arial 11',bg="white", fg="black",text='out_data fname:',anchor=W, justify=LEFT)
        self.lbl_inputpath=     Label(self.root,height=1,width=12,font='Arial 11',bg="white", fg="black",text='input fname   :',anchor=W, justify=LEFT)
        self.ed_indatapath=     Text(self.root, height=1, width=12, font='Arial 11', wrap=WORD)
        self.ed_outdatapath=    Text(self.root, height=1, width=12, font='Arial 11', wrap=WORD)
        self.ed_inputpath=      Text(self.root, height=1, width=12, font='Arial 11', wrap=WORD)

        #training settings
        self.lbl_trs_epochs=        Label(self.root,height=1,width=12,font='Arial 11',bg="white", fg="black",text='epochs:',anchor=W, justify=LEFT)
        self.lbl_trs_stoperror=     Label(self.root,height=1,width=12,font='Arial 11',bg="white", fg="black",text='stop error:',anchor=W, justify=LEFT)
        self.lbl_trs_ls=            Label(self.root,height=1,width=12,font='Arial 11',bg="white", fg="black",text='training speed:',anchor=W, justify=LEFT)
        self.lbl_trs_l1=            Label(self.root,height=1,width=12,font='Arial 11',bg="white", fg="black",text='l1:',anchor=W, justify=LEFT)
        self.lbl_trs_l2=            Label(self.root,height=1,width=12,font='Arial 11',bg="white", fg="black",text='l2:',anchor=W, justify=LEFT)
        self.lbl_trs_droprate=      Label(self.root,height=1,width=12,font='Arial 11',bg="white", fg="black",text='drop rate:',anchor=W, justify=LEFT)
        self.lbl_trs_ovf_epochs=    Label(self.root,height=1,width=12,font='Arial 11',bg="white", fg="black",text='ovf epochs:',anchor=W, justify=LEFT)

        self.ed_trs_epochs=         Text(self.root,height=1,width=12,font='Arial 11', wrap=WORD)
        self.ed_trs_stoperror=      Text(self.root,height=1,width=12,font='Arial 11', wrap=WORD)
        self.ed_trs_ls=             Text(self.root,height=1,width=12,font='Arial 11', wrap=WORD)
        self.ed_trs_l1=             Text(self.root,height=1,width=12,font='Arial 11', wrap=WORD)
        self.ed_trs_l2=             Text(self.root,height=1,width=12,font='Arial 11', wrap=WORD)
        self.ed_trs_droprate=       Text(self.root,height=1,width=12,font='Arial 11', wrap=WORD)
        self.ed_trs_ovf_epochs=     Text(self.root,height=1,width=12,font='Arial 11', wrap=WORD)
        self.lbl_losses=            Label(self.root,height=1,width=16,font='Arial 11',bg="white", fg="black",text='',anchor=W, justify=LEFT)



        self.ed_indatapath.insert(1.0, 'in_data.txt')
        self.ed_outdatapath.insert(1.0, 'out_data.txt')
        self.ed_inputpath.insert(1.0, 'input.txt')

        self.ed_indatapath.bind('<KeyRelease>', self.onChangePath)
        self.ed_outdatapath.bind('<KeyRelease>', self.onChangePath)
        self.ed_inputpath.bind('<KeyRelease>', self.onChangePath)

        #training settings binds
        self.ed_trs_epochs      .bind('<KeyRelease>', lambda event, u_index='epochs'        , format=int:self.onChangeSettings(event, u_index, format))
        self.ed_trs_stoperror   .bind('<KeyRelease>', lambda event, u_index='stop_error'     , format=float:self.onChangeSettings(event, u_index, format))
        self.ed_trs_ls          .bind('<KeyRelease>', lambda event, u_index='ls'             , format=float:self.onChangeSettings(event, u_index, format))
        self.ed_trs_l1          .bind('<KeyRelease>', lambda event, u_index='l1'             , format=float:self.onChangeSettings(event, u_index, format))
        self.ed_trs_l2          .bind('<KeyRelease>', lambda event, u_index='l2'             , format=float:self.onChangeSettings(event, u_index, format))
        self.ed_trs_droprate    .bind('<KeyRelease>', lambda event, u_index='drop_rate'      , format=float:self.onChangeSettings(event, u_index, format))
        self.ed_trs_ovf_epochs  .bind('<KeyRelease>', lambda event, u_index='overfit_epochs', format=int:self.onChangeSettings(event, u_index, format))

        self.ed_trs_epochs      .bind('<FocusOut>', lambda event, u_index='epochs'        , format=int:self.onChangeSettings(event, u_index, format))
        self.ed_trs_stoperror   .bind('<FocusOut>', lambda event, u_index='stop_error'     , format=float:self.onChangeSettings(event, u_index, format))
        self.ed_trs_ls          .bind('<FocusOut>', lambda event, u_index='ls'             , format=float:self.onChangeSettings(event, u_index, format))
        self.ed_trs_l1          .bind('<FocusOut>', lambda event, u_index='l1'             , format=float:self.onChangeSettings(event, u_index, format))
        self.ed_trs_l2          .bind('<FocusOut>', lambda event, u_index='l2'             , format=float:self.onChangeSettings(event, u_index, format))
        self.ed_trs_droprate    .bind('<FocusOut>', lambda event, u_index='drop_rate'      , format=float:self.onChangeSettings(event, u_index, format))
        self.ed_trs_ovf_epochs  .bind('<FocusOut>', lambda event, u_index='overfit_epochs', format=int:self.onChangeSettings(event, u_index, format))

        self.btnTrain          .bind('<Button 1>', self.onClickTrainBtn)
        self.btnRun            .bind('<Button 1>', self.onClickRunBtn)
        self.btnReloadData     .bind('<Button 1>', self.onClickReloadDataBtn)
        self.btnShowTrainingPlot.bind('<Button 1>', self.onClickShowTrainingPlot)
        self.btnShowTestingPlot.bind('<Button 1>', self.onClickShowTestingPlot)

        self.ed_indatapath      .place(x=140, y=10)
        self.ed_outdatapath     .place(x=140, y=40)
        self.ed_inputpath       .place(x=140, y=70)

        self.btnRun.place(x=270, y=80)
        self.btnTrain.place(x=270, y=130)
        self.btnReloadData.place(x=270, y=180)
        self.btnShowTestingPlot.place(x=270, y=230)
        self.btnShowTrainingPlot.place(x=270, y=280)

        self.lbl_losses.place(x=270,y=10)
        self.lbl_indatapath.place(x=10, y=10)
        self.lbl_outdatapath.place(x=10, y=40)
        self.lbl_inputpath.place(x=10, y=70)

        self.lbl_trs_epochs         .place(x=10, y=120)
        self.lbl_trs_stoperror      .place(x=10, y=150)
        self.lbl_trs_ovf_epochs     .place(x=10, y=180)
        self.lbl_trs_ls             .place(x=10, y=210)
        self.lbl_trs_l1             .place(x=10, y=240)
        self.lbl_trs_l2             .place(x=10, y=270)
        self.lbl_trs_droprate       .place(x=10, y=300)

        self.ed_trs_epochs          .place(x=140, y=120)
        self.ed_trs_stoperror       .place(x=140, y=150)
        self.ed_trs_ovf_epochs      .place(x=140, y=180)
        self.ed_trs_ls              .place(x=140, y=210)
        self.ed_trs_l1              .place(x=140, y=240)
        self.ed_trs_l2              .place(x=140, y=270)
        self.ed_trs_droprate        .place(x=140, y=300)


    def loadData(self):
        _file = False
        while _file == False:
            if os.path.isfile(self.s_indatapath):
                if os.path.isfile(self.s_outdatapath):
                    _file = True
        # import data
        self.X = np.genfromtxt(self.s_indatapath)
        self.Y = np.genfromtxt(self.s_outdatapath)
        self.X = np.float32(self.X)
        self.Y = np.float32(self.Y)

        self.n_inputs = self.X.shape[1]
        try:
            self.n_outputs = self.Y.shape[1]
        except:
            self.n_outputs=1
        self.n_datasize = self.X.shape[0]
        self.X = np.reshape(self.X, [self.n_datasize, self.n_inputs])
        self.Y = np.reshape(self.Y, [self.n_datasize, self.n_outputs])
        self.test_outputs=self.Y
        #self.test_outputs.fill(0)

        self.scaler = preprocessing.MinMaxScaler(feature_range=(Preprocessing_Min, Preprocessing_Max))

        self.X = self.scaler.fit_transform(self.X)
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=TestSizePercent, shuffle=True)

        self.train_size = self.X_train.shape[0]
        self.test_size = self.X_test.shape[0]

        self.data_is_loaded=True

        self.losses=0

        for i in range(self.train_size - 1, 0, -1):
            if (self.train_size % i == 0 and i < MaxBatchSize):
                self.batchsize = i
                self.n_batches_train = int(self.train_size / self.batchsize)
                break

    def initPlots(self):
        #plot
        self.trainingfig = plt.figure(figsize=(10, 7), dpi=80,num='Training plot')
        self.trainingplot=self.trainingfig.add_subplot(111)
        self.trainingani=animation.FuncAnimation(self.trainingfig, self.threadDrawTrainingPlot, interval=1000)
        self.trainingfignum=self.trainingfig.number

        self.testingfig = plt.figure(figsize=(10, 7), dpi=80,num='Testing plot')
        self.testingplot=self.testingfig.add_subplot(111)
        self.testingani=animation.FuncAnimation(self.testingfig, self.theadDrawTestingPlot, interval=1000)
        self.testingfignum=self.testingfig.number



    def __init__(self):
        self.initInterface()
        self.initPlots()
        self.initSettings()
        self.saveSettings()

        #loadData
        self.tryLoadData()

        #tf && model
        self.initModelName()
        self.initModelVariables()
        self.loadModel()
        self.updateError()

        #run
        self.root.mainloop()
        self.saveSettings()
        #self.root.withdraw()



z = app()
#z.ani = animation.FuncAnimation(z.fig, drawthread, interval=1000)
#z.root.mainloop()
