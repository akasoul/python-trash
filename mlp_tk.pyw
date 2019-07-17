from tkinter import *
from tkinter import filedialog
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.animation as animation
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
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
#neurons=50
#struct = np.array([[neurons,neurons,neurons,neurons,neurons],
#                   [f_l_f,f_l_f,f_l_f,f_l_f,f_l_f]])
#outputs_af = None


#nn structure
f_l_f = elu
neurons=100
struct = np.array([[neurons,neurons,neurons],
                   [f_l_f,f_l_f,f_l_f]])
outputs_af = None




Preprocessing_Min=-1.0
Preprocessing_Max=1.0
TestSizePercent = 0.1
MaxBatchSize=300




class app:


    def try_load_data(self):
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
                self.load_data()
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

    def init_model_variables(self):
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




    def init_model_name(self):
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


    def load_model(self):
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

    def set_ui_blocking(self,type,state):
        if type=='run':
            ui = [
                  self.btn_train,
                    ]
        if type=='train':
            ui = [self.ed_indatapath,
                  self.ed_outdatapath,
                  self.ed_inputpath,
                  self.btn_run,
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



    def on_click_run_btn(self,event):
        if self.run_is_launched==False:
            tt = threading.Thread(target=self.thread_run)
            tt.daemon = True
            tt.start()
        else:
            self.stop_run_is_pressed=True

    def on_click_saveplot_btn(self,event):
        if self.savepng_is_launched==False:
            tt = threading.Thread(target=self.thread_savepng)
            tt.daemon = True
            tt.start()

        if self.selectedplot.get() == 0:
            self.root.filename = filedialog.asksaveasfilename(initialdir="", title="Select file",
                                                              filetypes=(("png files", "*.png"), ("all files", "*.*")))
            print(self.root.filename)
            self.trainingfig.set_size_inches(40,20)
            self.trainingfig.savefig(self.root.filename + ".png", dpi=100, facecolor='w', edgecolor='w',
                                    orientation='portrait', papertype="a0", format=None,
                                    transparent=False, bbox_inches=None, pad_inches=0.1,
                                    frameon=None, metadata=None)
            self.trainingfig.set_size_inches(7,5)

    def on_click_train_btn(self,event):
        if self.training_is_launched==False:
            tt = threading.Thread(target=self.thread_train)
            tt.daemon = True
            tt.start()
        else:
            self.stop_train_is_pressed=True

    def on_click_reloaddata_btn(self,event):
        tt = threading.Thread(target=self.thread_reloaddata)
        tt.daemon = True
        tt.start()

    def on_click_select_plot(self):#, event):
        if self.selectedplot.get()==0:
            self.frm_testing.place_forget()
            self.frm_training.place(x=270, y=40)
            #self.selectedplot.set(1)
            return
        if self.selectedplot.get()==1:
            self.frm_training.place_forget()
            self.frm_testing.place(x=270, y=40)
            #self.selectedplot.set(0)
            return




    def on_change_path(self, event):
        if self.data_is_loaded==False:
            self.try_load_data()

    def on_change_settings(self, event, ui_index, format):
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
                ts = threading.Thread(target=self.thread_save_settings)
                ts.daemon = True
                ts.start()

    def thread_save_settings(self):
        #while(self.training_is_launched):
            #pass
        #while(self.run_is_launched):
            #pass
        self.saving_is_launched=True
        self.save_settings()
        self.saving_is_launched=False


    def thread_train(self):
        self.training_is_launched=True
        self.stop_train_is_pressed=False
        self.btn_train.config(text="stop")

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
        self.update_error()
        self.training_is_launched=False
        self.btn_train.config(text="start")
        return

    def thread_savepng(self):
        self.savepng_is_launched=True
        if self.selectedplot.get()==1:
            self.root.filename = filedialog.asksaveasfilename(initialdir = "",title = "Select file",filetypes = (("png files","*.png"),("all files","*.*")))
            print (self.root.filename)
            self.testingfig.set_size_inches(40,20)
            self.testingfig.savefig(self.root.filename, dpi=100, facecolor='w', edgecolor='w',
            orientation='portrait', papertype="a0", format=None,
            transparent=False, bbox_inches=None, pad_inches=0.1,
            frameon=None, metadata=None)
            self.testingfig.set_size_inches(7,5)

        if self.selectedplot.get() == 0:
            self.root.filename = filedialog.asksaveasfilename(initialdir="", title="Select file",
                                                              filetypes=(("png files", "*.png"), ("all files", "*.*")))
            print(self.root.filename)
            self.trainingfig.set_size_inches(40,20)
            self.trainingfig.savefig(self.root.filename , dpi=100, facecolor='w', edgecolor='w',
                                    orientation='portrait', papertype="a0", format=None,
                                    transparent=False, bbox_inches=None, pad_inches=0.1,
                                    frameon=None, metadata=None)
            self.trainingfig.set_size_inches(7,5)
        self.savepng_is_launched=False


    def thread_run(self):
        self.run_is_launched=True
        self.stop_run_is_pressed=False
        self.load_model()
        self.btn_run.config(text="stop")
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
        self.btn_run.config(text="start")

    def thread_reloaddata(self):
        self.btn_reloaddata['state']='disabled'
        if(self.try_load_data()==False):
            self.btn_reloaddata['bg']='red'
        self.btn_reloaddata['state'] = 'normal'
        self.update_error()

    def thread_draw(self, i):
        if(self.testing_model==True):
            self.testing_model=False
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

    def thread_draw2(self, i):
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

    def update_error(self):
        self.sess.run(self.iter.initializer,
                      feed_dict={self.x: self.X, self.y: self.Y, self.batch_size: self.n_datasize,
                                 self.drop_rate: 0,
                                 self.batch_normalization_active: False})
        self.losses = self.sess.run(self.loss)
        self.lbl_losses.config(text=self.losses)

    def save_settings(self):
        fname='settings.txt'
        keys=self.settings.keys()
        values=self.settings.values()
        f=open(fname,'w')
        for i in keys:
            s_value=str(i)+":"+str(self.settings[i])+"\n"
            f.write(s_value)


    def load_settings(self):
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

    def load_model_structure(self):
        fname='model.txt'
        if (os.path.isfile(fname)):
            f = open(fname, 'r')
            a = f.read()
            n=a.split('\n')

    def init_structure(self):
        return

    def init_settings(self):
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
        if self.load_settings()==False:
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

    def on_change_layers_count(self,event):
        #a1=self.list_ns_layers.getint(ACTIVE)
        if self.layers_count!=self.list_ns_layers.get(ACTIVE):
            self.layers_count = self.list_ns_layers.get(ACTIVE)
            j=len(self.layers)
            for i in range(0,j):
                self.layers[i].place_forget()
                self.layers_af[i].place_forget()
            self.lbl_out_label.place_forget()
            self.ed_ns_out_af.place_forget()
            border=10;
            for i in range(0,self.layers_count):
                self.layers[i].place(x=border, y=550)
                self.layers_af[i].place(x=border, y=590)
                border=border+50
            self.lbl_out_label.place(x=border, y=550)
            self.ed_ns_out_af.place(x=border, y=590)

    def init_interface(self):
        self.root = Tk()
        self.root.minsize(width=790,height=430)

        self.s_indatapath=''
        self.s_outdatapath=''
        self.s_inputpath=''

        self.new_path=True

        self.selectedplot=IntVar()
        self.selectedplot.set(0)
        self.btn_trainingplot=  Radiobutton(self.root,var=self.selectedplot,value=0,text='training data',command=self.on_click_select_plot)
        self.btn_testingplot=   Radiobutton(self.root,var=self.selectedplot,value=1,text='testing model',command=self.on_click_select_plot)

        self.frm_training=      Frame(self.root, bg='white', bd=5, height=200, width=300)
        self.frm_testing=       Frame(self.root, bg='white', bd=5, height=200, width=300)
        self.lbl_train=         Label(self.root,height=1,width=12,font='Arial 11',bg="white", fg="black",text='Train model',anchor=W, justify=LEFT)
        self.lbl_run=           Label(self.root,height=1,width=12,font='Arial 11',bg="white", fg="black",text='Run model',anchor=W, justify=LEFT)
        self.lbl_reloaddata=    Label(self.root,height=1,width=12,font='Arial 11',bg="white", fg="black",text='Reload data',anchor=W, justify=LEFT)
        self.btn_train=         Button(self.root,height=1,width=10,text='start')
        self.btn_run=           Button(self.root,height=1,width=10,text='start')
        self.btn_reloaddata=    Button(self.root,height=1,width=10,text='start')
        self.btn_saveplot=      Button(self.root,height=1,width=10,text='Save plot')
        #file paths
        self.lbl_indatapath=    Label(self.root,height=1,width=12,font='Arial 11',bg="white", fg="black",text='in_data fname :',anchor=W, justify=LEFT)
        self.lbl_outdatapath=   Label(self.root,height=1,width=12,font='Arial 11',bg="white", fg="black",text='out_data fname:',anchor=W, justify=LEFT)
        self.lbl_inputpath=     Label(self.root,height=1,width=12,font='Arial 11',bg="white", fg="black",text='input fname   :',anchor=W, justify=LEFT)
        self.ed_indatapath=     Text(self.root, height=1, width=15, font='Arial 11', wrap=WORD)
        self.ed_outdatapath=    Text(self.root, height=1, width=15, font='Arial 11', wrap=WORD)
        self.ed_inputpath=      Text(self.root, height=1, width=15, font='Arial 11', wrap=WORD)
        #training settings
        #self.lbl_trainingsettings=  Label(self.root,height=1,width=12,font='Arial 11',bg="white", fg="black",text='training settings:',anchor=W, justify=LEFT)
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

        self.lbl_losses=            Label(self.root,height=1,width=12,font='Arial 11',bg="white", fg="black",text='',anchor=W, justify=LEFT)

        #self.ed_ns_l1_neurons=     Text(self.root,height=1,width=5,font='Arial 11', wrap=WORD)
        #self.ed_ns_l2_neurons=     Text(self.root,height=1,width=5,font='Arial 11', wrap=WORD)
        #self.ed_ns_l3_neurons=     Text(self.root,height=1,width=5,font='Arial 11', wrap=WORD)
        #self.ed_ns_l4_neurons=     Text(self.root,height=1,width=5,font='Arial 11', wrap=WORD)
        #self.#ed_ns_l5_neurons=     Text(self.root,height=1,width=5,font='Arial 11', wrap=WORD)
        #self.ed_ns_l6_neurons=     Text(self.root,height=1,width=5,font='Arial 11', wrap=WORD)
        #self.ed_ns_l7_neurons=     Text(self.root,height=1,width=5,font='Arial 11', wrap=WORD)
        #self.lbl_out_label=         Label(self.root, height=1, width=5, font='Arial 11', bg="silver", fg="black", text='out:', anchor=W, justify=LEFT)
        #
        #self.ed_ns_l1_af=     Listbox(self.root,height=5,width=6,selectmode=SINGLE)
        #self.ed_ns_l2_af=     Listbox(self.root,height=5,width=6,selectmode=SINGLE)
        #self.ed_ns_l3_af=     Listbox(self.root,height=5,width=6,selectmode=SINGLE)
        #self.ed_ns_l4_af=     Listbox(self.root,height=5,width=6,selectmode=SINGLE)
        #self.ed_ns_l5_af=     Listbox(self.root,height=5,width=6,selectmode=SINGLE)
        #self.ed_ns_l6_af=     Listbox(self.root,height=5,width=6,selectmode=SINGLE)
        #self.ed_ns_l7_af=     Listbox(self.root,height=5,width=6,selectmode=SINGLE)
        #self.ed_ns_out_af=    Listbox(self.root,height=5,width=6, selectmode=SINGLE)


        self.ed_indatapath.insert(1.0, 'in_data.txt')
        self.ed_outdatapath.insert(1.0, 'out_data.txt')
        self.ed_inputpath.insert(1.0, 'input.txt')

        self.ed_indatapath.bind('<KeyRelease>', self.on_change_path)
        self.ed_outdatapath.bind('<KeyRelease>', self.on_change_path)
        self.ed_inputpath.bind('<KeyRelease>', self.on_change_path)
        #self.ed_indatapath.bind('<Button 1>', self.on_change_path)
        #self.ed_outdatapath.bind('<Button 1>', self.on_change_path)
        #self.ed_inputpath.bind('<Button 1>', self.on_change_path)

        #self.btn_trainingplot.bind('<Button 1>', self.on_click_select_plot)
        #self.btn_testingplot.bind('<Button 1>', self.on_click_select_plot)

        #training settings
        self.ed_trs_epochs      .bind('<KeyRelease>', lambda event, u_index='epochs'        , format=int:self.on_change_settings(event, u_index, format))
        self.ed_trs_stoperror   .bind('<KeyRelease>', lambda event, u_index='stop_error'     , format=float:self.on_change_settings(event, u_index, format))
        self.ed_trs_ls          .bind('<KeyRelease>', lambda event, u_index='ls'             , format=float:self.on_change_settings(event, u_index, format))
        self.ed_trs_l1          .bind('<KeyRelease>', lambda event, u_index='l1'             , format=float:self.on_change_settings(event, u_index, format))
        self.ed_trs_l2          .bind('<KeyRelease>', lambda event, u_index='l2'             , format=float:self.on_change_settings(event, u_index, format))
        self.ed_trs_droprate    .bind('<KeyRelease>', lambda event, u_index='drop_rate'      , format=float:self.on_change_settings(event, u_index, format))
        self.ed_trs_ovf_epochs  .bind('<KeyRelease>', lambda event, u_index='overfit_epochs', format=int:self.on_change_settings(event, u_index, format))

        self.ed_trs_epochs      .bind('<FocusOut>', lambda event, u_index='epochs'        , format=int:self.on_change_settings(event, u_index, format))
        self.ed_trs_stoperror   .bind('<FocusOut>', lambda event, u_index='stop_error'     , format=float:self.on_change_settings(event, u_index, format))
        self.ed_trs_ls          .bind('<FocusOut>', lambda event, u_index='ls'             , format=float:self.on_change_settings(event, u_index, format))
        self.ed_trs_l1          .bind('<FocusOut>', lambda event, u_index='l1'             , format=float:self.on_change_settings(event, u_index, format))
        self.ed_trs_l2          .bind('<FocusOut>', lambda event, u_index='l2'             , format=float:self.on_change_settings(event, u_index, format))
        self.ed_trs_droprate    .bind('<FocusOut>', lambda event, u_index='drop_rate'      , format=float:self.on_change_settings(event, u_index, format))
        self.ed_trs_ovf_epochs  .bind('<FocusOut>', lambda event, u_index='overfit_epochs', format=int:self.on_change_settings(event, u_index, format))

        self.btn_train          .bind('<Button 1>', self.on_click_train_btn)
        self.btn_run            .bind('<Button 1>', self.on_click_run_btn)
        self.btn_reloaddata     .bind('<Button 1>', self.on_click_reloaddata_btn)
        self.btn_saveplot       .bind('<Button 1>', self.on_click_saveplot_btn)


        self.ed_indatapath      .place(x=140, y=10)
        self.ed_outdatapath     .place(x=140, y=40)
        self.ed_inputpath       .place(x=140, y=70)

        self.lbl_reloaddata.place(x=10, y=100)
        self.btn_reloaddata.place(x=140, y=100)
        self.lbl_train.place(x=10, y=130)
        self.btn_train.place(x=140, y=130)
        self.lbl_run.place(x=10, y=160)
        self.btn_run.place(x=140, y=160)

        self.btn_trainingplot.place(x=270,y=10)
        self.btn_testingplot.place(x=400,y=10)
        self.lbl_losses.place(x=550,y=10)
        self.btn_saveplot.place(x=700,y=10)
        self.frm_training.place(x=270, y=40)
        self.lbl_indatapath.place(x=10, y=10)
        self.lbl_outdatapath.place(x=10, y=40)
        self.lbl_inputpath.place(x=10, y=70)
        #self.lbl_trainingsettings.place(x=10, y=150)

        #self.lbl_trainingsettings   .place(x=10, y=170)
        self.lbl_trs_epochs        .place(x=10, y=200)
        self.lbl_trs_stoperror      .place(x=10, y=230)
        self.lbl_trs_ovf_epochs     .place(x=10, y=260)
        self.lbl_trs_ls             .place(x=10, y=290)
        self.lbl_trs_l1             .place(x=10, y=320)
        self.lbl_trs_l2             .place(x=10, y=350)
        self.lbl_trs_droprate       .place(x=10, y=380)

        self.ed_trs_epochs         .place(x=140, y=200)
        self.ed_trs_stoperror       .place(x=140, y=230)
        self.ed_trs_ovf_epochs      .place(x=140, y=260)
        self.ed_trs_ls              .place(x=140, y=290)
        self.ed_trs_l1              .place(x=140, y=320)
        self.ed_trs_l2              .place(x=140, y=350)
        self.ed_trs_droprate        .place(x=140, y=380)

        #structure ui
        #self.list_ns_layers         =Listbox(self.root,height=5,width=15,selectmode=SINGLE)
        #self.layers_list            =[1,2,3,4,5,6,7]
        #self.af=['elu',
        #         'sigmoid',
        #         'tanh',
        #         'relu'
        #         ]
        #self.af_func=[tf.nn.elu,
        #         tf.nn.sigmoid,
        #         tf.nn.tanh,
        #         tf.nn.relu
        #         ]
        #
        #self.layers=[
        #    self.ed_ns_l1_neurons,
        #    self.ed_ns_l2_neurons,
        #    self.ed_ns_l3_neurons,
        #    self.ed_ns_l4_neurons,
        #    self.ed_ns_l5_neurons,
        #    self.ed_ns_l6_neurons,
        #    self.ed_ns_l7_neurons
        #]
        #self.layers_af=[self.ed_ns_l1_af,
        #                self.ed_ns_l2_af,
        #                self.ed_ns_l3_af,
        #                self.ed_ns_l4_af,
        #                self.ed_ns_l5_af,
        #                self.ed_ns_l6_af,
        #                self.ed_ns_l7_af
        #                ]
        #
        #for i in self.layers_list:
        #    self.list_ns_layers.insert(END,i)
        #for i in self.layers_af:
        #    for j in self.af:
        #        i.insert(END,j)
        #for j in self.af:
        #    self.ed_ns_out_af.insert(END,j)
        #
        #self.list_ns_layers.place(x=10, y=430)
        #self.list_ns_layers.bind('<Button 1>', self.on_change_layers_count)

    def load_data(self):
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

    def init_plots(self):
        #plot
        self.trainingfig = Figure(figsize=(7, 5), dpi=70)
        self.trainingplot=self.trainingfig.add_subplot(111)
        self.trainingcanvas = FigureCanvasTkAgg(self.trainingfig, master=self.frm_training)  # A tk.DrawingArea.
        self.trainingcanvas.get_tk_widget().pack(expand=True)
        self.trainingani=animation.FuncAnimation(self.trainingfig, self.thread_draw, interval=1000)

        self.testingfig = Figure(figsize=(7, 5), dpi=70)
        self.testingplot=self.testingfig.add_subplot(111)
        self.testingcanvas = FigureCanvasTkAgg(self.testingfig, master=self.frm_testing)  # A tk.DrawingArea.
        self.testingcanvas.get_tk_widget().pack(expand=True)
        self.testingani=animation.FuncAnimation(self.testingfig, self.thread_draw2, interval=1000)




    def __init__(self):
        self.init_interface()
        self.init_plots()
        self.init_settings()
        self.save_settings()

        #load_data
        self.try_load_data()

        #tf && model
        self.init_model_name()
        self.init_model_variables()
        self.load_model()
        self.update_error()

        #run
        self.root.mainloop()
        self.save_settings()
        #self.root.withdraw()



z = app()
#z.ani = animation.FuncAnimation(z.fig, drawthread, interval=1000)
#z.root.mainloop()
