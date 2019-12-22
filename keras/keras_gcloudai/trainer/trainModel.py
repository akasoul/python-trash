import numpy as np
from keras import optimizers, regularizers, callbacks, models, backend
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPool1D, Flatten, LSTM
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tensorflow import io
import argparse
import os
import random
from datetime import datetime


modelName = "model.h5"
useSettingsFile=False
testEnable=True

Preprocessing_Min = 0.0
Preprocessing_Max = 1.0
TestSizePercent = 0.2
BatchMod = 0.05
MaxBatchSize = 3000000000

DISABLE_LOG=True
ENABLE_TRAINING_LOG=False

layersNames = np.array(["conv1d", "dense", "max_pooling1d", "flatten", "lst"])
layersShortNames = np.array(["c1", "d", "p", "f", "lst"])


class historyCallback(callbacks.Callback):

    def initArrays2(self, _loss, _val_loss, _acc, _val_acc):
        self.loss = np.array([_loss], dtype=float)
        self.val_loss = np.array([_val_loss], dtype=float)
        self.acc = np.array([_acc], dtype=float)
        self.val_acc = np.array([_val_acc], dtype=float)

    def initArrays(self, _loss, _acc):
        self.loss = np.array([_loss], dtype=float)
        self.acc = np.array([_acc], dtype=float)

    def copyToGCS(self,inputPath,outputPath):
        with open(inputPath, mode='rb') as input_f:
            with io.gfile.GFile(outputPath, mode='w+')  as output_f:
                output_f.write(input_f.read())

    #metrics: train_acc,val_acc,full_acc,train_loss,val_loss,full_loss
    def initSettings(self,_modelName,_metrics,_ovfEpochs,_reductionEpochs,_reductionKoef,_logDir,minEpochsBetweenSavingModel=0):
        self.modelName=_modelName
        self.metrics=_metrics
        self.ovfEpochs=_ovfEpochs
        self.reductionEpochs=_reductionEpochs
        self.reductionKoef=_reductionKoef
        self.ovfCounter=0
        self.reductionCounter=0
        self.save=False
        self.bestAcc=0
        self.bestValAcc=0
        self.bestEpoch=0

        self.minEpochsBetweenSavingModel=minEpochsBetweenSavingModel
        self.bestLoss=999999999
        self.bestValLoss=999999999

        self.logDir=_logDir

    def initData(self,xData,yData,nDataSize,nInputs,nOutputs):
        self.X=xData
        self.Y=yData
        self.nDataSize=nDataSize
        self.nInputs=nInputs
        self.nOutputs=nOutputs


    def threadTest(self,epoch):
        try:
            import matplotlib.pyplot as plt
        except:
            return

        if(not testEnable):
            return
        if not os.path.isdir(self.logDir+'/training_marks/'):
            os.makedirs(self.logDir+'/training_marks/')

        prediction = self.model.predict(x=np.reshape(self.X, [self.nDataSize, self.nInputs, 1]))
        target=self.Y

        testingfig = plt.figure(num='Testing plot', figsize=(16, 9), dpi=100)
        testingplot = testingfig.add_subplot(111)
        testingplot.plot(target, linewidth=0.05, color='b')
        testingplot.plot(prediction, linewidth=0.05, color='r')
        testingfig.savefig(fname=self.logDir+'/training_marks/'+str(epoch))
        plt.close(testingfig)

    def on_epoch_end(self, epoch, logs=None):


        try:
            self._loss = logs.get('loss')
        except:
            pass

        try:
            self._val_loss = logs.get('val_loss')
        except:
            pass

        try:
            self._acc = logs.get('accuracy')
        except:
            pass

        try:
            self._val_acc = logs.get('val_accuracy')
        except:
            pass

        if(self._acc==None or self._val_acc==None):
            try:
                self._acc = logs.get('acc')
            except:
                pass

            try:
                self._val_acc = logs.get('val_acc')
            except:
                pass

        epoch = epoch + 1

        try:
            self.loss = np.append(self.loss, self._loss)
        except:
            self.loss = np.array([self._loss], dtype=float)

        try:
            self.acc = np.append(self.acc, self._acc)
        except:
            self.acc = np.array([self._acc], dtype=float)


        if(self.metrics=='val_acc' or self.metrics=='val_loss' or self.metrics=='full_acc'  or self.metrics=='full_loss'):
            try:
                self.val_loss = np.append(self.val_loss, self._val_loss)
            except:
                self.val_loss = np.array([self._val_loss], dtype=float)

            try:
                self.val_acc = np.append(self.val_acc, self._val_acc)
            except:
                self.val_acc = np.array([self._val_acc], dtype=float)




        if(self.metrics=='train_acc'):
            if(self._acc>=self.bestAcc):
                print("acc improved {0:6f} -> {1:6f}".format(self.bestAcc,self._acc))
                self.ovfCounter=0
                self.reductionCounter=0
                if(epoch-self.bestEpoch>self.minEpochsBetweenSavingModel):
                    self.model.save_weights(self.modelName)
                    self.bestEpoch=epoch
                    self.threadTest(epoch)
                self.bestAcc=self._acc
            else:
                self.ovfCounter+=1
                self.reductionCounter+=1

        if(self.metrics=='val_acc'):
            if(self._val_acc>=self.bestAccVal):
                print("val_acc improved {0:6f} -> {1:6f}".format(self.bestValAcc,self._val_acc))
                self.ovfCounter=0
                self.reductionCounter=0
                if(epoch-self.bestEpoch>self.minEpochsBetweenSavingModel):
                    self.model.save_weights(self.modelName)
                    self.bestEpoch=epoch
                    self.threadTest(epoch)
                self.bestValAcc=self._val_acc
            else:
                self.ovfCounter+=1
                self.reductionCounter+=1

        if(self.metrics=='full_acc'):
            if(self._acc>=self.bestAcc and self._val_acc>=self.bestValAcc):
                print("acc improved {0:6f} -> {1:6f}".format(self.bestAcc,self._acc))
                print("val_acc improved {0:6f} -> {1:6f}".format(self.bestValAcc,self._val_acc))
                self.ovfCounter=0
                self.reductionCounter=0
                if(epoch-self.bestEpoch>self.minEpochsBetweenSavingModel):
                    self.model.save_weights(self.modelName)
                    self.bestEpoch=epoch
                    self.threadTest(epoch)
                self.bestAcc=self._acc
                self.bestValAcc=self._val_acc
            else:
                self.ovfCounter+=1
                self.reductionCounter+=1

        if(self.metrics=='train_loss'):
            if(self._loss<=self.bestLoss):
                print("loss improved {0:6f} -> {1:6f}".format(self.bestLoss,self._loss))
                self.ovfCounter=0
                self.reductionCounter=0
                if(epoch-self.bestEpoch>self.minEpochsBetweenSavingModel):
                    self.model.save_weights(self.modelName)
                    self.bestEpoch=epoch
                    self.threadTest(epoch)
                self.bestLoss=self._loss
            else:
                self.ovfCounter+=1
                self.reductionCounter+=1

        if(self.metrics=='val_loss'):
            if(self._val_loss<=self.bestValLoss):
                print("val_loss improved {0:6f} -> {1:6f}".format(self.bestValLoss,self._val_loss))
                self.ovfCounter=0
                self.reductionCounter=0
                if(epoch-self.bestEpoch>self.minEpochsBetweenSavingModel):
                    self.model.save_weights(self.modelName)
                    self.bestEpoch=epoch
                    self.threadTest(epoch)
                self.bestValLoss=self._val_loss
            else:
                self.ovfCounter+=1
                self.reductionCounter+=1

        if(self.metrics=='full_loss'):
            if(self._loss<=self.bestLoss and self._val_loss<=self.bestValLoss):
                print("loss improved {0:6f} -> {1:6f}".format(self.bestLoss,self._loss))
                print("val_loss improved {0:6f} -> {1:6f}".format(self.bestValLoss,self._val_loss))
                self.ovfCounter=0
                self.reductionCounter=0
                if(epoch-self.bestEpoch>self.minEpochsBetweenSavingModel):
                    self.model.save_weights(self.modelName)
                    self.bestEpoch=epoch
                    self.threadTest(epoch)
                self.bestLoss=self._loss
                self.bestValLoss=self._val_loss
            else:
                self.ovfCounter+=1
                self.reductionCounter+=1


        if(self.reductionCounter>=self.reductionEpochs):
            old_lr=backend.get_value(self.model.optimizer.lr)
            new_lr=old_lr*self.reductionKoef
            backend.set_value(self.model.optimizer.lr,new_lr)
            print("learning rate reduced {0:5f} -> {1:5f}".format(old_lr,new_lr) )
            self.threadTest(epoch)

            self.reductionCounter=0

        if(self.ovfCounter>=self.ovfEpochs):
            self.model_stop_training=True



class app:


    def initModel(self):

        # model
        kernel_init = 'glorot_uniform'
        bias_init = 'zeros'
        kernel_reg = regularizers.l1_l2(l1=self.settings['l1'], l2=self.settings['l2'])
        bias_reg = regularizers.l1_l2(l1=self.settings['l1'], l2=self.settings['l2'])
        activity_reg = regularizers.l1_l2(l1=self.settings['l1'], l2=self.settings['l2'])
        kernel_size = 5
        filters = 5
        model = Sequential()

        #model.add(Flatten(input_shape=(self.nInputs, 1)))
        model.add(Conv1D(kernel_size=kernel_size, filters=20, activation='relu',input_shape=(self.nInputs, 1),
                   padding="same",
                   kernel_initializer=kernel_init,
                   bias_initializer=bias_init,
                   bias_regularizer=bias_reg,
                   kernel_regularizer=kernel_reg,
                   ))
        model.add(Dropout(self.settings['drop_rate']))
        model.add(MaxPool1D(pool_size=(3)))  # , strides=(1)))


        model.add(Conv1D(kernel_size=kernel_size, filters=20, activation='relu',
                         padding="same",
                         kernel_initializer=kernel_init,
                         bias_initializer=bias_init,
                         bias_regularizer=bias_reg,
                         kernel_regularizer=kernel_reg,
                         ))
        model.add(Dropout(self.settings['drop_rate']))
        model.add(MaxPool1D(pool_size=(3)))  # , strides=(1)))


        model.add(Conv1D(kernel_size=kernel_size, filters=20, activation='relu', padding="same",
                         kernel_initializer=kernel_init,
                         bias_initializer=bias_init,
                         bias_regularizer=bias_reg,
                         kernel_regularizer=kernel_reg,
                         ))
        model.add(Dropout(self.settings['drop_rate']))
        model.add(MaxPool1D(pool_size=(3)))  # , strides=(1)))


        model.add(Conv1D(kernel_size=kernel_size, filters=20, activation='relu', padding="same",
                         kernel_initializer=kernel_init,
                         bias_initializer=bias_init,
                         bias_regularizer=bias_reg,
                         kernel_regularizer=kernel_reg,
                         ))
        model.add(Dropout(self.settings['drop_rate']))
        model.add(MaxPool1D(pool_size=(3)))  # , strides=(1)))




        model.add(Flatten())


        model.add(Dense(self.nOutputs))

        optimizer=None
        #optimizer = optimizers.Adam(lr=self.settings['ls'], beta_1=0.9, beta_2=0.999, decay=0.0, amsgrad=False)
        try:
            optimizer = optimizers.RMSprop(learning_rate=self.settings['ls'], rho=0.9)
        except:
            optimizer = optimizers.RMSprop(lr=self.settings['ls'], rho=0.9)

        model.compile(
             loss='mean_squared_error',
            #loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
        print(model.summary())

        sName = ""
        sName+=str(self.nInputs)
        sName+='.'
        for i in model.layers:
            for j in range(0, layersNames.size):
                if (i.name.find(layersNames[j]) != -1):
                    #for k in i.input_shape:
                    #    if (k != None):
                    #        sName += str(k)
                    #        sName += "."
                    sName += layersShortNames[j]
                    sName += "."
        sName+=str(self.nOutputs)
        self.setModelName(sName)
        #if (os.path.isfile(self.job_dir+self.model_name)):
        #    model.load_weights(self.model_name)
        return model





    def setModelName(self, name):
        self.model_name = name
        self.model_name += ".h5"






    def threadTrain(self):
        self.training_is_launched = True

        backend.reset_uids()
        backend.clear_session()

        model = self.initModel()
        self.log('model initialized')

        score_train = model.evaluate(self.X_train, self.Y_train)  # , batch_size=500)
        score_test=None
        if(self.eval_size>0.0):
            score_test = model.evaluate(self.X_test, self.Y_test)  # , batch_size=500)

        # self.historyCallback.loss=np.array[score_train[0]]
        # self.historyCallback.acc=np.array[score_train[1]]
        # self.historyCallback.val_loss=np.array[score_test[0]]
        # self.historyCallback.val_acc=np.array[score_test[1]]

        #if (self.settings['metrics'] == 0):
        #    metr = 'full_acc'
        #if (self.settings['metrics'] == 1):
        #    metr = 'full_loss'


        metr = 'full_loss'
        if(metr=='train_acc' or metr=='train_loss'):
            self.historyCallback.initArrays(score_train[0], score_train[1])
        else:
            self.historyCallback.initArrays2(score_train[0], score_test[0], score_train[1], score_test[1])

        self.historyCallback.initData(self.X,self.Y,self.nDataSize,self.nInputs,self.nOutputs)

        if(self.sLogName==None):
            self.logDir = self.job_dir+"logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        else:
            self.logDir = self.job_dir+"logs/fit/" + self.sLogName

        self.tb_log = callbacks.TensorBoard(log_dir=self.logDir, histogram_freq=2000)
        self.historyCallback.initSettings(self.job_dir+self.model_name,metr,
                                          self.settings['overfit_epochs'],self.settings['reduction_epochs'],self.settings['ls_reduction_koef'],
                                          self.logDir,10)


        #monitor = None
        #mode = None
        #if (self.settings['metrics'] == 0):
        #    monitor = 'val_acc'
        #    mode = 'max'
        #if (self.settings['metrics'] == 1):
        #    monitor = 'val_loss'
        #    mode = 'min'

        self.callbacks = [
            #callbacks.EarlyStopping(patience=self.settings['overfit_epochs'], monitor=monitor),
            #callbacks.ModelCheckpoint(self.job_dir+self.model_name, monitor=monitor, verbose=1, save_best_only=True,
            #                          save_weights_only=True,
            #                          mode=mode, period=1),
            self.historyCallback,
            self.tb_log
        ]

        #model.fit_generator()
        self.log('start training')

        self.n_batches_train = int(self.nTrainSize * self.settings['batch_size'])
        batchesPerEpoch=1.0/self.n_batches_train
        print("1 epoch = {0} batches".format(batchesPerEpoch))

        if(self.eval_size>0.0):
            model.fit(x=self.X_train, y=self.Y_train, epochs=self.settings['epochs'], verbose=1, batch_size=self.n_batches_train,#shuffle=True,
                  callbacks=self.callbacks,
                  validation_data=(self.X_test, self.Y_test))
        else:
            model.fit(x=self.X_train, y=self.Y_train, epochs=self.settings['epochs'], verbose=1, batch_size=self.n_batches_train,#shuffle=True,
                  callbacks=self.callbacks)

        score = model.evaluate(self.X, self.Y)  # , batch_size=500)

        backend.reset_uids()
        backend.clear_session()


        self.training_is_launched = False

    def threadPredict(self):
        backend.reset_uids()
        backend.clear_session()

        model = self.initModel()
        try:
            model.load_weights(self.job_dir+self.model_name)
        except:
            print('no model')
            return
        else:
            print('model loaded')

        while True:
            if (os.path.isfile(self.job_dir+self.sDataInput1Path)):
                X0=None
                try:
                    X0=np.genfromtxt(self.job_dir+self.sDataInput1Path)
                #    X0 = self.loadFromFileTfGFile(self.job_dir + self.sDataInput1Path)
                #except:
                #    X0 = self.loadFromFile(self.job_dir + self.sDataInput1Path)
                except:
                    pass
                else:
                    X0 = np.float32(X0)
                    X0 = np.reshape(X0, [1, self.nInputs])
                    X0 = self.scaler.transform(X0)
                    X0 = np.reshape(X0, [1, self.nInputs, 1])

                    Y0 = np.zeros(shape=[1, self.nOutputs])
                    Y0 = np.reshape(Y0, [1, self.nOutputs])

                    os.remove(self.job_dir+self.sDataInput1Path)

                    p = model.predict(x=X0)

                    file = open(self.job_dir+'answer.txt', 'w')
                    output = ""
                    for i in range(self.nOutputs):
                        output += str(p[0][i])
                        output += " "
                    file.write(output)
                    file.close()
                    print(output)



    def threadTest(self,count):

        yMin=-1.3
        yMax=1.3

        backend.reset_uids()
        backend.clear_session()

        model = self.initModel()
        try:
            model.load_weights(self.job_dir+self.model_name)
        except:
            print('no model')
            return
        else:
            print('model loaded')

        try:
            import matplotlib.pyplot as plt
        except:
            return

        testingfig=np.empty(dtype=plt.Figure,shape=count)
        testingplot=np.empty(dtype=plt.Subplot,shape=count)
        ctime=datetime.now().strftime("%Y%m%d-%H%M%S")
        sample=random.sample(range(0,self.nDataSize),count)
        figCounter=0
        if not os.path.isdir(self.job_dir+'logs/test/'+ctime):
            os.makedirs(self.job_dir+'logs/test/'+ctime)

        prediction = model.predict(x=np.reshape(self.X, [self.nDataSize, self.nInputs, 1]))
        target=self.Y

        testingfig = plt.figure(num='Testing plot', figsize=(16, 9), dpi=100)
        testingplot = testingfig.add_subplot(111)
        if (yMin != 0 and yMax != 0):
            plt.ylim(yMin, yMax)
        testingplot.plot(target, linewidth=0.05, color='b')
        testingplot.plot(prediction, linewidth=0.05, color='r')
        testingfig.savefig(fname=self.job_dir + 'logs/test/' + ctime + '/all')

        for index in range(0,self.nDataSize):

            _y=self.Y[index]
            _p=prediction[index]
            _p=np.reshape(_p,self.nOutputs)


            testingfig = plt.figure(num='Testing plot '+str(index),figsize=(10, 7), dpi=80 )
            testingplot = testingfig.add_subplot(111)
            if(yMin!=0 and yMax!=0):
                plt.ylim(yMin, yMax)
            testingplot.plot(_y, linewidth=0.5, color='b')
            testingplot.plot(_p, linewidth=0.5, color='r')
            testingfig.savefig(fname=self.job_dir+'logs/test/'+ctime+'/'+str(index))
            plt.close(testingfig)

        #while True:
        #    pass
        ##plt.ioff()
        #backend.reset_uids()
        #backend.clear_session()







    def setSettings(self,settingName,settingValue):
        for i in self.settings:
            if i==settingName:
                self.settings[i]=self.settingsDtypes[i](settingValue)


    def initSettings(self):

        self.settingsDtypes = {
            'epochs': int,
            'stop_error': float,
            'ls': float,
            'l1': float,
            'l2': float,
            'drop_rate': float,
            'overfit_epochs': int,
            'reduction_epochs': int,
            'ls_reduction_koef':float,
            'metrics': int,
            'batch_size': float
        }
        self.settings = {
            'epochs': 50000,
            'stop_error': 0.0000000001,
            'ls': 0.001,
            'l1': 0.00,
            'l2': 0.00,
            'drop_rate': 0.00,
            'overfit_epochs': 5000,
            'reduction_epochs': 2500,
            'ls_reduction_koef':0.95,
            'metrics': 0,
            'batch_size': 0.1
        }

        self.sDataInputPath="in_data.txt"
        self.sDataInput1Path="input.txt"
        self.sDataOutputPath="out_data.txt"

        self.sLogName=None

        self.training_is_launched = False
        self.run_is_launched = False
        self.saving_is_launched = False
        self.data_is_loaded = False
        self.savepng_is_launched = False
        self.test_model = False
        self.model_is_tested = True
        self.testing_model = False

    def setLogName(self,logName):
        self.sLogName=logName

    def log(self,str):
        if DISABLE_LOG==True:
            return
        logfilename=self.job_dir+"log.txt"
        time=datetime.strftime(datetime.now(), "%Y.%m.%d %H:%M:%S")
        file=None
        try:
            file = open(logfilename, 'a')
        except:
            try:
                file = io.gfile.GFile(logfilename, 'a')
            except:
                try:
                    file = open(logfilename, 'w')
                except:
                    file = io.gfile.GFile(logfilename, 'w')

        if(file!=None):
            file.write(time+' ')
            file.write(str+'\n')
            file.close()


    def loadFromFile(self,filename):
        file = open(filename, 'r')
        strData = file.read()
        strData = strData.split()
        doubleData = np.array(strData, dtype=float)
        dim=int(doubleData.size/self.nDataSize)
        doubleData=np.reshape(doubleData,[self.nDataSize, dim])
        return doubleData

    def loadFromFileTfGFile(self,filename):
        file=io.gfile.GFile(filename,'r')

        #file = open(filename, 'r')
        strData = file.read()
        strData = strData.split()
        doubleData = np.array(strData, dtype=float)
        dim=int(doubleData.size/self.nDataSize)
        doubleData=np.reshape(doubleData,[self.nDataSize, dim])
        return doubleData


    def loadData(self):
        _file = False

        self.X=None
        self.Y=None

        # import data
        try:
            self.X = self.loadFromFileTfGFile(self.job_dir+self.sDataInputPath)
        except:
            self.X = self.loadFromFile(self.job_dir+self.sDataInputPath)

        try:
            self.Y = self.loadFromFileTfGFile(self.job_dir+self.sDataOutputPath)
        except:
            self.Y = self.loadFromFile(self.job_dir+self.sDataOutputPath)

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
        if(self.eval_size>0.0):
            self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y,
                                                                                test_size=self.eval_size, shuffle=True)
            self.nTestSize = self.X_test.shape[0]
        else:
            self.X_train = self.X
            self.Y_train = self.Y

        self.nTrainSize = self.X_train.shape[0]

        self.data_is_loaded = True

        self.losses = 0


        self.X = np.reshape(self.X, [self.nDataSize, self.nInputs, 1])
        self.Y = np.reshape(self.Y, [self.nDataSize, self.nOutputs])
        self.X_train = np.reshape(self.X_train, [self.nTrainSize, self.nInputs, 1])
        self.Y_train = np.reshape(self.Y_train, [self.nTrainSize, self.nOutputs])
        if(self.eval_size>0.0):
            self.X_test = np.reshape(self.X_test, [self.nTestSize, self.nInputs, 1])
            self.Y_test = np.reshape(self.Y_test, [self.nTestSize, self.nOutputs])



    def __init__(self,job_dir,data_size,eval_size):
        self.historyCallback = historyCallback()
        self.job_dir=job_dir
        self.eval_size=float(eval_size)
        self.nDataSize=int(data_size)
        # self.initPlots()
        self.initSettings()

        # loadData
        self.loadData()
        self.log('data loaded')




def main(job_dir,mode,data_size,eval_size,batch_size=None,epochs=None,overfit_epochs=None,reduction_epochs=None,ls_reduction_koef=None,ls=None,l1=None,l2=None,drop_rate=None):#, **args):
    z=app(job_dir,data_size,eval_size)
    #print(mode)
    #if(mode.find('train')>0):
    mode=int(mode)
    if(mode==0):
        if(epochs!=None):
            z.setSettings('epochs',epochs)
        if(overfit_epochs!=None):
            z.setSettings('overfit_epochs',overfit_epochs)
        if(reduction_epochs!=None):
            z.setSettings('reduction_epochs',reduction_epochs)
        if(ls_reduction_koef!=None):
            z.setSettings('ls_reduction_koef',ls_reduction_koef)
        if(ls!=None):
            z.setSettings('ls',ls)
        if(l1!=None):
            z.setSettings('l1',l1)
        if(l2!=None):
            z.setSettings('l2',l2)
        if(drop_rate!=None):
            z.setSettings('drop_rate',drop_rate)
        if(batch_size!=None):
            z.setSettings('batch_size',batch_size)
        print(z.settings)
        z.threadTrain()

    #if(mode.find('test')>0):
    if (mode == 1):
        #r=np.random()
        z.threadTest(z.nDataSize)

    #if(mode.find('predict')>0):
    if (mode == 2):
        z.threadPredict()

    #if(mode.find('optimise')>0):
    if (mode == 3):
        if(batch_size!=None):
            z.setSettings('batch_size',batch_size)

        l1_arr = np.array([0.0, 0.00001, 0.0001, 0.001])
        l2_arr = np.array([0.0, 0.00001, 0.0001, 0.001])
        ls_arr = np.array([0.000001, 0.00001, 0.0001, 0.001])
        dr_array = np.array([0.1, 0.3, 0.5])
        for i in ls_arr:
            for j in dr_array:
                for k in l1_arr:
                    # for l in l2_arr:
                    name = "ls={0} dr={1} l1={2} l2={3}".format(i, j, k, k)
                    z.setLogName(name)
                    z.setSettings('epochs', epochs)
                    z.setSettings('overfit_epochs', overfit_epochs)
                    z.setSettings('reduction_epochs', reduction_epochs)
                    z.setSettings('ls_reduction_koef', ls_reduction_koef)
                    z.setSettings('ls', i)
                    z.setSettings('l1', k)
                    z.setSettings('l2', k)
                    z.setSettings('drop_rate', j)
                    z.threadTrain()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input Arguments
    parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        required=True)
    parser.add_argument(
        '--mode',
        help='Work mode: train, test, optimisation',
        required=True)
    parser.add_argument(
        '--data-size',
        help='Size of train+eval data',
        required=True)
    parser.add_argument(
        '--eval-size',
        help='Eval size = ____ data size',
        required=True)
    parser.add_argument(
        '--batch-size',
        help='Batch size',
        required=True)
    parser.add_argument(
        '--epochs',
        help='Epochs',
        required=False)
    parser.add_argument(
        '--overfit-epochs',
        help='Overfit epochs',
        required=False)
    parser.add_argument(
        '--reduction-epochs',
        help='Reduction epochs',
        required=False)
    parser.add_argument(
        '--ls-reduction-koef',
        help='Reduction factor',
        required=False)
    parser.add_argument(
        '--ls',
        help='Learning speed',
        required=False)
    parser.add_argument(
        '--l1',
        help='L1',
        required=False)
    parser.add_argument(
        '--l2',
        help='L2',
        required=False)
    parser.add_argument(
        '--drop-rate',
        help='Drop rate',
        required=False)

    args = parser.parse_args()
    arguments = args.__dict__



    main(**arguments)


#python3 setup.py sdist bdist_wheel

#--job-dir=C:/Users/Anton/AppData/Roaming/MetaQuotes/Terminal/287469DEA9630EA94D0715D755974F1B/tester/files/jobr/EURUSD/
#--data-size=5000
#--eval-size=0.2
#--epochs=1000
#--overfit-epochs=5000
#--reduction-epochs=50000
#--ls-reduction-koef=0.95
#--ls=0.01
#--l1=0.000
#--l2=0.000
#--drop-rate=0.1

# --mode=0
# --data-size=20000
# --eval-size=0.2
# --batch-size=0.2
# --epochs=100000
# --overfit-epochs=50000
# --reduction-epochs=1000
# --ls-reduction-koef=0.99
# --ls=0.0001
# --l1=0.00001
# --l2=0.00001
# --drop-rate=0.15


#rms drop 0.1
#rms lr 0.001-0.01