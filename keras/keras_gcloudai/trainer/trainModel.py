import numpy as np
from keras import optimizers, regularizers, callbacks, models, backend
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPool1D, Flatten, LSTM
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tensorflow import io,gfile
import argparse
import os
from datetime import datetime


modelName = "model.h5"
useSettingsFile=False

Preprocessing_Min = 0.0
Preprocessing_Max = 1.0
TestSizePercent = 0.2
BatchMod = 0.2
MaxBatchSize = 3000000000

DISABLE_LOG=True
ENABLE_TRAINING_LOG=False

layersNames = np.array(["conv1d", "dense", "max_pooling1d", "flatten", "lst"])
layersShortNames = np.array(["c1d", "d", "mp1d", "fl", "lst"])


class historyCallback(callbacks.Callback):

    def initArrays(self, _loss, _val_loss, _acc, _val_acc):
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
    def initSettings(self,_modelName,_metrics,_ovfEpochs):
        self.modelName=_modelName
        self.metrics=_metrics
        self.ovfEpochs=_ovfEpochs
        self.ovfCounter=0
        self.save=False
        self.bestAcc=0
        self.bestValAcc=0
        self.bestEpoch=0

        self.bestLoss=999999999
        self.bestValLoss=999999999

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get('loss')
        acc = logs.get('acc')
        if(self.metrics=='val_acc' or self.metrics=='val_loss' or self.metrics=='full_acc'  or self.metrics=='full_loss'):
            val_loss = logs.get('val_loss')
            val_acc = logs.get('val_acc')

        epoch = epoch + 1

        try:
            self.loss = np.append(self.loss, loss)
        except:
            self.loss = np.array([loss], dtype=float)

        try:
            self.acc = np.append(self.acc, acc)
        except:
            self.acc = np.array([acc], dtype=float)


        if(self.metrics=='val_acc' or self.metrics=='val_loss' or self.metrics=='full_acc'  or self.metrics=='full_loss'):
            try:
                self.val_loss = np.append(self.val_loss, val_loss)
            except:
                self.val_loss = np.array([val_loss], dtype=float)

            try:
                self.val_acc = np.append(self.val_acc, val_acc)
            except:
                self.val_acc = np.array([val_acc], dtype=float)

        if(self.metrics=='train_acc'):
            if(acc>self.bestAcc):
                print("acc improved {0:6f} -> {1:6f}".format(self.bestAcc,acc))
                self.ovfCounter=0
                self.model.save_weights(self.modelName)
                self.bestAcc=acc
                self.bestEpoch=epoch
            else:
                self.ovfCounter+=1

        if(self.metrics=='val_acc'):
            if(val_acc>self.bestAccVal):
                print("val_acc improved {0:6f} -> {1:6f}".format(self.bestValAcc,val_acc))
                self.ovfCounter=0
                self.model.save_weights(self.modelName)
                self.bestValAcc=val_acc
                self.bestEpoch=epoch
            else:
                self.ovfCounter+=1

        if(self.metrics=='full_acc'):
            if(acc>self.bestAcc and val_acc>self.bestValAcc):
                print("acc improved {0:6f} -> {1:6f}".format(self.bestAcc,acc))
                print("val_acc improved {0:6f} -> {1:6f}".format(self.bestValAcc,val_acc))
                self.ovfCounter=0
                self.model.save_weights(self.modelName)
                self.bestAcc=acc
                self.bestValAcc=val_acc
                self.bestEpoch=epoch
            else:
                self.ovfCounter+=1

        if(self.metrics=='train_loss'):
            if(loss<self.bestLoss):
                print("loss improved {0:6f} -> {1:6f}".format(self.bestLoss,loss))
                self.ovfCounter=0
                self.model.save_weights(self.modelName)
                self.bestLoss=loss
                self.bestEpoch=epoch
            else:
                self.ovfCounter+=1

        if(self.metrics=='val_loss'):
            if(val_loss<self.bestValLoss):
                print("val_loss improved {0:6f} -> {1:6f}".format(self.bestValLoss,val_loss))
                self.ovfCounter=0
                self.model.save_weights(self.modelName)
                self.bestValLoss=val_loss
                self.bestEpoch=epoch
            else:
                self.ovfCounter+=1

        if(self.metrics=='full_loss'):
            if(loss<self.bestLoss and val_loss<self.bestValLoss):
                print("loss improved {0:6f} -> {1:6f}".format(self.bestLoss,loss))
                print("val_loss improved {0:6f} -> {1:6f}".format(self.bestValLoss,val_loss))
                self.ovfCounter=0
                self.model.save_weights(self.modelName)
                self.bestLoss=loss
                self.bestValLoss=val_loss
                self.bestEpoch=epoch
            else:
                self.ovfCounter+=1

        if(ENABLE_TRAINING_LOG):
            if(self.bestEpoch==epoch):
                try:
                    with open('training_temp.txt', 'a') as f:
                        try:
                            f.write("loss;{0:5f};val_loss;{1:5f};acc;{2:5f};val_acc;{3:5f};\n".format(loss,val_loss,acc,val_acc))
                        except:
                            f.write("loss;{0:5f};acc;{1:5f};\n".format(loss,acc))
                        f.close()

                except:
                    with open('training_temp.txt', 'w') as f:
                        try:
                            f.write("loss;{0:5f};val_loss;{1:5f};acc;{2:5f};val_acc;{3:5f};\n".format(loss,val_loss,acc,val_acc))
                        except:
                            f.write("loss;{0:5f};acc;{1:5f};\n".format(loss,acc))
                        f.close()

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
        kernel_size = 10
        filters = 5
        model = Sequential()
        model.add(
            Conv1D(kernel_size=kernel_size, filters=filters, activation='relu', input_shape=(self.nInputs, 1),
                   padding="same",
                   kernel_initializer=kernel_init,
                   bias_initializer=bias_init,
                   bias_regularizer=bias_reg,
                   kernel_regularizer=kernel_reg,
                   # activity_regularizer=activity_reg
                   ))
        model.add(MaxPool1D(pool_size=(5)))  # , strides=(1)))
        model.add(Conv1D(kernel_size=kernel_size, filters=filters, activation='relu', padding="same",
                         kernel_initializer=kernel_init,
                         bias_initializer=bias_init,
                         bias_regularizer=bias_reg,
                         kernel_regularizer=kernel_reg,
                         ))
        model.add(MaxPool1D(pool_size=(10)))  # , strides=(1)))
        model.add(Flatten())

        model.add(Dense(500, activation='relu',
                        kernel_initializer=kernel_init,
                        bias_initializer=bias_init,
                        bias_regularizer=bias_reg,
                        kernel_regularizer=kernel_reg,
                        ))
        model.add(Dropout(self.settings['drop_rate']))

        model.add(Dense(500, activation='relu',
                        kernel_initializer=kernel_init,
                        bias_initializer=bias_init,
                        bias_regularizer=bias_reg,
                        kernel_regularizer=kernel_reg,
                        ))
        model.add(Dropout(self.settings['drop_rate']))

        model.add(Dense(500, activation='relu',
                        kernel_initializer=kernel_init,
                        bias_initializer=bias_init,
                        bias_regularizer=bias_reg,
                        kernel_regularizer=kernel_reg,
                        ))
        model.add(Dropout(self.settings['drop_rate']))


        model.add(Dense(self.nOutputs))
        #model.add(Dense(self.nOutputs,activation='softmax'))
        #model.add(Dense(self.nOutputs, activation='tanh',
        #                kernel_initializer=kernel_init,
        #                bias_initializer=bias_init,
        #                 bias_regularizer=bias_reg,
        #                 kernel_regularizer=kernel_reg
        #                ))


        #optimizer = optimizers.Adam(lr=self.settings['ls'], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False);
        optimizer = optimizers.Adam(lr=self.settings['ls'], beta_1=0.9, beta_2=0.999, decay=0.0, amsgrad=False);

        model.compile(
             loss='mean_squared_error',
            #loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
        print(model.summary())

        sName = ""
        for i in model.layers:
            for j in range(0, layersNames.size):
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
        self.setModelName(sName)
        #if (os.path.isfile(self.job_dir+self.model_name)):
        #    model.load_weights(self.model_name)
        return model



    def initModel2(self):

        # model
        kernel_init = 'glorot_uniform'
        bias_init = 'zeros'
        kernel_reg = regularizers.l1_l2(l1=self.settings['l1'], l2=self.settings['l2'])
        bias_reg = regularizers.l1_l2(l1=self.settings['l1'], l2=self.settings['l2'])
        activity_reg = regularizers.l1_l2(l1=self.settings['l1'], l2=self.settings['l2'])
        kernel_size = 10
        filters = 5
        model = Sequential()


        model.add(Dense(50, activation='relu',input_shape=(self.nInputs, 1),
                        kernel_initializer=kernel_init,
                        bias_initializer=bias_init,
                        bias_regularizer=bias_reg,
                        kernel_regularizer=kernel_reg,
                        ))
        model.add(Dropout(self.settings['drop_rate']))
        model.add(Flatten())

        model.add(Dense(50, activation='relu',
                        kernel_initializer=kernel_init,
                        bias_initializer=bias_init,
                        bias_regularizer=bias_reg,
                        kernel_regularizer=kernel_reg,
                        ))
        model.add(Dropout(self.settings['drop_rate']))

        model.add(Dense(50, activation='relu',
                        kernel_initializer=kernel_init,
                        bias_initializer=bias_init,
                        bias_regularizer=bias_reg,
                        kernel_regularizer=kernel_reg,
                        ))
        model.add(Dropout(self.settings['drop_rate']))


        model.add(Dense(self.nOutputs))


        #optimizer = optimizers.Adam(lr=self.settings['ls'], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False);
        optimizer = optimizers.Adam(lr=self.settings['ls'], beta_1=0.9, beta_2=0.999, decay=0.0, amsgrad=False);

        model.compile(
             loss='mean_squared_error',
            #loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
        #print(model.summary())

        sName = ""
        for i in model.layers:
            for j in range(0, layersNames.size):
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
        self.setModelName(sName)
        print(self.model_name)
        #if (os.path.isfile(self.job_dir+self.model_name)):
        #    model.load_weights(self.model_name)
        return model


    def setModelName(self, name):
        self.model_name = name
        self.model_name += ".h5"

    def loadModel(self):
        try:
            self.model.load_weights(self.job_dir+modelName)
        except:
            print('no model')







    def threadTrain(self):
        self.training_is_launched = True

        #backend.reset_uids()
        #backend.clear_session()

        model = self.initModel2()
        self.log('model initialized')

        score_train = model.evaluate(self.X_train, self.Y_train)  # , batch_size=500)
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


        metr = 'train_loss'
        if(metr=='train_acc' or metr=='train_loss'):
            self.historyCallback.initArrays(score_train[0], score_train[1])
        else:
            self.historyCallback.initArrays(score_train[0], score_test[0], score_train[1], score_test[1])

        self.historyCallback.initSettings(self.job_dir+self.model_name,metr,self.settings['overfit_epochs'])

        logdir=self.job_dir+'/log/fit/' + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tb_log = callbacks.TensorBoard(
            log_dir=logdir,
            histogram_freq=1,
            write_graph=True,
            write_grads=True,
            write_images=True)

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
            self.historyCallback,self.tb_log
        ]

        #model.fit_generator()
        self.log('start training')

        if(self.eval_size>0.0):
            model.fit(x=self.X_train, y=self.Y_train, epochs=self.settings['epochs'], batch_size=self.n_batches_train,verbose=1,
                  callbacks=self.callbacks,
                  validation_data=(self.X_test, self.Y_test))
        else:
            model.fit(x=self.X_train, y=self.Y_train, epochs=self.settings['epochs'], batch_size=self.n_batches_train,verbose=1,
                  callbacks=self.callbacks)

        score = model.evaluate(self.X, self.Y)  # , batch_size=500)

        #backend.reset_uids()
        #backend.clear_session()


        self.training_is_launched = False



    def updateError(self):
        if (self.training_is_launched == True):
            return
        if (self.run_is_launched == True):
            return

        backend.reset_uids()
        backend.clear_session()

        model = self.initModel2()
        score = model.evaluate(self.X, self.Y)#, batch_size=500)

        backend.reset_uids()
        backend.clear_session()


    def loadSettings(self):
        keys = self.settings.keys()
        values = self.settings.values()
        fname = self.job_dir+'settings.txt'
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
                        try:
                            self.settingsUI[i].insert(1.0, a[y + 1:z])
                        except:
                            self.settingsUI[i].set(self.settingsDtypes[i](a[y + 1:z]))
                    else:
                        pass
                else:
                    pass
            else:
                self.settings[i] = self.settingsDefault[i]

        else:
            pass


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
            'metrics': int
        }
        self.settings = {
            'epochs': 50,
            'stop_error': 0.00000001,
            'ls': 0.001,
            'l1': 0.00,
            'l2': 0.00,
            'drop_rate': 0.00,
            'overfit_epochs': 5000,
            'metrics': 0
        }

        self.sDataInputPath="in_data.txt"
        self.sDataInput1Path="input.txt"
        self.sDataOutputPath="out_data.txt"


        self.training_is_launched = False
        self.run_is_launched = False
        self.saving_is_launched = False
        self.data_is_loaded = False
        self.savepng_is_launched = False
        self.test_model = False
        self.model_is_tested = True
        self.testing_model = False

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

        self.n_batches_train = int(self.nTrainSize * BatchMod)

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



def get_message():
    return "Hello World!"

def main(job_dir,data_size,eval_size,epochs=None):#, **args):
    z=app(job_dir,data_size,eval_size)
    if(epochs!=None):
        z.setSettings('epochs',epochs)
    z.threadTrain()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input Arguments
    parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
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
        '--epochs',
        help='Epochs',
        required=False)

    args = parser.parse_args()
    arguments = args.__dict__



    main(**arguments)


#python3 setup.py sdist bdist_wheel