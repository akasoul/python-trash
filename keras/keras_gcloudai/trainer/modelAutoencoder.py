import numpy as np
from keras import Model, optimizers, regularizers, callbacks, models, backend
from keras.utils import plot_model
from keras.models import Sequential,load_model
from keras.layers import Input, Dense, Dropout, Conv1D, MaxPool1D, Flatten, LSTM, concatenate, BatchNormalization, \
    Activation, add, AveragePooling1D, multiply, LeakyReLU, ReLU, ELU, UpSampling1D
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tensorflow import io
import argparse
import os
import random
import threading
import hashlib
from datetime import datetime

modelName = "model.h5"
useSettingsFile = False
testEnable = True

Preprocessing_Min = 0.0
Preprocessing_Max = 1.0
TestSizePercent = 0.2
BatchMod = 0.05
MaxBatchSize = 3000000000

DISABLE_LOG = True

layersNames = np.array(["conv1d", "dense", "max_pooling1d", "flatten", "lst", "concatenate"])
layersShortNames = np.array(["c1", "d", "p", "f", "lst", "ctn"])


class historyCallback(callbacks.Callback):#,callbacks.EarlyStopping):

    def initArrays2(self, _loss, _val_loss, _acc, _val_acc):
        self.loss = np.array([_loss], dtype=float)
        self.val_loss = np.array([_val_loss], dtype=float)
        self.acc = np.array([_acc], dtype=float)
        self.val_acc = np.array([_val_acc], dtype=float)

    def initArrays(self, _loss, _acc):
        self.loss = np.array([_loss], dtype=float)
        self.acc = np.array([_acc], dtype=float)

    def copyToGCS(self, inputPath, outputPath):
        with open(inputPath, mode='rb') as input_f:
            with io.gfile.GFile(outputPath, mode='w+')  as output_f:
                output_f.write(input_f.read())

    # metrics: train_acc,val_acc,full_acc,train_loss,val_loss,full_loss
    def initSettings(self, _modelName, _metrics, _ovfEpochs, _reductionEpochs, _reductionKoef, _logDir, _saveWholeModel,
                     minEpochsBetweenSavingModel=0):
        self.modelName = _modelName
        self.metrics = _metrics
        self.ovfEpochs = _ovfEpochs
        self.reductionEpochs = _reductionEpochs
        self.reductionKoef = _reductionKoef
        self.ovfCounter = 0
        self.reductionCounter = 0
        self.save = False
        self.minEpochsBetweenSavingModel = minEpochsBetweenSavingModel
        self.saveWholeModel=_saveWholeModel

        self.bestEpoch = -minEpochsBetweenSavingModel
        #self.best_weights=self.model.get_weights()
        self.bestLoss = self.loss[0]
        self.bestValLoss = self.val_loss[0]
        self.bestAcc = self.acc[0]
        self.bestValAcc = self.val_acc[0]

        self.logDir = _logDir

    def initData(self, xData, yData, nDataSize, nInputFiles, nOutputFiles):
        self.X = None
        self.Y = None
        self.inputFiles = nInputFiles
        self.outputFiles= nOutputFiles
        for i in range(0, self.inputFiles):
            if i == 0:
                self.X = list([xData[i]['data']])
            else:
                self.X.append(xData[i]['data'])

        for i in range(0, self.outputFiles):
            if i == 0:
                self.Y = list([yData[i]['data']])
            else:
                self.Y.append(yData[i]['data'])



    def threadTest(self, epoch):
        try:
            import matplotlib.pyplot as plt
        except:
            return

        if (not testEnable):
            return
        if not os.path.isdir(self.logDir + '/tests/texts/'):
            os.makedirs(self.logDir + '/tests/texts/')
        if not os.path.isdir(self.logDir + '/tests/images/'):
            os.makedirs(self.logDir + '/tests/images/')

        input = None
        output = None

        for i in range(0, self.inputFiles):
            if i == 0:
                input = list([self.X[i]])
            else:
                input.append(self.X[i])

        for i in range(0, self.outputFiles):
            if i == 0:
                output = list([self.Y[i]])
            else:
                output.append(self.Y[i])



        prediction = self.model.predict(x=input)

        #testingfig = plt.figure(num='Testing plot', figsize=(16, 9), dpi=100)
        #testingplot = testingfig.add_subplot(111)
        #testingplot.plot(output, linewidth=0.05, color='b')
        #testingplot.plot(prediction, linewidth=0.05, color='r')
        #testingfig.savefig(fname=self.logDir + '/tests/images/' + str(epoch))
        #plt.close(testingfig)

        #arrayToFile = np.column_stack((prediction[:, 0], output[:, 0]))

        arrayToFile = np.column_stack((prediction, output[0]))
        arrayToFile = np.reshape(arrayToFile,[arrayToFile.shape[0],arrayToFile.shape[1]])
        np.savetxt(self.logDir + '/tests/texts/' + str(epoch) + ".txt", arrayToFile, delimiter=" ", fmt='%1.3f')

        target = output[0]
        target=np.reshape(target,[target.shape[0], target.shape[1]])
        prediction=np.reshape(prediction, [prediction.shape[0], prediction.shape[1]])

        np.savetxt(self.logDir + '/tests/texts/' + str(epoch) + "target.txt", target, delimiter=" ", fmt='%1.3f')
        np.savetxt(self.logDir + '/tests/texts/' + str(epoch) + "prediction.txt", prediction, delimiter=" ", fmt='%1.3f')





    def on_epoch_end(self, epoch, logs=None):

        logs['lr'] = backend.get_value(self.model.optimizer.lr)

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

        if (self._acc == None or self._val_acc == None):
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

        if (
                self.metrics == 'val_acc' or self.metrics == 'val_loss' or self.metrics == 'full_acc' or self.metrics == 'full_loss'):
            try:
                self.val_loss = np.append(self.val_loss, self._val_loss)
            except:
                self.val_loss = np.array([self._val_loss], dtype=float)

            try:
                self.val_acc = np.append(self.val_acc, self._val_acc)
            except:
                self.val_acc = np.array([self._val_acc], dtype=float)

        if (self.metrics == 'train_acc'):
            if (self._acc > self.bestAcc):
                print("acc improved {0:6f} -> {1:6f}".format(self.bestAcc, self._acc))
                self.ovfCounter = 0
                self.reductionCounter = 0

                if(self.saveWholeModel==True):
                    self.model.save(self.modelName)
                else:
                    self.model.save_weights(self.modelName)
                if (epoch - self.bestEpoch > self.minEpochsBetweenSavingModel):
                    self.bestEpoch = epoch
                    self.threadTest(epoch)
                self.bestAcc = self._acc
            else:
                self.ovfCounter += 1
                self.reductionCounter += 1

        if (self.metrics == 'val_acc'):
            if (self._val_acc > self.bestAccVal):
                print("val_acc improved {0:6f} -> {1:6f}".format(self.bestValAcc, self._val_acc))
                self.ovfCounter = 0
                self.reductionCounter = 0
                self.best_weights = self.model.get_weights()
                if(self.saveWholeModel==True):
                    self.model.save(self.modelName)
                else:
                    self.model.save_weights(self.modelName)
                if (epoch - self.bestEpoch > self.minEpochsBetweenSavingModel):
                    self.bestEpoch = epoch
                    self.threadTest(epoch)
                self.bestValAcc = self._val_acc
            else:
                self.ovfCounter += 1
                self.reductionCounter += 1

        if (self.metrics == 'full_acc'):
            if (self._acc > self.bestAcc and self._val_acc >= self.bestValAcc):
                print("acc improved {0:6f} -> {1:6f}".format(self.bestAcc, self._acc))
                print("val_acc improved {0:6f} -> {1:6f}".format(self.bestValAcc, self._val_acc))
                self.ovfCounter = 0
                self.reductionCounter = 0
                self.best_weights = self.model.get_weights()
                if(self.saveWholeModel==True):
                    self.model.save(self.modelName)
                else:
                    self.model.save_weights(self.modelName)
                if (epoch - self.bestEpoch > self.minEpochsBetweenSavingModel):
                    self.bestEpoch = epoch
                    self.threadTest(epoch)
                self.bestAcc = self._acc
                self.bestValAcc = self._val_acc
            else:
                self.ovfCounter += 1
                self.reductionCounter += 1

        if (self.metrics == 'train_loss'):
            if (self._loss < self.bestLoss):
                print("loss improved {0:6f} -> {1:6f}".format(self.bestLoss, self._loss))
                self.ovfCounter = 0
                self.reductionCounter = 0
                self.best_weights = self.model.get_weights()
                if(self.saveWholeModel==True):
                    self.model.save(self.modelName)
                else:
                    self.model.save_weights(self.modelName)
                if (epoch - self.bestEpoch > self.minEpochsBetweenSavingModel):
                    self.bestEpoch = epoch
                    self.threadTest(epoch)
                self.bestLoss = self._loss
            else:
                self.ovfCounter += 1
                self.reductionCounter += 1

        if (self.metrics == 'val_loss'):
            if (self._val_loss < self.bestValLoss):
                print("val_loss improved {0:6f} -> {1:6f}".format(self.bestValLoss, self._val_loss))
                self.ovfCounter = 0
                self.reductionCounter = 0
                self.best_weights = self.model.get_weights()
                if(self.saveWholeModel==True):
                    self.model.save(self.modelName)
                else:
                    self.model.save_weights(self.modelName)
                if (epoch - self.bestEpoch > self.minEpochsBetweenSavingModel):
                    self.bestEpoch = epoch
                    self.threadTest(epoch)
                self.bestValLoss = self._val_loss
            else:
                self.ovfCounter += 1
                self.reductionCounter += 1

        if (self.metrics == 'full_loss'):
            if (self._loss < self.bestLoss and self._val_loss <= self.bestValLoss):
                print("loss improved {0:6f} -> {1:6f}".format(self.bestLoss, self._loss))
                print("val_loss improved {0:6f} -> {1:6f}".format(self.bestValLoss, self._val_loss))
                self.ovfCounter = 0
                self.reductionCounter = 0
                self.best_weights = self.model.get_weights()
                if(self.saveWholeModel==True):
                    self.model.save(self.modelName)
                else:
                    self.model.save_weights(self.modelName)
                if (epoch - self.bestEpoch > self.minEpochsBetweenSavingModel):
                    self.bestEpoch = epoch
                    self.threadTest(epoch)
                self.bestLoss = self._loss
                self.bestValLoss = self._val_loss
            else:
                self.ovfCounter += 1
                self.reductionCounter += 1

        if (self.reductionCounter >= self.reductionEpochs):
            old_lr = backend.get_value(self.model.optimizer.lr)
            new_lr = old_lr * self.reductionKoef
            backend.set_value(self.model.optimizer.lr, new_lr)
            #if (self.saveWholeModel == True):
            #    self.model = load_model(self.modelName)
            #else:
            #    self.model.load_weights(self.modelName)
            try:
                self.model.set_weights(self.best_weights)
            except:
                pass
            else:
                print("learning rate reduced {0:5f} -> {1:5f}".format(old_lr, new_lr))

            self.reductionCounter = 0

        if (self.ovfCounter >= self.ovfEpochs):
            self.model_stop_training = True


class elements:
    def __init__(self, settings):
        self.settings = settings

    def conv1DMPLayer(self, input, kernel_size, filters_count, pool_size, activation, kernel_init, bias_init,
                      inputShape=0):
        kernel_reg = regularizers.l1_l2(l1=self.settings['l1'], l2=self.settings['l2'])
        bias_reg = regularizers.l1_l2(l1=self.settings['l1'], l2=self.settings['l2'])

        if (inputShape != 0):
            output = Conv1D(kernel_size=kernel_size, filters=filters_count, activation=activation,
                            input_shape=(inputShape, 1),
                            padding="same",
                            kernel_initializer=kernel_init,
                            bias_initializer=bias_init,
                            bias_regularizer=bias_reg,
                            kernel_regularizer=kernel_reg,
                            )(input)
            output = Dropout(self.settings['drop_rate'])(output)
            output = MaxPool1D(pool_size=(pool_size))(output)
            output = BatchNormalization()(output)
            return output
        else:
            output = Conv1D(kernel_size=kernel_size, filters=filters_count, activation=activation,
                            padding="same",
                            kernel_initializer=kernel_init,
                            bias_initializer=bias_init,
                            bias_regularizer=bias_reg,
                            kernel_regularizer=kernel_reg,
                            )(input)
            output = Dropout(self.settings['drop_rate'])(output)
            output = MaxPool1D(pool_size=(pool_size))(output)
            output = BatchNormalization()(output)
            return output

    def conv1DLayer(self, input, kernel_size, strides, activation, kernel_init, bias_init, inputShape=0):
        kernel_reg = regularizers.l1_l2(l1=self.settings['l1'], l2=self.settings['l2'])
        bias_reg = regularizers.l1_l2(l1=self.settings['l1'], l2=self.settings['l2'])

        if (inputShape != 0):
            output = Conv1D(kernel_size=kernel_size, filters=1, activation=activation,
                            input_shape=(inputShape, 1),
                            padding="valid",
                            strides=strides,
                            kernel_initializer=kernel_init,
                            bias_initializer=bias_init,
                            bias_regularizer=bias_reg,
                            kernel_regularizer=kernel_reg,
                            )(input)
            output = Dropout(self.settings['drop_rate'])(output)
            output = BatchNormalization()(output)
            return output
        else:
            output = Conv1D(kernel_size=kernel_size, filters=1, activation=activation,
                            padding="valid",
                            strides=strides,
                            kernel_initializer=kernel_init,
                            bias_initializer=bias_init,
                            bias_regularizer=bias_reg,
                            kernel_regularizer=kernel_reg,
                            )(input)
            output = Dropout(self.settings['drop_rate'])(output)
            output = BatchNormalization()(output)
            return output

    def conv1DResLayer(self, input, kernel_size, filters, convolutions, activation, kernel_init, bias_init,
                       pooling=False, pool_size=1, inputShape=0):
        kernel_reg = regularizers.l1_l2(l1=self.settings['l1'], l2=self.settings['l2'])
        bias_reg = regularizers.l1_l2(l1=self.settings['l1'], l2=self.settings['l2'])

        if (inputShape != 0):
            x = input
            output = input
            for i in range(0, convolutions):
                output = Conv1D(kernel_size=kernel_size, filters=filters, activation=None,
                                input_shape=(inputShape, 1),
                                padding="same",
                                kernel_initializer=kernel_init,
                                bias_initializer=bias_init,
                                bias_regularizer=bias_reg,
                                kernel_regularizer=kernel_reg,
                                )(output)
                output = BatchNormalization()(output)
                output = Activation(activation=activation)(output)

            output = Conv1D(kernel_size=kernel_size, filters=filters, activation=None,
                            padding="same",
                            kernel_initializer=kernel_init,
                            bias_initializer=bias_init,
                            bias_regularizer=bias_reg,
                            kernel_regularizer=kernel_reg,
                            )(output)
            output = BatchNormalization()(output)
            output = add([output, x])
            output = Activation(activation=activation)(output)
            # if(pooling==True):
            #    output=MaxPool1D(pool_size=pool_size)(output)
            return output
        else:
            x = input
            output = input
            for i in range(0, convolutions):
                output = Conv1D(kernel_size=kernel_size, filters=filters, activation=None,
                                padding="same",
                                kernel_initializer=kernel_init,
                                bias_initializer=bias_init,
                                bias_regularizer=bias_reg,
                                kernel_regularizer=kernel_reg,
                                )(output)
                output = BatchNormalization()(output)
                output = Activation(activation=activation)(output)
            output = Conv1D(kernel_size=kernel_size, filters=filters, activation=None,
                            padding="same",
                            kernel_initializer=kernel_init,
                            bias_initializer=bias_init,
                            bias_regularizer=bias_reg,
                            kernel_regularizer=kernel_reg,
                            )(output)
            output = BatchNormalization()(output)
            output = add([output, x])
            output = Activation(activation=activation)(output)
            # if(pooling==True):
            #    output=MaxPool1D(pool_size=pool_size)(output)
            return output

    def resUnit(self, input, filters, activation, kernel_init, bias_init, inputShape=0):
        kernel_reg = regularizers.l1_l2(l1=self.settings['l1'], l2=self.settings['l2'])
        bias_reg = regularizers.l1_l2(l1=self.settings['l1'], l2=self.settings['l2'])

        useActivationInside = False

        def conv1dinput(kernel):

            return Conv1D(kernel_size=kernel, filters=filters, activation=None,
                          input_shape=(inputShape, 1),
                          padding="same",
                          kernel_initializer=kernel_init,
                          bias_initializer=bias_init,
                          bias_regularizer=bias_reg,
                          kernel_regularizer=kernel_reg,
                          )

        def conv1d(kernel):
            return Conv1D(kernel_size=kernel, filters=filters, activation=None,
                          padding="same",
                          kernel_initializer=kernel_init,
                          bias_initializer=bias_init,
                          bias_regularizer=bias_reg,
                          kernel_regularizer=kernel_reg,
                          )

        t0 = input
        t1 = input
        t2 = input
        t3 = input
        if (inputShape != 0):
            t1 = conv1dinput(1)(t1)
            t1 = BatchNormalization()(t1)
            if (useActivationInside == True):
                t1 = Activation(activation=activation)(t1)

            t2 = conv1dinput(1)(t2)
            t2 = BatchNormalization()(t2)
            if (useActivationInside == True):
                t2 = Activation(activation=activation)(t2)

            t3 = conv1dinput(1)(t3)
            t3 = BatchNormalization()(t3)
            if (useActivationInside == True):
                t3 = Activation(activation=activation)(t3)
        else:
            t1 = conv1d(1)(t1)
            t1 = BatchNormalization()(t1)
            if (useActivationInside == True):
                t1 = Activation(activation=activation)(t1)

            t2 = conv1d(1)(t2)
            t2 = BatchNormalization()(t2)
            if (useActivationInside == True):
                t2 = Activation(activation=activation)(t2)

            t3 = conv1d(1)(t3)
            t3 = BatchNormalization()(t3)
            if (useActivationInside == True):
                t3 = Activation(activation=activation)(t3)

        t2 = conv1d(3)(t2)
        t2 = BatchNormalization()(t2)
        if (useActivationInside == True):
            t2 = Activation(activation=activation)(t2)

        t3 = conv1d(3)(t3)
        t3 = BatchNormalization()(t3)
        if (useActivationInside == True):
            t3 = Activation(activation=activation)(t3)

        t3 = conv1d(5)(t3)
        t3 = BatchNormalization()(t3)
        if (useActivationInside == True):
            t3 = Activation(activation=activation)(t3)

        t = add([t0, t1, t2, t3])
        t = Activation(activation=activation)(t)
        return t



    def resUnit4(self, input, filters, activation, kernel_init, bias_init, inputShape=0):
        kernel_reg = regularizers.l2(self.settings['l2'])
        bias_reg = regularizers.l2(self.settings['l2'])

        def conv1dinput(kernel):

            return Conv1D(kernel_size=kernel, filters=filters, activation=None,
                          input_shape=(inputShape, 1),
                          padding="same",
                          kernel_initializer=kernel_init,
                          bias_initializer=bias_init,
                          bias_regularizer=bias_reg,
                          kernel_regularizer=kernel_reg,
                          )

        def conv1d(kernel):
            return Conv1D(kernel_size=kernel, filters=filters, activation=None,
                          padding="same",
                          kernel_initializer=kernel_init,
                          bias_initializer=bias_init,
                          bias_regularizer=bias_reg,
                          kernel_regularizer=kernel_reg,
                          )

        t0 = input
        t1 = input
        t2 = input
        t3 = input
        if (inputShape != 0):
            t1 = conv1dinput(5)(t1)
            t2 = conv1dinput(5)(t2)
            t3 = conv1dinput(5)(t3)
        else:
            t1 = conv1d(5)(t1)
            t2 = conv1d(5)(t2)
            t3 = conv1d(5)(t3)

        t2 = conv1d(3)(t2)
        t3 = conv1d(3)(t3)

        t3 = conv1d(1)(t3)

        output = add([t0, t1, t2, t3])

        output = activation(output)
        return output

    def resUnit2(self, input, filters, kernel_size, activation, kernel_init, bias_init, inputShape=0, useBatchnorm=False):
        kernel_reg = regularizers.l2(self.settings['l2'])
        bias_reg = regularizers.l2(self.settings['l2'])

        def conv1dinput(kernel):

            return Conv1D(kernel_size=kernel, filters=filters, activation=None,
                          input_shape=(inputShape, 1),
                          padding="same",
                          kernel_initializer=kernel_init,
                          bias_initializer=bias_init,
                          bias_regularizer=bias_reg,
                          kernel_regularizer=kernel_reg,
                          )

        def conv1d(kernel):
            return Conv1D(kernel_size=kernel, filters=filters, activation=None,
                          padding="same",
                          kernel_initializer=kernel_init,
                          bias_initializer=bias_init,
                          bias_regularizer=bias_reg,
                          kernel_regularizer=kernel_reg,
                          )

        if(useBatchnorm==True):
            input=BatchNormalization()(input)

        t0 = input
        t1 = input

        if (inputShape != 0):
            t1 = conv1dinput(kernel_size)(t1)
        else:
            t1 = conv1d(kernel_size)(t1)

        t1=activation(t1)
        t1=conv1d(kernel_size)(t1)

        output = add([t0, t1])

        output = activation(output)
        output=Dropout(self.settings['drop_rate'])(output)
        return output

    def resUnit3(self, input, filters, kernel_size, activation, kernel_init, bias_init, inputShape=0, advFilters=False, batchNormalization=True):
        kernel_reg = regularizers.l2(self.settings['l2'])
        bias_reg = regularizers.l2(self.settings['l2'])

        def conv1dinput(kernel):

            return Conv1D(kernel_size=kernel, filters=filters, activation=None,
                          input_shape=(inputShape, 1),
                          padding="same",
                          kernel_initializer=kernel_init,
                          bias_initializer=bias_init,
                          bias_regularizer=bias_reg,
                          kernel_regularizer=kernel_reg,
                          )

        def conv1d(kernel):
            return Conv1D(kernel_size=kernel, filters=filters, activation=None,
                          padding="same",
                          kernel_initializer=kernel_init,
                          bias_initializer=bias_init,
                          bias_regularizer=bias_reg,
                          kernel_regularizer=kernel_reg,
                          )


        t0 = input
        t1 = input

        if (inputShape != 0):
            t1 = conv1dinput(kernel_size)(t1)

            if(advFilters==True):
                t0=conv1dinput(kernel_size)(t0)

        else:
            t1 = conv1d(kernel_size)(t1)

            if(advFilters==True):
                t0 = conv1d(kernel_size)(t0)


        if(batchNormalization==True):
            t1=BatchNormalization()(t1)
            if(advFilters==True):
                t0=BatchNormalization()(t0)
        t1=activation(t1)

        t1=conv1d(kernel_size)(t1)

        if(batchNormalization==True):
            t1=BatchNormalization()(t1)

        output = add([t0, t1])

        output = activation(output)
        if(self.settings['drop_rate']>0):
            output=Dropout(self.settings['drop_rate'])(output)
        return output

    def convUnit(self, input, filters, kernel_size, activation, kernel_init, bias_init, inputShape=0):
        kernel_reg = regularizers.l2(self.settings['l2'])
        bias_reg = regularizers.l2(self.settings['l2'])

        def conv1dinput(kernel):

            return Conv1D(kernel_size=kernel, filters=filters, activation=None,
                          input_shape=(inputShape, 1),
                          padding="same",
                          kernel_initializer=kernel_init,
                          bias_initializer=bias_init,
                          bias_regularizer=bias_reg,
                          kernel_regularizer=kernel_reg,
                          )

        def conv1d(kernel):
            return Conv1D(kernel_size=kernel, filters=filters, activation=None,
                          padding="same",
                          kernel_initializer=kernel_init,
                          bias_initializer=bias_init,
                          bias_regularizer=bias_reg,
                          kernel_regularizer=kernel_reg,
                          )


        t1 = input

        if (inputShape != 0):
            t1 = conv1dinput(kernel_size)(t1)
        else:
            t1 = conv1d(kernel_size)(t1)

        t1=activation(t1)

        t1=Dropout(self.settings['drop_rate'])(t1)
        return t1

    def convUnit3(self, input, filters, kernel_size, activation, kernel_init, bias_init, inputShape=0):
        kernel_reg = regularizers.l2(self.settings['l2'])
        bias_reg = regularizers.l2(self.settings['l2'])

        def conv1dinput(kernel):

            return Conv1D(kernel_size=kernel, filters=filters, activation=None,
                          input_shape=(inputShape, 1),
                          padding="same",
                          kernel_initializer=kernel_init,
                          bias_initializer=bias_init,
                          bias_regularizer=bias_reg,
                          kernel_regularizer=kernel_reg,
                          )

        def conv1d(kernel):
            return Conv1D(kernel_size=kernel, filters=filters, activation=None,
                          padding="same",
                          kernel_initializer=kernel_init,
                          bias_initializer=bias_init,
                          bias_regularizer=bias_reg,
                          kernel_regularizer=kernel_reg,
                          )


        t1 = input

        if (inputShape != 0):
            t1 = conv1dinput(kernel_size)(t1)
        else:
            t1 = conv1d(kernel_size)(t1)

        t1=BatchNormalization()(t1)
        t1=activation(t1)

        return t1



    def resUnit235(self, input, filters, activation, kernel_init, bias_init, inputShape=0):
        kernel_reg = regularizers.l2(self.settings['l2'])
        bias_reg = regularizers.l2(self.settings['l2'])
        useActivationInside = False

        def conv1dinput(kernel):

            return Conv1D(kernel_size=kernel, filters=filters, activation=None,
                          input_shape=(inputShape, 1),
                          padding="same",
                          # strides=2,
                          kernel_initializer=kernel_init,
                          bias_initializer=bias_init,
                          bias_regularizer=bias_reg,
                          kernel_regularizer=kernel_reg,
                          )

        def conv1d(kernel):
            return Conv1D(kernel_size=kernel, filters=filters, activation=None,
                          padding="same",
                          # strides=2,
                          kernel_initializer=kernel_init,
                          bias_initializer=bias_init,
                          bias_regularizer=bias_reg,
                          kernel_regularizer=kernel_reg,
                          )

        def pool():
            return MaxPool1D(pool_size=2)

        # input = BatchNormalization()(input)
        # input=pool()(input)
        t0 = input
        t2 = input
        t3 = input
        t5 = input

        if (inputShape != 0):
            t2 = conv1dinput(2)(t2)
            t3 = conv1dinput(3)(t3)
            t5 = conv1dinput(5)(t5)
        else:
            t2 = conv1d(2)(t2)
            t3 = conv1d(3)(t3)
            t5 = conv1d(5)(t5)

        t2 = conv1d(2)(t2)
        t3 = conv1d(3)(t3)
        t5 = conv1d(5)(t5)

        out = add([t0, t2, t3, t5])
        out = activation(out)
        out = Dropout(self.settings['drop_rate'])(out)
        return out

    def conv1DMPResLayer(self, input, kernel_size, filters, activation, kernel_init, bias_init, inputShape=0):
        kernel_reg = regularizers.l1_l2(l1=self.settings['l1'], l2=self.settings['l2'])
        bias_reg = regularizers.l1_l2(l1=self.settings['l1'], l2=self.settings['l2'])

        if (inputShape != 0):
            x = input
            output = Conv1D(kernel_size=kernel_size, filters=filters, activation=None,
                            input_shape=(inputShape, 1),
                            padding="same",
                            # strides=strides,
                            kernel_initializer=kernel_init,
                            bias_initializer=bias_init,
                            bias_regularizer=bias_reg,
                            kernel_regularizer=kernel_reg,
                            )(input)
            output = Dropout(self.settings['drop_rate'])(output)
            output = BatchNormalization()(output)
            output = Conv1D(kernel_size=kernel_size, filters=filters, activation=None,
                            padding="same",
                            # strides=strides,
                            kernel_initializer=kernel_init,
                            bias_initializer=bias_init,
                            bias_regularizer=bias_reg,
                            kernel_regularizer=kernel_reg,
                            )(output)
            output = Dropout(self.settings['drop_rate'])(output)
            output = BatchNormalization()(output)
            output = add([output, x])
            output = Activation(activation=activation)(output)
            output = MaxPool1D(pool_size=(2))(output)
            return output
        else:
            x = input
            output = Conv1D(kernel_size=kernel_size, filters=filters, activation=activation,
                            padding="same",
                            # strides=strides,
                            kernel_initializer=kernel_init,
                            bias_initializer=bias_init,
                            bias_regularizer=bias_reg,
                            kernel_regularizer=kernel_reg,
                            )(input)
            output = Dropout(self.settings['drop_rate'])(output)
            output = BatchNormalization()(output)
            output = Conv1D(kernel_size=kernel_size, filters=filters, activation=None,
                            padding="same",
                            # strides=strides,
                            kernel_initializer=kernel_init,
                            bias_initializer=bias_init,
                            bias_regularizer=bias_reg,
                            kernel_regularizer=kernel_reg,
                            )(output)
            output = Dropout(self.settings['drop_rate'])(output)
            output = BatchNormalization()(output)
            output = add([output, x])
            output = Activation(activation=activation)(output)
            output = MaxPool1D(pool_size=(2))(output)
            return output

    def denseUnit(self, input, units, activation, kernel_init, bias_init):
        kernel_reg = regularizers.l2(self.settings['l2'])
        bias_reg = regularizers.l2(self.settings['l2'])

        output = input

        output = Dense(units=units,
                       kernel_initializer=kernel_init,
                       bias_initializer=bias_init,
                       bias_regularizer=bias_reg,
                       kernel_regularizer=kernel_reg,
                       )(output)
        output = activation(output)

        return output

    def LSTMUnit(self, input, units, activation, kernel_init, bias_init):
        kernel_reg = regularizers.l2(self.settings['l2'])
        bias_reg = regularizers.l2(self.settings['l2'])

        output = input
        # output = BatchNormalization()(output)
        output = LSTM(units=units,
                       kernel_initializer=kernel_init,
                       bias_initializer=bias_init,
                       bias_regularizer=bias_reg,
                       kernel_regularizer=kernel_reg,
                       )(output)
        output = activation(output)
        output = Dropout(self.settings['drop_rate'])(output)
        return output


class app:

    def initModel(self):
        e = elements(self.settings)
        # model

        kernel_init = 'he_normal'
        kernel_init = 'glorot_uniform'
        bias_init = 'zeros'
        kernel_reg = regularizers.l1_l2(l1=self.settings['l1'], l2=self.settings['l2'])
        bias_reg = regularizers.l1_l2(l1=self.settings['l1'], l2=self.settings['l2'])
        activity_reg = regularizers.l1_l2(l1=self.settings['l1'], l2=self.settings['l2'])

        filters = [64,64,128,256,512,1024]
        kernel_size=10

        #activation = ReLU()
        #activation=None
        batchNormalization=True
        activation='tanh'
        activation = ELU()

        def conv(filters):
            return Conv1D(kernel_size=kernel_size, filters=filters, activation=activation,
                              padding="same",
                              kernel_initializer=kernel_init,
                              bias_initializer=bias_init,
                              bias_regularizer=bias_reg,
                              kernel_regularizer=kernel_reg)

        def convi(filters,inputShape):
            return Conv1D(kernel_size=kernel_size, filters=filters, activation=activation,
                              input_shape=inputShape,
                              padding="same",
                              kernel_initializer=kernel_init,
                              bias_initializer=bias_init,
                              bias_regularizer=bias_reg,
                              kernel_regularizer=kernel_reg)


        # Encoder
        input = Input(shape=self.inputShape, name='input')

        x=input

        conv1 = convi(10, (self.X[0]['shape'], 1))(x)
        pool = MaxPool1D(pool_size=2, padding="same")(conv1)

        conv1 = conv(10)(pool)
        pool = MaxPool1D(pool_size=2, padding="same")(conv1)

        encoded=pool


        # Decoder
        e_input = Input(shape=(25,10), name='e_input')
        x = e_input

        conv1 = convi(10, [encoded.shape[1], encoded.shape[2]])(x)
        up = UpSampling1D(size=2)(conv1)

        conv1 = conv(10)(up)
        up = UpSampling1D(size=2)(conv1)

        decoded=conv(1)(up)



        encoder = Model(input, encoded, name="encoder")
        pass
        decoder = Model(e_input, decoded, name="decoder")
        autoencoder = Model(input, decoder(encoder(input)), name="autoencoder")

        #autoencoder=Model(input,decoded,name="autoencoder")

        model=autoencoder

        optimizer = None
        optimizer = optimizers.Adam(lr=self.settings['ls'], beta_1=0.9, beta_2=0.999, decay=0.0, amsgrad=True)
        #optimizer = optimizers.RMSprop(lr=self.settings['ls'], rho=0.9)
        #optimizer=optimizers.SGD(learning_rate=self.settings['ls'])#,momentum=0.1)

        autoencoder.compile(
            loss='mean_squared_error',
            #loss='categorical_crossentropy',
            optimizer=optimizer,
            # metrics=['accuracy','binary_accuracy'])
            metrics=['accuracy'])

        print(encoder.summary())
        print(decoder.summary())

        self.setModelName(autoencoder)
        #plot_model(model,to_file=self.job_dir + self.model_name + ".png")
        return autoencoder, encoder, decoder




    def initModel2(self):

        kernel_init = 'he_normal'
        kernel_init = 'glorot_uniform'
        bias_init = 'zeros'
        kernel_reg = regularizers.l1_l2(l1=self.settings['l1'], l2=self.settings['l2'])
        bias_reg = regularizers.l1_l2(l1=self.settings['l1'], l2=self.settings['l2'])

        kernel_size=10

        #activation = ReLU()
        #activation=None
        activation='tanh'
        activation = ELU()

        def conv(filters):
            return Conv1D(kernel_size=kernel_size, filters=filters, activation=activation,
                              padding="same",
                              kernel_initializer=kernel_init,
                              bias_initializer=bias_init,
                              bias_regularizer=bias_reg,
                              kernel_regularizer=kernel_reg)

        def convi(filters,inputShape):
            return Conv1D(kernel_size=kernel_size, filters=filters, activation=activation,
                              input_shape=inputShape,
                              padding="same",
                              kernel_initializer=kernel_init,
                              bias_initializer=bias_init,
                              bias_regularizer=bias_reg,
                              kernel_regularizer=kernel_reg)


        input = Input(shape=self.inputShape, name='input')

        x=convi(10,(self.X[0]['shape'],1))(input)
        x=MaxPool1D(pool_size=2, padding="same")(x)
        x=conv(10)(x)
        x=MaxPool1D(pool_size=2, padding="same")(x)
        encoded=x

        e_input = Input(shape=(25,10), name='e_input')
        x=convi(10,(encoded.shape[1],encoded.shape[2]) )(encoded)
        x=UpSampling1D(size=2)(x)
        x=conv(10)(x)
        x=UpSampling1D(size=2)(x)

        x=conv(1)(x)
        decoded=x






        encoder = Model(input, encoded, name="encoder")
        pass
        decoder = Model(e_input, decoded, name="decoder")
        autoencoder = Model(input, decoder(encoder(input)), name="autoencoder")

        #autoencoder=Model(input,decoded,name="autoencoder")

        model=autoencoder

        optimizer = None
        optimizer = optimizers.Adam(lr=self.settings['ls'], beta_1=0.9, beta_2=0.999, decay=0.0, amsgrad=True)
        #optimizer = optimizers.RMSprop(lr=self.settings['ls'], rho=0.9)
        #optimizer=optimizers.SGD(learning_rate=self.settings['ls'])#,momentum=0.1)

        autoencoder.compile(
            loss='mean_squared_error',
            #loss='categorical_crossentropy',
            optimizer=optimizer,
            # metrics=['accuracy','binary_accuracy'])
            metrics=['accuracy'])

        print(encoder.summary())
        print(decoder.summary())

        self.setModelName(autoencoder)
        #plot_model(model,to_file=self.job_dir + self.model_name + ".png")
        return autoencoder, encoder, decoder




    def setModelName(self, model):
        sName = ""
        for i in range(self.inputFiles):
            sName += str(self.X[i]['shape'])
            sName += '.'
        for i in model.layers:
            for j in range(0, layersNames.size):
                sName += i.name
                sName += "."
        for i in range(self.outputFiles):
            sName += str(self.Y[i]['shape'])
            sName += '.'
        name = hashlib.md5()
        name.update(sName.encode())

        self.model_name = name.hexdigest()
        self.model_name += ".h5"
        print(self.model_name)


    def lrSchedule(self, epoch):
        lr = self.settings['ls']
        return lr
        if epoch > 50:
            lr *= 0.1
            return lr
        if epoch > 100:
            lr *= 0.01
            return lr
        if epoch > 150:
            lr *= 0.001
            return lr
        if epoch > 200:
            lr *= 0.0001
            return lr
        return lr

    def threadTrain(self):
        self.training_is_launched = True

        backend.reset_uids()
        backend.clear_session()

        model, encoder, decoder = self.initModel()
        if (self.ctr == True):
            try:
                if(self.settings['saveWholeModel']==True):
                    model=load_model(self.job_dir + self.model_name)
                else:
                    model.load_weights(self.job_dir + self.model_name)
            except:
                print('no model')

            else:
                print('model loaded')

        train_x = None
        train_y = None
        test_x = None
        test_y = None
        all_x = None
        all_y = None
        for i in range(0, self.inputFiles):
            if i == 0:
                train_x = list([self.X_train[i]['data']])
                all_x = list([self.X[i]['data']])
                if (self.eval_size > 0.0):
                    test_x = list([self.X_test[i]['data']])
            else:
                train_x.append(self.X_train[i]['data'])
                all_x.append(self.X[i]['data'])
                if (self.eval_size > 0.0):
                    test_x.append(self.X_test[i]['data'])

        for i in range(0, self.outputFiles):
            if i == 0:
                train_y = list([self.Y_train[i]['data']])
                all_y = list([self.Y[i]['data']])
                if (self.eval_size > 0.0):
                    test_y = list([self.Y_test[i]['data']])
            else:
                train_y.append(self.Y_train[i]['data'])
                all_y.append(self.Y[i]['data'])
                if (self.eval_size > 0.0):
                    test_y.append(self.Y_test[i]['data'])

        #train_y = list([self.Y_train[0]['data']])
        #all_y = list([self.Y[0]['data']])
        #if (self.eval_size > 0.0):
        #    test_y = list([self.Y_test[0]['data']])

        self.n_batches_train = int(self.nTrainSize * self.settings['batch_size'])
        bcount = None
        try:
            bcount = 1 / self.settings['batch_size']
        except:
            pass
        else:
            print("1 epoch = {0} batches ({1})".format(bcount,self.n_batches_train))

        score_train = model.evaluate(test_x, test_y, batch_size=self.n_batches_train)  # , batch_size=500)
        score_test = None
        if (self.eval_size > 0.0):
            score_test = model.evaluate(train_x, train_y, batch_size=self.n_batches_train)  # , batch_size=500)
        print("loss {0} \nacc {1}".format(score_train[0], score_train[1]))
        if (self.eval_size > 0.0):
            print("val_loss {0} \nval_acc {1}".format(score_test[0], score_test[1]))

        # self.historyCallback.loss=np.array[score_train[0]]
        # self.historyCallback.acc=np.array[score_train[1]]
        # self.historyCallback.val_loss=np.array[score_test[0]]
        # self.historyCallback.val_acc=np.array[score_test[1]]
        # if (self.settings['metrics'] == 0):
        #    metr = 'full_acc'
        # if (self.settings['metrics'] == 1):
        #    metr = 'full_loss'

        metr = 'full_loss'
        if (metr == 'train_acc' or metr == 'train_loss'):
            self.historyCallback.initArrays(score_train[0], score_train[1])
        else:
            self.historyCallback.initArrays2(score_train[0], score_test[0], score_train[1], score_test[1])

        self.historyCallback.initData(self.X, self.Y, self.nDataSize, self.inputFiles, self.outputFiles)


        if (self.sLogName == None):
            self.logDir = self.job_dir + "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        else:
            self.logDir = self.job_dir + "logs/fit/" + self.sLogName

        if not os.path.isdir(self.logDir):
            os.makedirs(self.logDir)
        # self.runTensorboard()

        self.tb_log = callbacks.TensorBoard(log_dir=self.logDir, histogram_freq=0, write_graph=True,
                                            write_grads=False, write_images=False)
        self.historyCallback.initSettings(self.job_dir + self.model_name, metr,
                                          self.settings['overfit_epochs'], self.settings['reduction_epochs'],
                                          self.settings['ls_reduction_koef'],
                                          self.logDir, self.settings['saveWholeModel'], 1)
        lrScheduler = callbacks.LearningRateScheduler(self.lrSchedule)

        self.callbacks = [
            self.historyCallback,
            self.tb_log

        ]

        # model.fit_generator()
        self.log('start training')

        if (self.eval_size > 0.0):
            model.fit(x=train_x, y=train_y, epochs=self.settings['epochs'], verbose=1,
                      batch_size=self.n_batches_train,
                      # shuffle=True,
                      callbacks=self.callbacks,
                      validation_data=(test_x, test_y))
        else:
            model.fit(x=train_x, y=train_y, epochs=self.settings['epochs'], verbose=1,
                      batch_size=self.n_batches_train,
                      # shuffle=True,
                      callbacks=self.callbacks)

        # score = model.evaluate(self.X, self.Y)  # , batch_size=500)

        #w1=model.layers[1].get_weights()
        #w2=encoder.layers[1].get_weights()
        #
        #input=np.reshape(test_x[0][0], self.getList(1,self.inputShape))
        #output_model=model.predict(input)
        #output_enc=encoder.predict(input)
        #output_dec=decoder.predict(output_enc)
        #
        #np.savetxt("D:/autoencoder/decoded.txt",np.squeeze(input),delimiter="\n")
        #np.savetxt("D:/autoencoder/encoded.txt",np.squeeze(output_enc),delimiter="\n")
        #print(output_model,output_enc,output_dec)

        backend.reset_uids()
        backend.clear_session()

        self.training_is_launched = False

    def threadPredict(self):
        backend.reset_uids()
        backend.clear_session()

        model = self.initModel()
        try:
            if (self.settings['saveWholeModel'] == True):
                model = load_model(self.job_dir + self.model_name)
            else:
                model.load_weights(self.job_dir + self.model_name)
        except:
            print('no model')
            return
        else:
            print('model loaded')

        while True:
            allFilesFound = False
            #if (self.inputFiles == 1):
            #    if (os.path.isfile(self.job_dir + self.sDataInput1Path)):
            #        allFilesFound = True
            #else:
            for i in range(0, self.inputFiles):
                if (os.path.isfile(self.job_dir + self.sDataInput1PathM.format(0))):
                    allFilesFound = True
                else:
                    allFilesFound = False
                    break

            if (allFilesFound):

                goNext = False
                X0 = None
                #if (self.inputFiles == 1):
                #    try:
                #        new_data = np.genfromtxt(self.loadData(self.job_dir + self.sDataInput1Path))
                #    except:
                #        goNext = True
                #        continue
                #    X0 = list([new_data])
                #else:
                try:
                    new_data = np.genfromtxt(self.job_dir + self.sDataInput1PathM.format(0))
                except:
                    goNext = True
                    continue
                X0 = list([new_data])
                for i in range(1, self.inputFiles):
                    try:
                        new_data = np.genfromtxt(self.job_dir + self.sDataInput1PathM.format(i))
                    except:
                        goNext = True
                        continue
                    X0.append(new_data)

                if (goNext == False):
                    for i in range(0, self.inputFiles):
                        X0[i] = np.reshape(X0[i], self.getList(1,self.inputShape))

                    input = None
                    for i in range(0, self.inputFiles):
                        if i == 0:
                            input = list([X0[i]])
                        else:
                            input.append(X0[i])

                    #if (self.inputFiles == 1):
                    #    while(os.path.isfile(self.job_dir + self.sDataInput1Path)):
                    #        try:
                    #            os.remove(self.job_dir + self.sDataInput1Path)
                    #        except:
                    #            pass
                    #else:
                    for i in range(0, self.inputFiles):
                        while(os.path.isfile(self.job_dir + self.sDataInput1PathM.format(i))):
                            try:
                                os.remove(self.job_dir + self.sDataInput1PathM.format(i))
                            except:
                                pass

                    p = model.predict(x=input)

                    file = open(self.job_dir + 'answer.txt', 'w')
                    output = ""
                    for i in range(2):
                        output += str(p[0][i])
                        output += " "
                    file.write(output)
                    file.close()
                    print(output)

    def threadTest(self, count):

        yMin = -1.3
        yMax = 1.3

        backend.reset_uids()
        backend.clear_session()

        model = self.initModel()
        try:
            if (self.settings['saveWholeModel'] == True):
                model = load_model(self.job_dir + self.model_name)
            else:
                model.load_weights(self.job_dir + self.model_name)
        except:
            print('no model')
            return
        else:
            print('model loaded')

        try:
            import matplotlib.pyplot as plt
        except:
            return

        testingfig = np.empty(dtype=plt.Figure, shape=count)
        testingplot = np.empty(dtype=plt.Subplot, shape=count)
        ctime = datetime.now().strftime("%Y%m%d-%H%M%S")
        sample = random.sample(range(0, self.nDataSize), count)
        figCounter = 0
        if not os.path.isdir(self.job_dir + 'logs/test/' + ctime):
            os.makedirs(self.job_dir + 'logs/test/' + ctime)

        input = None
        output = None

        for i in range(0, self.inputFiles):
            if i == 0:
                input = list([self.X[i]['data']])
            else:
                input.append(self.X[i]['data'])

        for i in range(0, self.outputFiles):
            if i == 0:
                output = list([self.Y[i]['data']])
            else:
                output.append(self.Y[i]['data'])

        #output = self.Y[0]['data']

        prediction = model.predict(x=input)

        np.savetxt(self.job_dir + "prediction.txt", prediction, delimiter=" ")
        np.savetxt(self.job_dir + "output.txt", output, delimiter=" ")

        testingfig = plt.figure(num='Testing plot', figsize=(16, 9), dpi=100)
        testingplot = testingfig.add_subplot(111)
        if (yMin != 0 and yMax != 0):
            plt.ylim(yMin, yMax)
        testingplot.plot(output, linewidth=0.05, color='b')
        testingplot.plot(prediction, linewidth=0.05, color='r')
        testingfig.savefig(fname=self.job_dir + 'logs/test/' + ctime + '/all')

        for index in range(0, self.nDataSize):

            _y = self.Y[0]['data'][index]
            _p = prediction[index]
            _p = np.reshape(_p, self.Y[0]['shape'])

            testingfig = plt.figure(num='Testing plot ' + str(index), figsize=(10, 7), dpi=80)
            testingplot = testingfig.add_subplot(111)
            if (yMin != 0 and yMax != 0):
                plt.ylim(yMin, yMax)
            testingplot.plot(_y, linewidth=0.5, color='b')
            testingplot.plot(_p, linewidth=0.5, color='r')
            testingfig.savefig(fname=self.job_dir + 'logs/test/' + ctime + '/' + str(index))
            plt.close(testingfig)

        # while True:
        #    pass
        ##plt.ioff()
        # backend.reset_uids()
        # backend.clear_session()

    def setSettings(self, settingName, settingValue):
        for i in self.settings:
            if i == settingName:
                self.settings[i] = self.settingsDtypes[i](settingValue)

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
            'ls_reduction_koef': float,
            'metrics': int,
            'batch_size': float,
            'saveWholeModel': bool
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
            'ls_reduction_koef': 0.95,
            'metrics': 0,
            'batch_size': 0.1,
            'saveWholeModel': False
        }

        self.sDataInput1Path = "input.txt"
        self.sDataInput1PathM = "input{0}.txt"
        self.sDataInputPath = "in_data.txt"
        self.sTrainDataInputPath = "in_data_train.txt"
        self.sTestDataInputPath = "in_data_test.txt"
        self.sDataInputPathM = "in_data{0}.txt"


        self.sDataOutputPath = "out_data.txt"
        self.sTrainDataOutputPath = "out_data_train.txt"
        self.sTestDataOutputPath = "out_data_test.txt"



        self.sTrainDataInputPathM = "sinoffset_train{0}.txt"
        self.sTestDataInputPathM  = "sinoffset_test{0}.txt"


        self.sTrainDataOutputPathM = "sinoffset_train{0}.txt"
        self.sTestDataOutputPathM  = "sinoffset_test{0}.txt"


        self.sLogName = None

        self.training_is_launched = False
        self.run_is_launched = False
        self.saving_is_launched = False
        self.savepng_is_launched = False
        self.test_model = False
        self.model_is_tested = True
        self.testing_model = False
        self.ctr = False

        self.inputFiles = 1
        self.outputFiles = 1

        self.inputShape=(100,1)
        self.outputShape=(100,1)

    def setLogName(self, logName):
        self.sLogName = logName

    def log(self, str):
        if DISABLE_LOG == True:
            return
        logfilename = self.job_dir + "log.txt"
        time = datetime.strftime(datetime.now(), "%Y.%m.%d %H:%M:%S")
        file = None
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

        if (file != None):
            file.write(time + ' ')
            file.write(str + '\n')
            file.close()

    def loadFromFile(self, filename, oneFrame=False):
        file = None
        try:
            file = open(filename, 'r')
        except:
            file = io.gfile.GFile(filename, 'r')

        strData = file.read()
        strData = strData.split()
        doubleData = np.array(strData, dtype=np.float32)


        return doubleData

    def loadData(self, shape, path, oneFrame=False):
        data = self.loadFromFile(path, oneFrame)
        #inputs = data.shape[1]
        out = {'data': data,
               'shape': shape}
        return out




    def prepareTrainData(self):

        self.nTrainSize = int(self.nDataSize)

        self.X_train = np.array([self.loadData(self.getList(self.nTrainSize,self.inputShape), self.job_dir + self.sTrainDataInputPathM.format(0))])
        for i in range(1, self.inputFiles):
            self.X_train = np.append(self.X_train,
                                     self.loadData(self.getList(self.nTrainSize,self.inputShape), self.job_dir + self.sTrainDataInputPathM.format(i)))

        self.Y_train = np.array([self.loadData(self.getList(self.nTrainSize,self.outputShape), self.job_dir + self.sTrainDataOutputPathM.format(0))])
        for i in range(1, self.outputFiles):
            self.Y_train = np.append(self.Y_train,
                                     self.loadData(self.getList(self.nTrainSize,self.outputShape), self.job_dir + self.sTrainDataOutputPathM.format(i)))




        for i in range(0, self.inputFiles):
            self.X_train[i]['data'] = np.reshape(self.X_train[i]['data'],self.getList(self.nTrainSize,self.inputShape))


        for i in range(0, self.outputFiles):
            self.Y_train[i]['data'] = np.reshape(self.Y_train[i]['data'], self.getList(self.nTrainSize,self.outputShape))


        self.X = np.empty(shape=self.inputFiles, dtype=dict)
        self.Y = np.empty(shape=self.outputFiles, dtype=dict)

        for i in range(0, self.inputFiles):
            self.X[i] = {'data': None,
                         'shape': self.X_train[i]['shape']}
        for i in range(0, self.outputFiles):
            self.Y[i] = {'data': None,
                         'shape': self.Y_train[i]['shape']}




    def prepareTestData(self):

        self.nTestSize = int(self.nDataSize)

        self.X_test = np.array([self.loadData(self.getList(self.nTestSize,self.inputShape), self.job_dir + self.sTestDataInputPathM.format(0))])
        for i in range(1, self.inputFiles):
            self.X_test = np.append(self.X_test, self.loadData(self.getList(self.nTestSize,self.inputShape), self.job_dir + self.sTestDataInputPathM.format(i)))

        self.Y_test = np.array([self.loadData(self.getList(self.nTestSize,self.outputShape), self.job_dir + self.sTestDataOutputPathM.format(0))])
        for i in range(1, self.outputFiles):
            self.Y_test = np.append(self.Y_test, self.loadData(self.getList(self.nTestSize,self.outputShape), self.job_dir + self.sTestDataOutputPathM.format(i)))



        for i in range(0, self.inputFiles):
            self.X_test[i]['data'] = np.reshape(self.X_test[i]['data'],self.getList(self.nTestSize,self.inputShape))


        for i in range(0, self.outputFiles):
            self.Y_test[i]['data'] = np.reshape(self.Y_test[i]['data'], self.getList(self.nTestSize,self.outputShape))



    def prepareData(self):
        self.nDataSize = int(self.nTestSize + self.nTrainSize)

        self.X = np.empty(shape=self.inputFiles, dtype=dict)
        self.Y = np.empty(shape=self.outputFiles, dtype=dict)

        for i in range(0, self.inputFiles):
            self.X[i] = {'data': None,
                         'shape': None}
        for i in range(0, self.outputFiles):
            self.Y[i] = {'data': None,
                         'shape': None}


        for i in range(0, self.inputFiles):
            self.X[i]['data'] = np.append(self.X_train[i]['data'], self.X_test[i]['data'])
            self.X[i]['shape'] = self.X_train[i]['shape']
            self.X[i]['data'] = np.reshape(self.X[i]['data'], self.getList(self.nDataSize,self.inputShape))

        for i in range(0, self.outputFiles):
            self.Y[i]['data'] = np.append(self.Y_train[i]['data'], self.Y_test[i]['data'])
            self.Y[i]['shape'] = self.Y_train[i]['shape']
            self.Y[i]['data'] = np.reshape(self.Y[i]['data'], self.getList(self.nDataSize,self.outputShape))

    def runTensorboard(self):
        ts = threading.Thread(target=self.threadTensorboard)
        ts.daemon = True
        ts.start()

    def threadTensorboard(self):
        cmd = "tensorboard --logdir {0}".format(self.logDir)
        os.system(cmd)

    def __init__(self, job_dir, data_size, eval_size):
        self.historyCallback = historyCallback()
        self.job_dir = job_dir
        self.eval_size = float(eval_size)
        self.nDataSize = int(data_size)
        # self.initPlots()
        self.initSettings()

        # loadData
        # self.prepareData()
        self.prepareTrainData()
        self.prepareTestData()
        self.prepareData()
        self.log('data loaded')

    def getTuple(self,a,b):
        out=list(a)
        for i in b:
            out.append(i)
        out=tuple(out)
        return out

    def getList(self,a,b):
        out=list([a])
        for i in b:
            out.append(i)

        return out

def main(job_dir, mode, ctr, data_size, eval_size, batch_size, epochs, overfit_epochs, reduction_epochs,
         ls_reduction_koef, ls, l1, l2, drop_rate):  # , **args):
    z = app(job_dir, data_size, eval_size)
    # print(mode)
    # if(mode.find('train')>0):
    mode = int(mode)
    ctr = int(ctr)

    if (epochs != None):
        z.setSettings('epochs', epochs)
    if (overfit_epochs != None):
        z.setSettings('overfit_epochs', overfit_epochs)
    if (reduction_epochs != None):
        z.setSettings('reduction_epochs', reduction_epochs)
    if (ls_reduction_koef != None):
        z.setSettings('ls_reduction_koef', ls_reduction_koef)
    if (ls != None):
        z.setSettings('ls', ls)
    if (l1 != None):
        z.setSettings('l1', l1)
    if (l2 != None):
        z.setSettings('l2', l2)
    if (drop_rate != None):
        z.setSettings('drop_rate', drop_rate)
    if (batch_size != None):
        z.setSettings('batch_size', batch_size)
    print(z.settings)
    if (ctr == 1):
        z.ctr = True


    if (mode == 0):
        z.threadTrain()

    # if(mode.find('test')>0):
    if (mode == 1):
        # r=np.random()
        z.threadTest(z.nDataSize)

    # if(mode.find('predict')>0):
    if (mode == 2):
        z.threadPredict()

    # if(mode.find('optimise')>0):
    if (mode == 3):
        z.ctr = False

        ls_arr = np.array([0.000000001, 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001])
        for i in ls_arr:
                    # for l in l2_arr:
                    name = "ls={0}".format(i)
                    z.setLogName(name)
                    z.setSettings('epochs', epochs)
                    z.setSettings('overfit_epochs', overfit_epochs)
                    z.setSettings('reduction_epochs', reduction_epochs)
                    z.setSettings('ls_reduction_koef', ls_reduction_koef)
                    z.setSettings('ls', i)
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
        '--ctr',
        help='Load the model and continue training',
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
        required=True)
    parser.add_argument(
        '--overfit-epochs',
        help='Overfit epochs',
        required=True)
    parser.add_argument(
        '--reduction-epochs',
        help='Reduction epochs',
        required=True)
    parser.add_argument(
        '--ls-reduction-koef',
        help='Reduction factor',
        required=True)
    parser.add_argument(
        '--ls',
        help='Learning speed',
        required=True)
    parser.add_argument(
        '--l1',
        help='L1',
        required=True)
    parser.add_argument(
        '--l2',
        help='L2',
        required=True)
    parser.add_argument(
        '--drop-rate',
        help='Drop rate',
        required=True)

    args = parser.parse_args()
    arguments = args.__dict__

    main(**arguments)





#--job-dir=D:/data_sinoffset/
#--mode=2
#--ctr=0
#--data-size=100
#--eval-size=0.2
#--batch-size=0.0256
#--epochs=100000
#--overfit-epochs=5000
#--reduction-epochs=2000
#--ls-reduction-koef=0.85
#--ls=0.001
#--l1=0.00
#--l2=0.0001
#--drop-rate=0.15