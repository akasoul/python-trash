import numpy as np
import random

sTrainDataInputPathM = "in_data_train{0}.txt"
sTestDataInputPathM = "in_data_test{0}.txt"
sTrainDataOutputPath = "out_data_train.txt"
sTestDataOutputPath = "out_data_test.txt"

inputFiles=4

xTrain = None
for i in range(0, inputFiles):
    if i==0:
        xTrain=list([np.genfromtxt(sTrainDataInputPathM.format(i))])
    else:
        xTrain.append(np.genfromtxt(sTrainDataInputPathM.format(i)) )

yTrain = np.genfromtxt(sTrainDataOutputPath)



xTest = None
for i in range(0, inputFiles):
    if i==0:
        xTest=list([np.genfromtxt(sTestDataInputPathM.format(i))] )
    else:
        xTest.append(np.genfromtxt(sTestDataInputPathM.format(i)) )

yTest = np.genfromtxt(sTestDataOutputPath)



delCounter=20000
counter=0
while(counter<delCounter):
    index=random.randrange(0,yTrain.shape[0],1)
    if(yTrain[index][2] == 1.0):
        for i in range(0,inputFiles):
            xTrain[i]=np.delete(xTrain[i],index,0)
        yTrain=np.delete(yTrain,index,0)
        counter+=1


counter=0
while(counter<delCounter):
    index=random.randrange(0,yTest.shape[0],1)
    if(yTest[index][2] == 1.0):
        for i in range(0,inputFiles):
            xTest[i]=np.delete(xTest[i],index,0)
        yTest=np.delete(yTest,index,0)
        counter+=1

for i in range(0, inputFiles):
    np.savetxt(sTrainDataInputPathM.format(i),xTrain[i], fmt="%.5f")
    np.savetxt(sTestDataInputPathM.format(i),xTest[i], fmt="%.5f")

np.savetxt(sTrainDataOutputPath,yTrain, fmt="%.1f")
np.savetxt(sTestDataOutputPath,yTest, fmt="%.1f")
