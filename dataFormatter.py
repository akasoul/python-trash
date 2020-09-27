import numpy as np
import os

inTrain = r"C:\Users\antonvoloshuk\AppData\Roaming\MetaQuotes\Terminal\287469DEA9630EA94D0715D755974F1B\tester\files\jobr\EURUSD\in_data_train{0}.txt"
inTest = r"C:\Users\antonvoloshuk\AppData\Roaming\MetaQuotes\Terminal\287469DEA9630EA94D0715D755974F1B\tester\files\jobr\EURUSD\in_data_test{0}.txt"
outTrain = r"C:\Users\antonvoloshuk\AppData\Roaming\MetaQuotes\Terminal\287469DEA9630EA94D0715D755974F1B\tester\files\jobr\EURUSD\out_data_train{0}.txt"
outTest = r"C:\Users\antonvoloshuk\AppData\Roaming\MetaQuotes\Terminal\287469DEA9630EA94D0715D755974F1B\tester\files\jobr\EURUSD\out_data_test{0}.txt"

inFilesCount=4
outFilesCount=1
split=1000

inFmt="%.5f"
outFmt="%.1f"


for i in range(0,inFilesCount):
    a=np.genfromtxt(inTrain.format(i))
    test=a[:split]
    train=a[split:]
    os.remove(inTrain.format(i))
    os.remove(inTest.format(i))
    np.savetxt(inTrain.format(i),train,fmt=inFmt)
    np.savetxt(inTest.format(i),test,fmt=inFmt)



for i in range(0,outFilesCount):
    a=np.genfromtxt(outTrain.format(i))
    test=a[:split]
    train=a[split:]
    os.remove(outTrain.format(i))
    os.remove(outTest.format(i))
    np.savetxt(outTrain.format(i),train,fmt=outFmt)
    np.savetxt(outTest.format(i),test,fmt=outFmt)
