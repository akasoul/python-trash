import numpy as np
import matplotlib.pyplot as plt


path=r"/Users/antonvoloshuk/Documents/Dinamika/sinwave2.txt"
a=np.genfromtxt(path)
#a = np.genfromtxt(r"C:\Users\antonvoloshuk\AppData\Roaming\MetaQuotes\Terminal\287469DEA9630EA94D0715D755974F1B\tester\files\jobr\EURUSD\logs\fit\20200630-105551\tests\texts\1390.txt")
#b = np.genfromtxt(r"C:\Users\antonvoloshuk\AppData\Roaming\MetaQuotes\Terminal\287469DEA9630EA94D0715D755974F1B\tester\files\jobr\EURUSD\out_data_train0.txt")

#a=np.reshape(a,[10000,200])
fig = plt.figure(num='fig', figsize=(16, 9), dpi=100)
testingplot1 = fig.add_subplot(1,1,1)
#testingplot2 = fig.add_subplot(2,1,2)
#trainingplot = fig.add_subplot(2,1,2)


index=1000

testingplot1.plot(a, linewidth=0.5, color='b')
#testingplot1.plot(a[index][100:200], linewidth=0.5, color='r')
#testingplot1.plot(a[index][200:300], linewidth=0.5, color='g')
#testingplot1.plot(b[index][0:100], linewidth=0.5, color='y')

#testingplot2.plot(b[3300:3400,0], linewidth=0.5, color='b')
#testingplot2.plot(b[3300:3400,2], linewidth=0.5, color='r')

plt.show()


# index=1
# testingplot.plot(a[index,0:99], linewidth=0.5, color='b')
# testingplot.plot(a[index,2:200], linewidth=0.5, color='r')
#
#
# plt.show()
