import numpy as np
import matplotlib.pyplot as plt




a = np.genfromtxt(r"D:\TrainingData\data0706\logs\fit\20200608-105809\tests\texts\7376.txt")


fig = plt.figure(num='fig', figsize=(16, 9), dpi=100)
testingplot = fig.add_subplot(1,1,1)
#trainingplot = fig.add_subplot(2,1,2)

testingplot.plot(a[2798,0:99], linewidth=0.5, color='b')
testingplot.plot(a[2798,100:200], linewidth=0.5, color='r')

plt.show()
