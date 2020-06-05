import numpy as np
import matplotlib.pyplot as plt




a = np.genfromtxt("D:/data0306/logs/fit/20200605-115353/tests/texts/516.txt")


fig = plt.figure(num='fig', figsize=(16, 9), dpi=100)
testingplot = fig.add_subplot(1,1,1)
#trainingplot = fig.add_subplot(2,1,2)

testingplot.plot(a[  10,:], linewidth=0.5, color='b')
testingplot.plot(a[2510,:], linewidth=0.5, color='r')

plt.show()
