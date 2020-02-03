import numpy as np
import matplotlib.pyplot as plt

b = np.genfromtxt("3373.txt")



testingfig = plt.figure(num='Testing plot', figsize=(16, 9), dpi=100)
testingplot = testingfig.add_subplot(111)
testingplot.plot(b[:,0], linewidth=0.5, color='b')
testingplot.plot(b[:,1], linewidth=0.5, color='r')
#testingplot.axhline(y=0.5,linewidth=0.5,color='b',linestyle='-')
#testingplot.axhline(y=sr0,linewidth=0.4,color='g',linestyle='-')
#testingplot.axhline(y=-sr0,linewidth=0.4,color='g',linestyle='-')
#testingplot.axhline(y=sr1,linewidth=0.4,color='k',linestyle='-.')
#testingplot.axhline(y=-sr1,linewidth=0.4,color='k',linestyle='-.')
plt.show()
