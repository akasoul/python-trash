import numpy as np
import matplotlib.pyplot as plt




a = np.genfromtxt("C:/Users/antonvoloshuk/AppData/Roaming/MetaQuotes/Terminal/287469DEA9630EA94D0715D755974F1B/tester/files\jobr\EURUSD\logs/fit/20200607-202347/tests/texts/54.txt")


fig = plt.figure(num='fig', figsize=(16, 9), dpi=100)
testingplot = fig.add_subplot(1,1,1)
#trainingplot = fig.add_subplot(2,1,2)

testingplot.plot(a[1,0:99], linewidth=0.5, color='b')
testingplot.plot(a[1,100:200], linewidth=0.5, color='r')

plt.show()
