import numpy as np
import matplotlib.pyplot as plt

b = np.genfromtxt("1072.txt")

#count1=0
#count2=0
#count3=0
#for i in b:
#    if(i[0]==1):
#        count1+=1
#    if(i[1]==1):
#        count2+=1
#    if(i[2]==1):
#        count3+=1
#print("{0} {1} {2}".format(count1,count2,count3))

testingfig = plt.figure(num='fig', figsize=(16, 9), dpi=100)
testingplot = testingfig.add_subplot(111)
testingplot.plot(b[:,0], linewidth=0.5, color='b')
testingplot.plot(b[:,3], linewidth=0.5, color='r')
#testingplot.axhline(y=0.5,linewidth=0.5,color='b',linestyle='-')
#testingplot.axhline(y=sr0,linewidth=0.4,color='g',linestyle='-')
#testingplot.axhline(y=-sr0,linewidth=0.4,color='g',linestyle='-')
#testingplot.axhline(y=sr1,linewidth=0.4,color='k',linestyle='-.')
#testingplot.axhline(y=-sr1,linewidth=0.4,color='k',linestyle='-.')
plt.show()
