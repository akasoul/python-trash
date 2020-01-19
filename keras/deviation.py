import numpy as np
import matplotlib.pyplot as plt


a = np.genfromtxt("prediction.txt")
np.savetxt("1.txt", a, delimiter=" ",fmt='%1.3f')

#a=a[:,0]

b = np.genfromtxt("output.txt")
#b=b[:,0]
z=(a,b,a,b)
c=np.array([a,b])
c=np.column_stack((a,b,a,b))
np.savetxt("2.txt", c, delimiter=" ",fmt='%1.3f')

s=None
for i in a:
    if(s==None):
        s=abs(i)
    else:
        s+=abs(i)
s=s/a.size

sr1=0
sr0=0
count1=0
count0=0
for i in range(0,a.size):
    if(b[i]==1 or b[i]==-1):
        count1+=1
        sr1+=abs(a[i])
    if (b[i] == 0):
        count0+=1
        sr0+=abs(a[i])
sr0/=count0
sr1/=count1
testingfig = plt.figure(num='Testing plot', figsize=(16, 9), dpi=100)
testingplot = testingfig.add_subplot(111)
testingplot.plot(b, linewidth=0.2, color='b')
testingplot.plot(a, linewidth=0.2, color='r')
testingplot.axhline(y=0.5,linewidth=0.2,color='b',linestyle='-')
#testingplot.axhline(y=sr0,linewidth=0.4,color='g',linestyle='-')
#testingplot.axhline(y=-sr0,linewidth=0.4,color='g',linestyle='-')
#testingplot.axhline(y=sr1,linewidth=0.4,color='k',linestyle='-.')
#testingplot.axhline(y=-sr1,linewidth=0.4,color='k',linestyle='-.')
plt.show()
