import numpy as np
import matplotlib.pyplot as plt

b = np.genfromtxt("out_data_train.txt")

count1=0
count2=0
count3=0
for i in b:
    if(i[0]==1):
        count1+=1
    if(i[1]==1):
        count2+=1
    if(i[2]==1):
        count3+=1
print("train stats: {0} {1} {2}".format(count1,count2,count3))


a = np.genfromtxt("out_data_test.txt")

count1=0
count2=0
count3=0
for i in a:
    if(i[0]==1):
        count1+=1
    if(i[1]==1):
        count2+=1
    if(i[2]==1):
        count3+=1
print("test stats: {0} {1} {2}".format(count1,count2,count3))

fig = plt.figure(num='fig', figsize=(16, 9), dpi=100)
testingplot = fig.add_subplot(2,1,1)
trainingplot = fig.add_subplot(2,1,2)

testingplot.plot(a[:,2], linewidth=0.5, color='b')
trainingplot.plot(b[:,2], linewidth=0.5, color='r')

plt.show()
