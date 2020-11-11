import os
import numpy as np
import matplotlib.pyplot as plt
import math
a=[]
for i in range(0,100):
    a.append(math.sin(2*3.1*i/100))

fig = plt.figure(num='fig', figsize=(16, 9), dpi=100)
myplot = fig.add_subplot(1,1,1)
myplot.plot(a, linewidth=0.5, color='b')
plt.show()