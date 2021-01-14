import os
import numpy as np

mod=10000

files=os.listdir()
for i in files:
    if(i.find('.txt')!=-1):
        try:
            a=np.genfromtxt(i)
        except:
            pass
        else:
            a=a*mod
            os.remove(i)
            np.savetxt(i,a)
