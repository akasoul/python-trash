import numpy as np
import os

for file in os.listdir():
    num=""
    if file.find("state")!= -1:
        num=file[5:len(file)-4]
        filename2="reward"+num+".txt"
        if(os.path.isfile(filename2)):
            pass

indata=np.load("indata.npy")
outdata=np.load("outdata.npy")


s1=indata.shape[0]
indata2=np.delete(indata,1,0)
outdata2=np.delete(outdata,1,0)

pass
