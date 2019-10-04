import numpy as np
import os

# for file in os.listdir():
#     num = ""
#     if file.find("state") != -1:
#         num = file[5:len(file) - 4]
#         filename2 = "reward" + num + ".txt"
#         if (os.path.isfile(filename2)):
#             pass

def sortArr(data):
    s1=data.shape[0]
    s2=data.shape[1]
    temp_data=data
    out_data=temp_data[np.argmax(temp_data[:,0]),:]
    temp_data = np.delete(temp_data, temp_data[np.argmax(temp_data[:, 0]), :])
    s1-=1
    temp_data.reshape([s1,s2])
    while(outdata.shape[0]!=data.shape[0]):
        out_data = np.append(out_data,temp_data[np.argmax(temp_data[:, 0]), :])
        temp_data=np.delete(temp_data,temp_data[np.argmax(temp_data[:, 0]), :])
        s1-=1
        temp_data.reshape([s1,s2])

    return out_data

indata=np.load("indata.npy")
outdata=np.load("outdata.npy")
outdata=np.genfromtxt("123.txt")

s1=indata.shape[0]
indata2=np.delete(indata,1,0)
outdata2=np.sort(outdata,0)

data1=sortArr(outdata)
pass
