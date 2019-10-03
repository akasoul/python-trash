import numpy as np



indata=np.load("indata.npy")
outdata=np.load("outdata.npy")


s1=indata.shape[0]
indata2=np.delete(indata,1,0)
outdata2=np.delete(outdata,1,0)

pass
