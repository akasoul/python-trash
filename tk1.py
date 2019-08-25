import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
plt.style.use('seaborn-pastel')
from matplotlib.figure import Figure

from tkinter import *
import random
import time
import threading

class tkChart:


    def threadDraw(self,i):
        if(self.redraw==True):
            self.redraw=False
            self.plot.clear()
            self.plot.plot(self.data, color='b',
                                   label='blue')
            self.plot.plot(self.data2, color='r',
                                   label='red')


    def threadAddData(self):
        while True:
            self.data=np.append(self.data,random.random()%10)
            self.data2=np.append(self.data2,random.random()%10)
            self.redraw=True
            time.sleep(1)

    def onClick(self,event):
        self.trainingani=animation.FuncAnimation(self.fig, self.threadDraw, interval=1000)
        self.fig.show()


    def __init__(self):
        self.root = Tk()
        self.root.minsize(width=30,height=20)

        self.btnRun=           Button(self.root,height=1,width=10,text='start')
        self.btnRun.place(x=10, y=10)
        self.btnRun            .bind('<Button 1>', self.onClick)

        self.redraw=False

        self.data=np.array([1,2,3,2,1])
        self.data2=np.array([3,2,1,2,3])

        self.fig=plt.figure(figsize=(10, 7), dpi=80,num='title')
        self.fig2=plt.figure(figsize=(10, 7), dpi=80,num='title')
        self.plot=self.fig.add_subplot(111)



        tt = threading.Thread(target=self.threadAddData)
        tt.daemon = True
        tt.start()

        self.root.mainloop()



z=tkChart()