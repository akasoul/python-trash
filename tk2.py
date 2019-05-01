from tkinter import *
import matplotlib as plt
import numpy as np
import matplotlib.animation as animation
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
import threading
import os
import random
import time

spath='data.txt'
spath2='data2.txt'

#a=2  # type: int

def load_params(fname,list,list_dtypes,list_values):
    if(os.path.isfile(fname)):
        f = open(fname, 'r')
        a = f.read()
        for i in range(list.__len__()):
            x=a.find(list[i])
            if(x!=-1):
                y=a.find(":",x)
                if(y!=-1):
                    z=a.find("\n",y)
                    if(z!=-1):
                        list_values.append(list_dtypes[i](a[y+1:z]))

def save_params(fname,list,list_dtypes,list_values):
    if(os.path.isfile(fname)):
        os.remove(fname)
    f = open(fname, 'w+')
    if(list.__len__()==list_dtypes.__len__() and list_dtypes.__len__()==list_values.__len__()):
        for i in range(list.__len__()):
            s_value=list[i]+":"+str(list_values[i])+"\n"
            f.write(s_value)


class app:

    def checkpath_thread(self):
        print('thread start')
        while(True):
            if(self.new_path==True):
                if(os.path.isfile(self.s_indatapath)):
                    self.ed_indatapath['bg']= 'green'
                else:
                    self.ed_indatapath['bg'] = 'red'

                if(os.path.isfile(self.s_outdatapath)):
                    self.ed_outdatapath['bg']= 'green'
                else:
                    self.ed_outdatapath['bg'] = 'red'

                if(os.path.isfile(self.s_inputpath)):
                    self.ed_inputpath['bg']= 'green'
                else:
                    self.ed_inputpath['bg'] = 'red'

                self.new_path=False


    def press_button(self,event):
        fpath=self.dpath.get(1.0,END)
        print(fpath)
        fpath=fpath.rstrip()
        fpath=fpath.lstrip()
        try:
            a = np.genfromtxt(fpath)
        except:
            print("error")
        else:
            print(a)
            self.lbl['text']=a
            #v = StringVar()
            #Label(self.lbl, textvariable=v).pack()
            #v.set(a)


    def on_changed(self,event):
        fpath=self.ed_indatapath.get(1.0, END)
        fpath=fpath.rstrip()
        fpath=fpath.lstrip()
        if(self.s_indatapath!=fpath):
            self.s_indatapath = fpath
            self.new_path=True

        fpath=self.ed_outdatapath.get(1.0, END)
        fpath=fpath.rstrip()
        fpath=fpath.lstrip()
        if(self.s_outdatapath!=fpath):
            self.s_outdatapath = fpath
            self.new_path=True

        fpath=self.ed_inputpath.get(1.0, END)
        fpath=fpath.rstrip()
        fpath=fpath.lstrip()
        if(self.s_inputpath!=fpath):
            self.s_inputpath = fpath
            self.new_path=True

    def add_data_thread(self):
        while True:
            self.tdata=np.append(self.tdata,random.randint(0,10))

    def draw_thread(self,i):
        self.trainingplot.clear()
        self.trainingplot.plot(self.tdata)


    def __init__(self):
        self.root = Tk()
        self.root.minsize(width=600,height=500)

        self.s_indatapath=''
        self.s_outdatapath=''
        self.s_inputpath=''

        self.new_path=True

        self.lbl_indatapath=    Label(self.root,height=1,width=12,font='Arial 11',bg="white", fg="black",text='in_data fname :',anchor=W, justify=LEFT)
        self.lbl_outdatapath=   Label(self.root,height=1,width=12,font='Arial 11',bg="white", fg="black",text='out_data fname:',anchor=W, justify=LEFT)
        self.lbl_inputpath=     Label(self.root,height=1,width=12,font='Arial 11',bg="white", fg="black",text='input fname   :',anchor=W, justify=LEFT)

        self.frm=Frame(self.root,bg='white',bd=5,height=200, width=300)

        #self.btn = Button(self.root,  # родительское окно
        #             text="Click me",  # надпись на кнопке
        #             width=10, height=5,  # ширина и высота
        #             bg="white", fg="black")  # цвет фона и надписи
        self.ed_indatapath  = Text(self.root, height=1, width=15, font='Arial 11', wrap=WORD)
        self.ed_outdatapath = Text(self.root, height=1, width=15, font='Arial 11', wrap=WORD)
        self.ed_inputpath   = Text(self.root, height=1, width=15, font='Arial 11', wrap=WORD)

        self.ed_indatapath.insert(1.0, 'in_data.txt')
        self.ed_outdatapath.insert(1.0, 'out_data.txt')
        self.ed_inputpath.insert(1.0, 'input.txt')

        self.ed_indatapath.bind('<Key>', self.on_changed)
        self.ed_outdatapath.bind('<Key>', self.on_changed)
        self.ed_inputpath.bind('<Key>', self.on_changed)
        self.ed_indatapath.bind('<Button 1>', self.on_changed)
        self.ed_outdatapath.bind('<Button 1>', self.on_changed)
        self.ed_inputpath.bind('<Button 1>', self.on_changed)

        self.ed_indatapath.place(x=120, y=10)
        self.ed_outdatapath.place(x=120, y=40)
        self.ed_inputpath.place(x=120, y=70)

        self.frm.place(x=250, y=10)
        #self.btn.place(x=10, y=5)
        self.lbl_indatapath.place(x=10, y=10)
        self.lbl_outdatapath.place(x=10, y=40)
        self.lbl_inputpath.place(x=10, y=70)

#settings input\output
        t = threading.Thread(target=self.checkpath_thread)
        t.daemon = True
        t.start()
        params=["xxc","vvh","asd","qew"]
        params_dtypes=[int,int,float,float]
        params_values=list()
        load_params("data.txt",params,params_dtypes,params_values)
        save_params("data.txt",params,params_dtypes,params_values)


        #plot
        self.fig = Figure(figsize=(5, 4), dpi=50)
        self.trainingplot=self.fig.add_subplot(111)

        self.tdata=np.array(5)
        for index in range(0,5):
            zzz=random.randint(0,10)
            self.tdata=np.append(self.tdata,zzz)


        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frm)  # A tk.DrawingArea.
        self.canvas.get_tk_widget().pack(expand=0)
        #self.canvas.show()

        dt = threading.Thread(target=self.add_data_thread)
        dt.daemon = True
        dt.start()

        self.ani=animation.FuncAnimation(self.fig,self.draw_thread,interval=1000)



#        t = np.arange(0, 3, .01)
#        self.trainingplot.plot(t)#, 2 * np.sin(2 * np.pi * t))
#        canvas = FigureCanvasTkAgg(self.fig, master=self.frm)  # A tk.DrawingArea.
#        canvas.draw()
#        canvas.get_tk_widget().pack(expand=0)

        #run
        self.root.mainloop()



z = app()
#z.ani = animation.FuncAnimation(z.fig, drawthread, interval=1000)
#z.root.mainloop()
