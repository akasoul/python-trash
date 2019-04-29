from tkinter import *
import matplotlib as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
import threading

a = 2


#a=2  # type: int

class app:

    def press_button(self,event):
        fpath=self.dpath.get(1.0,END)
        print(fpath)
        fpath=fpath.rstrip()
        fpath=fpath.lstrip()
        try:
            a = np.genfromtxt(fpath)
        except:
            print("error")
        print(a)
        v = StringVar()
        Label(self.lbl, textvariable=v).pack()

        v.set(a)

    def __init__(self):
        self.root = Tk()
        self.root.minsize(500,500)
        self.lbl=Label(self.root,  # родительское окно
                     text="Click me",  # надпись на кнопке
                     width=10, height=5,  # ширина и высота
                     bg="white", fg="black")  # цвет фона и надписи
        self.btn = Button(self.root,  # родительское окно
                     text="Click me",  # надпись на кнопке
                     width=10, height=5,  # ширина и высота
                     bg="white", fg="black")  # цвет фона и надписи
        self.dpath = Text(self.root,height=7,width=7,font='Arial 14',wrap=WORD)
        self.dpath.insert(1.0,'data.txt')
        self.btn.bind("<Button 1>",self.press_button)
        self.dpath.place(x=10, y=200)
        self.btn.place(x=10, y=5)
        self.lbl.place(x=100, y=200)
        self.root.mainloop()


z = app()

