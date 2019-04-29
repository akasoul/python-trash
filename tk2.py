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

    def __init__(self):
        root = Tk()
        root.minsize(500,500)

        btn = Button(root,  # родительское окно
                     text="Click me",  # надпись на кнопке
                     width=10, height=5,  # ширина и высота
                     bg="white", fg="black")  # цвет фона и надписи

        btn.place(x=10, y=5)
        root.mainloop()


z = app()

