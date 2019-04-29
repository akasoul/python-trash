from tkinter import *
import matplotlib as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
import threading

a = 2


#a=2  # type: int

def app():
    root = Tk()
    root.minsize(500,500)
    frame1=Frame(root,bg='green',bd=50)

    redraw = True

    fig = Figure(figsize=(3, 2), dpi=100)
    t = np.arange(0, 3, .01)
    ax=fig.add_subplot(111)
    ax.plot(t, a * np.sin(a * np.pi * t))


    canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
    canvas.draw()
    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=0)

    def Hello(event):
        print("someshit")
        #a=1
        global a
        a *= 2
        print(a)
        ax.plot(t, a * np.sin(a * np.pi * t))
        canvas.draw()

    def redrawprocess():
            global a
            print(a)
            a *= 2
            ax.cla()
            ax.plot(t, a * np.sin(a * np.pi * t))
            canvas.draw()



    btn = Button(root,  # родительское окно
                 text="Click me",  # надпись на кнопке
                 width=30, height=5,  # ширина и высота
                 bg="white", fg="black")  # цвет фона и надписи

    btn.place(x=0, y=220)



    root.mainloop()
    def start1():
        _t = threading.Thread(target=redrawprocess)
        _t.daemon = True
        _t.start()

app()
app.start1()
