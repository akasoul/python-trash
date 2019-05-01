from tkinter import *
import tensorflow as tf
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
    def thread_check_path(self):
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

    def run_tf(self):
        with tf.Session() as self.sess:
            # Model init
            self.sess.run(tf.global_variables_initializer())

    def block_trs_ui(self):
        ui=[self.ed_trs_droprate,
            self.ed_trs_l1,
            self.ed_trs_l2,
            self.ed_trs_ls]
        for i in ui:
            i['state']='disabled'
            i['bg']='silver'

    def unblock_trs_ui(self):
        ui=[self.ed_trs_droprate,
            self.ed_trs_l1,
            self.ed_trs_l2,
            self.ed_trs_ls]
        for i in ui:
            i['state']='enabled'
            i['bg']='white'

    def on_click_run_btn(self,event):
        self.block_trs_ui()

    def on_changed_path(self, event):
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

    def check_for_dtype(self, event, ui_index, format):
        ui_trs = [
            self.ed_trs_epoches,
            self.ed_trs_stoperror,
            self.ed_trs_ls,
            self.ed_trs_l1,
            self.ed_trs_l2,
            self.ed_trs_droprate]
        i=ui_index
        value=ui_trs[i].get(1.0,END)
        value = value.rstrip()
        try:
            format(value)
        except:
            ui_trs[i]['bg']='red'
        else:
            ui_trs[i]['bg']='white'

    def thread_add_data(self):
        mod=1
        while True:
            self.tdata=np.append(self.tdata,random.randint(0-mod,10+mod))
            mod=mod+1
            time.sleep(1)

    def thread_draw(self, i):
        self.trainingplot.clear()
        min_index=np.argmin(self.tdata)
        self.trainingplot.plot(self.tdata)
        ymin,ymax=self.trainingplot.get_ylim()
        #self.trainingplot.plot([ymin,ymax])
        self.trainingplot.axvline(x=min_index, color='k', linestyle='--')


    def init_interface(self):
        self.root = Tk()
        self.root.minsize(width=600,height=500)

        self.s_indatapath=''
        self.s_outdatapath=''
        self.s_inputpath=''

        self.new_path=True

        self.frm=Frame(self.root,bg='white',bd=5,height=200, width=300)
        self.btn_train=         Button(self.root,height=1,width=10,text='train')
        self.btn_run=         Button(self.root,height=1,width=10,text='run')
        #file paths
        self.lbl_indatapath=    Label(self.root,height=1,width=12,font='Arial 11',bg="white", fg="black",text='in_data fname :',anchor=W, justify=LEFT)
        self.lbl_outdatapath=   Label(self.root,height=1,width=12,font='Arial 11',bg="white", fg="black",text='out_data fname:',anchor=W, justify=LEFT)
        self.lbl_inputpath=     Label(self.root,height=1,width=12,font='Arial 11',bg="white", fg="black",text='input fname   :',anchor=W, justify=LEFT)
        self.ed_indatapath  = Text(self.root, height=1, width=15, font='Arial 11', wrap=WORD)
        self.ed_outdatapath = Text(self.root, height=1, width=15, font='Arial 11', wrap=WORD)
        self.ed_inputpath   = Text(self.root, height=1, width=15, font='Arial 11', wrap=WORD)
        #training settings
        self.lbl_trainingsettings=  Label(self.root,height=1,width=12,font='Arial 11',bg="white", fg="black",text='training settings:',anchor=W, justify=LEFT)
        self.lbl_trs_epoches=       Label(self.root,height=1,width=12,font='Arial 11',bg="white", fg="black",text='epoches:',anchor=W, justify=LEFT)
        self.lbl_trs_stoperror=     Label(self.root,height=1,width=12,font='Arial 11',bg="white", fg="black",text='stop error:',anchor=W, justify=LEFT)
        self.lbl_trs_ls=            Label(self.root,height=1,width=12,font='Arial 11',bg="white", fg="black",text='training speed:',anchor=W, justify=LEFT)
        self.lbl_trs_l1=            Label(self.root,height=1,width=12,font='Arial 11',bg="white", fg="black",text='l1:',anchor=W, justify=LEFT)
        self.lbl_trs_l2=            Label(self.root,height=1,width=12,font='Arial 11',bg="white", fg="black",text='l2:',anchor=W, justify=LEFT)
        self.lbl_trs_droprate=      Label(self.root,height=1,width=12,font='Arial 11',bg="white", fg="black",text='drop rate:',anchor=W, justify=LEFT)
        self.ed_trs_epoches=        Text(self.root,height=1,width=12,font='Arial 11', wrap=WORD)
        self.ed_trs_stoperror=      Text(self.root,height=1,width=12,font='Arial 11', wrap=WORD)
        self.ed_trs_ls=             Text(self.root,height=1,width=12,font='Arial 11', wrap=WORD)
        self.ed_trs_l1=             Text(self.root,height=1,width=12,font='Arial 11', wrap=WORD)
        self.ed_trs_l2=             Text(self.root,height=1,width=12,font='Arial 11', wrap=WORD)
        self.ed_trs_droprate=       Text(self.root,height=1,width=12,font='Arial 11', wrap=WORD)


        self.ed_indatapath.insert(1.0, 'in_data.txt')
        self.ed_outdatapath.insert(1.0, 'out_data.txt')
        self.ed_inputpath.insert(1.0, 'input.txt')

        self.ed_indatapath.bind('<Key>', self.on_changed_path)
        self.ed_outdatapath.bind('<Key>', self.on_changed_path)
        self.ed_inputpath.bind('<Key>', self.on_changed_path)
        self.ed_indatapath.bind('<Button 1>', self.on_changed_path)
        self.ed_outdatapath.bind('<Button 1>', self.on_changed_path)
        self.ed_inputpath.bind('<Button 1>', self.on_changed_path)
        #training settings
        self.ed_trs_epoches     .bind('<Key>', lambda event, u_index=0, format=int:self.check_for_dtype(event, u_index, format))
        self.ed_trs_stoperror   .bind('<Key>', lambda event, u_index=1, format=float:self.check_for_dtype(event, u_index, format))
        self.ed_trs_ls          .bind('<Key>', lambda event, u_index=2, format=float:self.check_for_dtype(event, u_index, format))
        self.ed_trs_l1          .bind('<Key>', lambda event, u_index=3, format=float:self.check_for_dtype(event, u_index, format))
        self.ed_trs_l2          .bind('<Key>', lambda event, u_index=4, format=float:self.check_for_dtype(event, u_index, format))
        self.ed_trs_droprate    .bind('<Key>', lambda event, u_index=5, format=float:self.check_for_dtype(event, u_index, format))
        self.ed_trs_epoches     .bind('<BackSpace>', lambda event, u_index=0, format=int:self.check_for_dtype(event, u_index, format))
        self.ed_trs_stoperror   .bind('<BackSpace>', lambda event, u_index=1, format=float:self.check_for_dtype(event, u_index, format))
        self.ed_trs_ls          .bind('<BackSpace>', lambda event, u_index=2, format=float:self.check_for_dtype(event, u_index, format))
        self.ed_trs_l1          .bind('<BackSpace>', lambda event, u_index=3, format=float:self.check_for_dtype(event, u_index, format))
        self.ed_trs_l2          .bind('<BackSpace>', lambda event, u_index=4, format=float:self.check_for_dtype(event, u_index, format))
        self.ed_trs_droprate    .bind('<BackSpace>', lambda event, u_index=5, format=float:self.check_for_dtype(event, u_index, format))

        self.btn_run            .bind('<Button 1>', self.on_click_run_btn)

        self.ed_indatapath      .place(x=1440, y=10)
        self.ed_outdatapath     .place(x=140, y=40)
        self.ed_inputpath       .place(x=140, y=70)
        self.btn_train.place(x=10, y=110)
        self.btn_run.place(x=100, y=110)

        self.frm.place(x=270, y=10)
        self.lbl_indatapath.place(x=10, y=10)
        self.lbl_outdatapath.place(x=10, y=40)
        self.lbl_inputpath.place(x=10, y=70)
        self.lbl_trainingsettings.place(x=10, y=150)

        self.lbl_trainingsettings   .place(x=10, y=170)
        self.lbl_trs_epoches        .place(x=10, y=200)
        self.lbl_trs_stoperror      .place(x=10, y=230)
        self.lbl_trs_ls             .place(x=10, y=260)
        self.lbl_trs_l1             .place(x=10, y=290)
        self.lbl_trs_l2             .place(x=10, y=320)
        self.lbl_trs_droprate       .place(x=10, y=350)
        self.ed_trs_epoches         .place(x=140, y=200)
        self.ed_trs_stoperror       .place(x=140, y=230)
        self.ed_trs_ls              .place(x=140, y=260)
        self.ed_trs_l1              .place(x=140, y=290)
        self.ed_trs_l2              .place(x=140, y=320)
        self.ed_trs_droprate        .place(x=140, y=350)



    def load_data(self):
        _file = False
        while _file == False:
            if os.path.isfile(self.s_indatapath):
                if os.path.isfile(self.s_outdatapath):
                    _file = True
        # import data and preprocess
        self.X = np.genfromtxt(self.s_indatapath)
        self.Y = np.genfromtxt(self.s_outdatapath)
        self.X = np.float32(self.X)
        self.Y = np.float32(self.Y)
        data_size = self.ns_datasize
        self.n_inputs = self.X.shape[1]
        self.ns_datasize = self.X.shape[0]
        X = np.reshape(self.X, [data_size, self.ns_inputs])
        Y = np.reshape(self.Y, [data_size, self.ns_outputs])

    def __init__(self):
        self.init_interface()

        network_settings={'data_size': 0,
                          'data_size': 0,
                          'data_size': 0,
                          'data_size': 0,
                          }

#settings input\output
        t = threading.Thread(target=self.thread_check_path)
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

        dt = threading.Thread(target=self.thread_add_data)
        dt.daemon = True
        dt.start()

        self.ani=animation.FuncAnimation(self.fig, self.thread_draw, interval=1000)



#        t = np.arange(0, 3, .01)
#        self.trainingplot.plot(t)#, 2 * np.sin(2 * np.pi * t))
#        canvas = FigureCanvasTkAgg(self.fig, master=self.frm)  # A tk.DrawingArea.
#        canvas.draw()
#        canvas.get_tk_widget().pack(expand=0)

        #run
        self.run_tf()
        self.root.mainloop()
        #self.root.withdraw()



z = app()
#z.ani = animation.FuncAnimation(z.fig, drawthread, interval=1000)
#z.root.mainloop()
