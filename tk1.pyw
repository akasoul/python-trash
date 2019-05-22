from tkinter import *

root = Tk()

Label(root, text="This is a test").grid(row=0, column=0)

mytext1 = Text(root, width=30, height=5)
mytext1.grid(row=1, column=0, sticky="nsew")

mytext2 = Text(root, width=30, height=5)
mytext2.grid(row=2, column=0, sticky="nsew")

#root.columnconfigure(0, weight=1)
#root.rowconfigure(0, weight=0) # not needed, this is the default behavior
#root.rowconfigure(1, weight=1)
#root.rowconfigure(2, weight=1)

root.mainloop()