import os

a=os.listdir()
for i in a:
    if i.find(".png")<0:
        os.rename(i,i+".png")