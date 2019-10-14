import shutil
import numpy as np
import os


pc_path=np.array(
[
"C:/Users/Anton/AppData/Roaming/MetaQuotes/Terminal/287469DEA9630EA94D0715D755974F1B/MQL4"
]
)

gd_path=np.array(
[
"C:/Users/Anton/Google Диск/MQL4"
]
)

def copy(src,dst):
    if(os.path.getmtime(src)!=os.path.getmtime(dst) or os.path.getsize(src)!=os.path.getsize(dst)):
        shutil.copy(src,dst)

for i in range(pc_path.size):
    src=pc_path[i]
    dst=gd_path[i]
    if(os.path.isfile(src)):
        copy(src,dst)
    else:
        content=os.listdir(src)

        for j in content:
            try:
                shutil.copy()
        try:
            pass
            #shutil.rmtree(dst)
        except:
            pass
        shutil.copytree(src,dst,copy_function=copy(src,dst))



