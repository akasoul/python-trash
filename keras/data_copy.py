import shutil
import numpy as np
import os


pc_path=np.array(
[
"C:\\Users\\Anton\\AppData\\Roaming\\MetaQuotes\\Terminal\\287469DEA9630EA94D0715D755974F1B\\MQL4"
]
)

gd_path=np.array(
[
"C:/Users/Anton/Google Диск/MQL4"
]
)


def copy(src,dst):
    if(os.path.isdir(src)):
        src=os.path.normpath(src)
        dst=os.path.normpath(dst)
        if not os.path.exists(dst):
            os.mkdir(dst)

        for item in os.listdir(src):
            copy(os.path.join(src,item),os.path.join(dst,item))
    else:
        if(os.path.isfile(dst)):
            if(os.path.getsize(src)!=os.path.getsize(dst)):
                if(os.path.getmtime(src)!=os.path.getmtime(dst)):
                    shutil.copy(src, dst)
        else:
            shutil.copy(src, dst)


for i in range(pc_path.size):
    src=pc_path[i]
    dst=gd_path[i]

    copy(src,dst)



