import shutil
import numpy as np
import os


pc_path=np.array(
[
"C:/Users/Anton/AppData/Roaming/MetaQuotes/Terminal/287469DEA9630EA94D0715D755974F1B/MQL4/"
]
)

gd_path=np.array(
[
"C:/Users/Anton/Google Диск/MQL4"
]
)


for i in range(pc_path.size):
    src=pc_path[i]
    dst=gd_path[i]
    if(os.path.isfile):
        shutil.copy(src,dst)
    else:
        shutil.rmtree(dst)
        shutil.copytree(src,dst)

