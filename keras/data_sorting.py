import shutil
import numpy as np
import os

terminal_path="C:/Users/Anton/AppData/Roaming/MetaQuotes/Terminal/287469DEA9630EA94D0715D755974F1B/"
google_drive_path="C:/Users/Anton/Google Диск/"


filenames_in=np.array(
[
terminal_path+"MQL4/"
]
)

filenames_out=np.array(
[
#google_drive_path+"MQL4"
"D:/MQL_4/"
]
)


for i in range(filenames_in.size):
    os.chdir(filenames_in[i])
    for file1 in os.listdir():

        if(os.path.isfile(file1)):
            shutil.copy(filenames_in[i]+file1,filenames_out[i]+file1)
        if(os.path.isdir(file1)):

            os.chdir(file1)
            file1+="/"
            for file2 in os.listdir():

                if (os.path.isfile(file2)):
                    shutil.copy(filenames_in[i] +file1+ file2, filenames_out[i] +file1+ file2)
                if (os.path.isdir(file2)):

                    os.chdir(file2)
                    file2+="/"

                    for file3 in os.listdir():
                        if (os.path.isfile(file3)):
                            shutil.copy(filenames_in[i] +file1+ file2+ file3, filenames_out[i] +file1+ file2+ file3)
                        if (os.path.isdir(file3)):

                            os.chdir(file3)
                            file3+="/"

                            for file4 in os.listdir():
                                if (os.path.isfile(file4)):
                                    shutil.copy(filenames_in[i] +file1+ file2+file3+ file4, filenames_out[i] +file1+ file2+file3+ file4)



