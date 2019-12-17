from os import path,mkdir,listdir,makedirs,chdir
from shutil import copy as _copy
from numpy import array,unique,append




chdir("C:\ProgramData\Dinamika\Database")
dir="Декабрь 2019\\"
strtofind="122019"







def copy(src,dst):
    if(path.isdir(src)):
        src=path.normpath(src)
        dst=path.normpath(dst)
        if not path.exists(dst):
            mkdir(dst)

        for item in listdir(src):
            copy(path.join(src,item),path.join(dst,item))
    else:
        if(path.isfile(dst)):
            if(path.getsize(src)!=path.getsize(dst)):
                if(path.getmtime(src)!=path.getmtime(dst)):
                    _copy(src, dst)
        else:
            _copy(src, dst)





strarr=None
a=listdir()
for i in a:
    if path.isdir(i):
        b=listdir(i)
        for j in b:
            if(j.find(strtofind)>0):
                try:
                    if(strarr==None):
                        strarr=array(i)
                    else:
                        strarr=append(strarr,i)
                except:
                    strarr = append(strarr, i)

strarr=unique(strarr)

if(not path.isdir(dir)):
    makedirs(dir)

for i in strarr:
    if(i!=None):
        src=i
        dst=dir+i
        copy(src, dst)
        #shutil.rmtree(src,ignore_errors=True)
print(strarr)
