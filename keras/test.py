import requests

url = 'https://drive.google.com/file/d/1qv_wW4vQiMu3DwC8LHDlooj0r-LMlTft/view?usp=sharing'
myfile = requests.get(url, allow_redirects=True)
open('d:/outdata.txt', 'wb').write(myfile.content)