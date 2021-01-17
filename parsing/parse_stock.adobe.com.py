import requests
import numpy as np
import urllib
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
import json

url="https://stock.adobe.com/ru/search?filters%5Bcontent_type%3Aphoto%5D=1&filters%5Bcontent_type%3Aillustration%5D=1&filters%5Bcontent_type%3Azip_vector%5D=1&filters%5Bcontent_type%3Avideo%5D=1&filters%5Bcontent_type%3Atemplate%5D=1&filters%5Bcontent_type%3A3d%5D=1&filters%5Bcontent_type%3Aimage%5D=1&filters%5Borientation%5D=square&filters%5Borientation_type%5D%5Bis_square%5D=true&k=grass&order=relevance&safe_search=1&search_type=pagination&limit=100&search_page={0}&get_facets=0"
country="8"

numbers=np.empty(0)
numbersInfo=np.empty(0)

images=[]
counter=0
for i in range(1,100):
    print("page ",i)
    response = requests.get(url.format(i))
    startPos=0
    endPos=0



    startPos = response.text.find('<div class="flex-1">', startPos)
    while(startPos != -1):

        startPos=response.text.find('<img src="',startPos)
        endPos=response.text.find('.jpg',startPos)
        if(startPos>0 and endPos>0):
            startPos += 10
            endPos += 4
            imgURL=response.text[startPos:endPos]
            if(imgURL.find('https') != -1):
                images.append(imgURL)
                resource = None
                try:
                    resource=urllib.request.urlopen(imgURL)
                except:
                    break
                else:
                    output = open(str(counter)+".jpg", "wb")
                    output.write(resource.read())
                    output.close()
                    counter += 1
                    startPos=endPos

images=np.array(images)
np.savetxt("images.txt",images,delimiter='\n')