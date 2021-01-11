import requests
import numpy as np
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
import json

url="https://www.neberitrubku.ru/nomer-telefona/"
country="8"

numbers=np.empty(0)
numbersInfo=np.empty(0)

def parseResponse(response):
    startPos = response.find('<div class="ratings">')
    if (startPos > 0):
        endPos = response.find('</div>', startPos)
        if (endPos > 0):
            data = response[startPos:endPos]

            dataParsed = ""
            endPos = 0
            startPos = 0
            while (startPos != -1):
                startPos = data.find('<li', endPos)
                if (startPos > 0):
                    startPos = data.find('>', startPos)
                    if (startPos > 0):
                        endPos = data.find('</li>', startPos)
                        dataParsed += data[startPos + 1:endPos] + " "
            return dataParsed
    return ""



errors=[]
defaultNumbers = []
#for j in range(0, 500):
for j in range(0, 10000000):
    number = str.format("{0:7d}", j)
    number = number.replace(" ", "0")
    defaultNumbers.append(number)

defaultCodes=[]
for i in range(1,1000):
    defaultCodes.append(str.format("{0:3d}", i).replace(" ","0"))


validCodes=[]
def checkCodes(code):
    newURL = url + country + code + "0000001"
    response = requests.get(newURL)
    if (response.status_code == 404):
        #print(newURL)
        return
    findpos = response.text.find("Invalid phone number")
    if (findpos > 0):
        #print(newURL)
        return
    validCodes.append(code)



pool = ThreadPool(1000)
pool.map(checkCodes,defaultCodes)

for i in range(0, validCodes.__len__()):
    print(validCodes[i])
    numbersForRequests=[]
    countryCode="8"
    code=str(validCodes[i])

    info={}

    def sendRequest(number):
        newURL = url + countryCode + code + number
        #print(newURL)
        response = None
        try:
            response=requests.get(newURL)
        except:
            print("pass at", newURL)
            errors.append(newURL)
            return

        data = parseResponse(response.text)
        if (data.find('x') > 0):
            info[countryCode + code + number]=data


    pool = ThreadPool(1000)
    pool.map(sendRequest, defaultNumbers)

    with open(str(code)+'.txt', 'w') as f:
        json.dump(info, f)

npErrors=np.array(errors)
np.savetxt("errors.txt",npErrors,delimiter='\n')

