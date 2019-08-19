import socket
import sys
from google.cloud import texttospeech


def parseString(data, delimiter):
        prevIndex = 0
        decodedData = data  # .decode('utf-8')
        delimiter = delimiter.encode()[0]
        count = None
        index = 0
        for i in range(0, len(decodedData)):
                if (decodedData[i] == delimiter):
                        res=decodedData[prevIndex:i].decode()
                        count=int(res)
                        prevIndex = i + 1
                        index = index + 1
                        if (index == 1):
                                break
        data = data[prevIndex:len(data)]
        return count, data



addr='194.67.87.166'
#addr='localhost'

port=9072

_SIZE=32


mes="Аудио для отладки"
mes=mes.encode('utf-8')


cmd="tr ru de "
count=int(0.99+((len(mes)+len(cmd)+len("00000"))/_SIZE))
cmd+=str("%.5d" % count)
cmd+=" "
cmd=cmd.encode('utf-8')




sock = socket.socket()
sock.connect((addr, port))
sock.send(cmd)
sock.send(mes)
data = sock.recv(_SIZE)
count,data=parseString(data," ")
for i in range(0,count-1):
        data+=sock.recv(_SIZE)
sock.close()
#data=data.decode()

cmd="sp de-DE 0.9 "
count=int(0.99+((len(data.decode())+len(cmd)+len("00000"))/_SIZE))
cmd+=str( count )
cmd+=" "
cmd=cmd.encode('utf-8')

mes=data

sock = socket.socket()
sock.connect((addr, port))
sock.send(cmd)
sock.send(mes)
data = sock.recv(_SIZE)
count,data=parseString(data," ")
print("Receiving ",count)

for i in range(0, count - 1):
        data += sock.recv(_SIZE)

sock.close()
#data=data.decode()
sock.close()
#data=data.decode()

#data=audioresponse
fname = '1608cout.wav'
with open(fname, 'wb') as output:
        output.write(data)

