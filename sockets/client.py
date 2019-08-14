import socket
import sys

#addr='194.67.87.166'
addr='localhost'

_SIZE=32

#
mes="О моих волосах можно сказать следующее: ранняя седина или облысение, тонкие, блестящие, прямые, светлые, рыжие или соломенного цвета. Тест."
mes=mes.encode('utf-8')
cmd=b'sp ru-RU 0.95 '
count=int(0.99+((len(mes)+len(cmd))/_SIZE))
cmd+=str( count ).encode('utf-8')
#

sock = socket.socket()
sock.connect((addr, 9071))


sock.send(cmd)
sock.send(mes)

data = sock.recv(144)
sock.close()

print(len(mes)," ",sys.getsizeof(mes)," ",mes)
print(len(data)," ",sys.getsizeof(data)," ",data)
