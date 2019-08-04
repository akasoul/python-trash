import socket


addr='194.67.87.166'
#addr='localhost'



sock = socket.socket()
sock.connect((addr, 9071))
sock.send(b'qwerty ewqewq')

data = sock.recv(1024)
sock.close()

print(data)