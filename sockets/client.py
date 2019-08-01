import socket


addr='194.67.87.166'



sock = socket.socket()
sock.connect((addr, 9090))
sock.send(b'a89skiiurjksa')

data = sock.recv(1024)
sock.close()

print(data)