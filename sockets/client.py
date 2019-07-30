import socket

sock = socket.socket()
sock.connect(('185.58.204.228', 9090))
sock.send(b'someshit!')

data = sock.recv(1024)
sock.close()

print(data)