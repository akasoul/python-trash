import socket

sock = socket.socket()
sock.bind(('', 9090))
sock.listen(1)
while True:


        conn, addr = sock.accept()

        print('connected:', addr)

        while True:
            data = conn.recv(1024)
            if not data:
                break
            pwd=data[0:6].decode()
            data=data[6:len(data)].decode()
            if(pwd=='a89ski'):
                data=data.encode()
                conn.send(data.upper())

        conn.close()