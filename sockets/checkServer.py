import socket
import time
import os

addr='localhost'
port=9073
delay_s=3

def main():
        sock=socket.socket()
        try:
            sock.connect((addr,port))
        except:
            print("launching server")
            os.system("server.py")
            time.sleep(delay_s)
        else:
            print("server is already running")
            sock.close()
            time.sleep(delay_s)

while True:
    main()