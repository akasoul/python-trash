import socket
import time
import threading

class SockConnection:

    def __init__(self,_port,_log_fname,_code):
        self.is_opened=False
        self.log_fname=_log_fname
        self.code=_code
        self.port=_port
        try:
            self.sock = socket.socket()
        except:
            pass
        else:
            self.is_opened=True
        if(self.is_opened==True):
            try:
                self.sock.bind(('', self.port))
            except:
                pass
            try:
                self.sock.listen(1)
            except:
                pass
            self.Loop()

    def Log(self,data):
        file = open(self.log_fname, 'a+')
        file.write(data)
        file.close()

    def AnswerThread(self,conn,addr):
        try:
            data = conn.recv(1024)
        except:
            pass
        else:
            pwd = data[0:6].decode()
            data = data[6:len(data)].decode()
            if (pwd == self.code):
                self.Log(" " + data)
                ans = str(addr) + str(data)
                ans = ans.encode()
                conn.send(ans.upper())
                conn.close()

    def Loop(self):
        while True:
            try:
                conn,addr=self.sock.accept()
            except:
                pass
            else:
                self.Log("\n"+str(time.ctime())+" "+str(addr))
                print('connected:', addr)
                tt = threading.Thread(target=self.AnswerThread, args=(conn, addr))
                tt.daemon = True
                tt.start()

            continue

z=SockConnection(9071,"log.txt","qwerty")