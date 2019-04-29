from time import sleep
a=1;


def someshit(somedata):

    global a
    a = somedata*2


while(True):
        someshit(a)
        print(a)
        sleep(1)