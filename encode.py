path = "/Users/antonvoloshuk/Documents/Dinamika/Database/1/24012017112258.demo"
a=None
with open(path,mode='rb') as file:
    a=file.read()
for i in range(0, len(a)):
    #b=int.from_bytes(a[i],byteorder='little')
    print(a[i])