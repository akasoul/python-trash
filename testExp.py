import matplotlib.pyplot as plt
import numpy as np
import math



widthPixel=1920
heightPixel=1080
diagPixel=math.sqrt(widthPixel*widthPixel + heightPixel*heightPixel)

pW=widthPixel/diagPixel
pH=heightPixel/diagPixel


widthPoint=21.5*pW
heightPoint=21.5*pH

a=math.sqrt(widthPoint*widthPoint + heightPoint*heightPoint)


print(widthPoint,heightPoint)
