#!/usr/bin/python
from PIL import Image
import os, sys, time

path = "C:\\Users\\dabes\\Desktop\\TensorFlow\\NIST_data\\0\\hsf_0\\"
dirs = os.listdir( path )

def resize():
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            print (f)
            print('e', e)
            im.thumbnail((32,32), Image.ANTIALIAS)
            im.save(f + ' 32x32.png', 'PNG', quality=100)
            print('waiting;')
            time.sleep(15)

resize()
