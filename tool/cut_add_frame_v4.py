#!/usr/bin/env python
# coding: utf-8

import os
import sys
import numpy as np
import cv2
import re
import pdb
from PIL import Image
from PIL import ImageFile 
from os import mkdir
from os.path import isdir
from skimage import io

'''
========================================
Version 4

Another version for tif file split
Combine all possible conditions
========================================
'''




path = '/home/GDDC-CV1/Desktop/whole_dataset/image'
path_out='/home/GDDC-CV1/Desktop/whole_dataset/image_512'
case=1
colour_mode = 1
# 1-> colour mode, 
# 0-> gray mode, 
#-1-> alpha mode

address_list=[]
pattern=re.compile(r'.*2.tif')
for home, dirs, files in os.walk(path):
    for filename in files:
        if pattern.findall(filename):
            address_list.append(os.path.join(home, filename))


for image in address_list:
    a, b = os.path.splitext(image)
    img_name=image.split('/')[-1].split('.')[0]

    if case ==1: 
        img = io.imread(image)
        img = np.array(img) 
        # This line is for picture has [F, F, T ,,,,,]
        #img = 255 * np.array(img).astype('uint8')
    elif case == 2:
        img=cv2.imread(image, colour_mode)
    else:
        img = Image.open(image)
    pdb.set_trace()
    width=img.shape[0]
    hight=img.shape[1]
    #width, hight = img.size
    w = 512  #切割成1000*1000
    id = 1
    i = 0
    while (i + w <= hight):
        j = 0
        while (j + w <= width):
            if len(img.shape) == 3:
                new_img = img[i:i+w, j:j+w, :]
                if new_img.shape != (w,w,3):
                    pass
                else:
                    try:
                        cv2.imwrite( path_out + '/' + img_name + "_" + str(id) + b , new_img, [cv2.IMWRITE_JPEG_QUALITY, 100])
                    except:
                        print('error')


            elif len(img.shape) == 2:
                new_img = img[i:i+w, j:j+w]

                if new_img.shape != (w,w):
                    pass
                else:
                    try:
                        cv2.imwrite( path_out + '/' + img_name + "_" + str(id) + b , new_img, [cv2.IMWRITE_JPEG_QUALITY, 100])
                    except:
                        print('error')
            
            id += 1
            j += w   #滑动步长
        i = i + w





