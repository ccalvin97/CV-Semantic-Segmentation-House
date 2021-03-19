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
ImageFile.LOAD_TRUNCATED_IMAGES = True
from os import mkdir
from os.path import isdir
from skimage import io
import pandas as pd
import random
import shutil
random.seed(2)

'''
========================================
Version 5

check X data dimension == 1000,1000,3
check Y data dimension and range between 0 and 1
Add data split function
Add all black filter function
========================================
'''






def data_resize(path_x, path_y, width=512, height=512):
    '''
    path_x: train_x or val_x 
    path_y: train_y or val_y
    PS. This function will overwrite original data
    '''
    address_list=[]
    pattern=re.compile(r'.*.jpeg')
    for home, dirs, files in os.walk(path_x):
        for filename in files:
            if pattern.findall(filename):
                address_list.append(os.path.join(home, filename))
    try:
        address_list_y=[]
        pattern=re.compile(r'.*.tif')
        for home, dirs, files in os.walk(path_y):
            for filename in files:
                if pattern.findall(filename):
                    address_list_y.append(os.path.join(home, filename))
    except:
        print('Warning - No Label')
    
    for image in address_list:
        img = Image.open(image)
        img = img.resize((width, height),Image.ANTIALIAS) #resize image with high-quality
        img.save(image)

    try:
        for image in address_list_y:
            img = Image.open(image)
            img = img.resize((width, height),Image.ANTIALIAS) #resize image with high-quality
            img.save(image)
    except:
        print('Warning - No Label')


def data_train_val_split(pathX, pathY, out_x, out_y):
    '''
    Function: out_x and out_y are empty dir, the programme will split the data to out dir
    pathX: train x dir
    pathY: train y dir
    out_x: val x dir
    out_y: val y dir
    '''
    X_train_files = os.listdir(pathX)
    X_train_files.sort()
    Y_train_files = os.listdir(pathY)
    Y_train_files.sort()
    
    data = list(zip(X_train_files, Y_train_files))
    random.shuffle(data)
    X_train_files[:], Y_train_files[:] = zip(*data)
    
    
    if len(X_train_files)==len(Y_train_files):
        print('pass')
    else:
        print('file number mismatch')
        sys.exit()
    
    offset = int(len(Y_train_files) * ratio)
    val_file_x = X_train_files[:offset]
    val_file_y = Y_train_files[:offset]
    
    for i in val_file_x:
        shutil.move(pathX + '/' + i, out_x)
    for j in val_file_y:
        shutil.move(pathY + '/' + j, out_y)
    


#######################################################
class check_x():
    def __init__(self, path, path_label):
        self.path = path
        self.path_label = path_label


        address_list=[]
        pattern=re.compile(r'.*.jpeg')
        for home, dirs, files in os.walk(self.path):
            for filename in files:
                if pattern.findall(filename):
                    address_list.append(os.path.join(home, filename))
        self.address_list=address_list

# check X data
    def check_data_dim(self):
        res=[]
        i=0
        for image in self.address_list:
           # a, b = os.path.splitext(image)
            img_name=image.split('/')[-1]
            img = io.imread(image)
            img = np.array(img)
            if img.shape == (512,512,3):
                continue
            else:
                print('dim error')
                pdb.set_trace()
                res.append(img_name)
                i=i+1
        print(i)
        return res
    
    def delete_invalid_data_dim(self, res):
        '''
        path=path of the image
        path_label= path of the image label
        res=invalid image name ex: train1_11.tif
        '''
        for i in res:
            os.remove(self.path+'/'+i)
            os.remove(self.path_label+'/'+i)
    





#######################################################
# Check Y data
class check_y():
    def __init__(self, path, path_label):
        self.path = path
        self.path_label = path_label

        address_list_y=[]
        pattern=re.compile(r'.*.tif')
        for home, dirs, files in os.walk(self.path_label):
            for filename in files:
                if pattern.findall(filename):
                    address_list_y.append(os.path.join(home, filename))

        self.address_list_y = address_list_y
        
        
    def check_y_data(self):
        '''
        Function: Check Y data dim & range
        Target: Force data range from 0-1
        '''
        res=[]
        i=0
        for image in self.address_list_y:
           # a, b = os.path.splitext(image)
            img_name=image.split('/')[-1]
            img = io.imread(image)
            img = np.array(img)
            if img.shape == (1000,1000):
                continue
            else:
                print('dim error')
                res.append(img_name)
                i=i+1

            if ((img>1) | (img<0)).any:
                print('value error')
                pdb.set_trace()
        print(i)
   
    def modify_y_data(self):
        '''
        Function: change dim of label data from [1000,1000,3] -> [1000,1000] and range from 0 - 255
        PS. This function will overwrite original data
        '''
        res=[]
        i=0
        for image in self.address_list_y:
           # a, b = os.path.splitext(image)
            img_name=image.split('/')[-1]
            img = Image.open(image)
            img1 = np.array(img)
            if img1.shape == (512,512):
                continue
            else:
                pdb.set_trace()
                print('dim error')
                img1 = img.convert("L")
                img1=np.array(img1)
                img1 = Image.fromarray((img1).astype(np.uint8))
                img1.save(image)
                i=i+1

        print(i)



#################################################

class check_x_size():
    def __init__(self, path, path_label, threshold=280):
        '''
        check train_x size and delete small size data and corresponding label data
        '''
        self.path = path
        self.path_label = path_label
        self.threshold=threshold
 
        address_list=[]
        pattern=re.compile(r'.*.tif')
        for home, dirs, files in os.walk(self.path):
            for filename in files:
                if pattern.findall(filename):
                    address_list.append(os.path.join(home, filename))
        self.address_list=address_list

    def check_size(self):
        res=[]
        for i in self.address_list:
            img_name=i.split('/')[-1]
            data_size=os.path.getsize(i)
            data_size=data_size/float(1024)
            if data_size<self.threshold:
                res.append(img_name)
                print(img_name) 
        return res

    def delete_small_data(self, res):
        for i in res:

            os.remove(self.path+'/'+i)
            os.remove(self.path_label+'/'+i)

 


# Check black_label data
class check_black():
    def __init__(self, path, path_label):
        self.path = path
        self.path_label = path_label

        address_list_y=[]
        pattern=re.compile(r'.*.tif')
        for home, dirs, files in os.walk(self.path_label):
            for filename in files:
                if pattern.findall(filename):
                    address_list_y.append(os.path.join(home, filename))

        self.address_list_y = address_list_y


    def check_y_data(self):
        '''
        Function: Check Y data black or not
        '''
        res=[]
        i=0
        for image in self.address_list_y:
           # a, b = os.path.splitext(image)
            img_name=image.split('/')[-1]
            img = io.imread(image)
            img = np.array(img)
            if np.max(img)<2:
                res.append(img_name)
                i=i+1
                print(img_name)
        print(i)
        return res

    def delete_black_data(self, res):
        for i in res:

            os.remove(self.path+'/'+i)
            os.remove(self.path_label+'/'+i)



#### Convert black -> white white -> black ####
class blk_white():
    def __init__(self, out, path_label):
        self.out = out
        self.path_label = path_label

        address_list_y=[]
        pattern=re.compile(r'.*.tif')
        for home, dirs, files in os.walk(self.path_label):
            for filename in files:
                if pattern.findall(filename):
                    address_list_y.append(os.path.join(home, filename))

        self.address_list_y = address_list_y


    def convert_blk_white(self):
        '''
        Function: Check Y data black or not
        '''
        res=[]
        i=0
        for image in self.address_list_y:
           # a, b = os.path.splitext(image)
            img_name=image.split('/')[-1]
            img = io.imread(image)
            img = np.array(img)
            img = 255 - img
            img = Image.fromarray(img)
            img.save(self.out+'/'+img_name)




#### Convert tif -> jpg + 二值化 ####
class type_change():
    def __init__(self, path_label, path, out, out_y, pic_type):
        self.path = path
        self.path_label = path_label
        self.out = out
        self.out_y = out_y
        self.pic_type = pic_type
        '''
        out: output x dir
        path: input x dir
        path_label: dir y dir
        out_y: dir  
        pic_type: str, picture type, ex: 'tif'

        '''
        address_list=[]
        pattern=re.compile(r'.*.'+self.pic_type)
        for home, dirs, files in os.walk(self.path):
            for filename in files:
                if pattern.findall(filename):
                    address_list.append(os.path.join(home, filename))
        self.address_list=address_list
        

        address_list_y=[]
        pattern=re.compile(r'.*.'+self.pic_type)
        for home, dirs, files in os.walk(self.path_label):
            for filename in files:
                if pattern.findall(filename):
                    address_list_y.append(os.path.join(home, filename))
        self.address_list_y = address_list_y


    def convert_type(self):
        for image in self.address_list_y:
            img_name=image.split('/')[-1]            
            img = cv2.imread(image, 0)  #这里选择-1，不进行转化
            #img=Image.open(image)
            #img = img.convert('1')
            img = np.array(img)
            ret, img = cv2.threshold(img, 2, 1, 0)
            img = Image.fromarray(img, mode ='P')
            img.save(self.out_y+'/'+img_name.split('.')[0]+".png")
        '''
        for image in self.address_list:
            img_name=image.split('/')[-1]
            img = cv2.imread(image,-1)  #这里选择-1，不进行转化
            cv2.imwrite(self.out+'/'+img_name.split('.')[0]+".png", img)
        '''



#### Picture Mirror Fill Function ####
class fill_transform():
    def __init__(self,  path, out, w, pic_type):
        self.out = out
        self.path = path
        self.w = w
        self.pic_type = pic_type
        '''
        out: output dir
        path: input dir
        w: wide and long of output picture
        pic_type: str, picture type, ex: 'tif'
        '''
        address_list=[]
        pattern=re.compile(r'.*.'+self.pic_type)
        for home, dirs, files in os.walk(self.path):
            for filename in files:
                if pattern.findall(filename):
                    address_list.append(os.path.join(home, filename))
        self.address_list=address_list



    def convert_type(self):
        for image in self.address_list:
            a, b = os.path.splitext(image)
            img_name=image.split('/')[-1].split('.')[0]
            img = Image.open(image)
            width, hight = img.size
            fill_w = int((self.w - width)/2)
            img = np.array(img)
            img = cv2.copyMakeBorder(img, fill_w, fill_w, fill_w, fill_w,  borderType=cv2.BORDER_REFLECT)
            cv2.imwrite( self.out + '/' + img_name + "_"  + b , img, [cv2.IMWRITE_JPEG_QUALITY, 100])





#### Convert pic type  ####
class type_change_one():
    def __init__(self ,path, out, pic_type, pic_type_out, binary_pic):
        self.path = path
        self.out = out
        self.pic_type = pic_type
        self.pic_type_out = pic_type_out
        self.binary_pic = binary_pic
        '''
        out: output x dir
        path: input x dir
        path_label: dir y dir
        out_y: dir
        pic_type: str, picture type, ex: 'tif'
        binary_pic: str, yes-> 二值圖, no->正常
        '''
        address_list=[]
        pattern=re.compile(r'.*.'+self.pic_type)
        for home, dirs, files in os.walk(self.path):
            for filename in files:
                if pattern.findall(filename):
                    address_list.append(os.path.join(home, filename))
        self.address_list=address_list


    def convert_type(self):

        for image in self.address_list:
            img_name=image.split('/')[-1]
            img = cv2.imread(image,-1)  #这里选择-1，不进行转化

            if self.binary_pic == 'yes':
                img = np.array(img)
                ret, img = cv2.threshold(img, 2, 1, 0)
                img = Image.fromarray(img, mode ='P')
                img.save(self.out+'/'+img_name.split('.')[0]+"."+self.pic_type_out)
            else:
                cv2.imwrite(self.out+'/'+img_name.split('.')[0]+"."+self.pic_type_out, img)












