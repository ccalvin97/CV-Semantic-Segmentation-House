#!/usr/bin/env python
# coding: utf-8

import os
import sys
import numpy as np
import cv2
import re
import pdb
from feature_engineering import *


'''
path = data dir for train_x or val_x
path_label = data dir for train_y or val_y
out_x: val x dir
out_y: val y dir
'''
path = '/home/GDDC-CV1/Desktop/data_1024/val_y_png'
#path_label='/home/GDDC-CV1/Desktop/data_1024/train_y'
#out_x='/home/GDDC-CV1/Desktop/data_512/val_x'
#out_y='/home/GDDC-CV1/Desktop/data_512/val_y'
out = '/home/GDDC-CV1/Desktop/data_1024/new'
#out_y='/home/GDDC-CV1/Desktop/data_1024/train_y_png'
ratio=0.3
pic_type='png'
pic_type_out='png'
binary_pic='yes'


#data_train_val_split(path, path_label, out_x, out_y)
#check X data
#res=check_x(path, path_label).check_data_dim()
#delete_invalid_data_dim(path, path_label, res)
#check_y(path, path_label).check_y_data()
#check_y(path, path_label).modify_y_data()
#data_resize(path, path_label)
#res=check_x_size(path, path_label).check_size()
#check_x_size(path, path_label).delete_small_data(res)
#res=check_black(path, path_label).check_y_data()
#check_black(path, path_label).delete_black_data(res)
#blk_white(out, path_label).convert_blk_white()
#size_change(path_label).convert_type()
#type_change(path_label, path, out, out_y, 'tif').convert_type()
#fill_transform(path, out, 1024, 'jpeg').convert_type()
type_change_one(path, out, pic_type, pic_type_out, binary_pic).convert_type()



