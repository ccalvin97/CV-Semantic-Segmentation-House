# -*- coding: utf-8 -*-
#!/usr/bin/env python



'''
===========================
V10:
    version from colab
    solve the problem of indentation
    All basic function done, prediction, visualisation, training and validation
    add mIOU criteria
    ok to run, mute warning and hyperparameter testing
    load weight
    load model
===========================

'''
    

from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

################ CPU only ##############
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
########################################
import datetime
import numpy as np
import cv2
from os import mkdir
from os.path import isdir
import pandas as pd
import re
import csv
import pdb
import argparse
import cv_unet_v10 as modelling
from colorama import  init, Fore, Back, Style
#from keras import backend as K
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow.compat.v1 as tf #使用1.0版本的方法
from tensorflow.keras.models import load_model
#tf.disable_v2_behavior() #禁用2.0版本的方法
#import tensorflow as tf
# mute warning
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

'''
pathX = '/content/gdrive/My Drive/CV_image_segmentation/out/train/'     # training x^M
pathY = '/content/gdrive/My Drive/CV_image_segmentation/out/label/'     # training label^M
path_pred = '/content/gdrive/My Drive/CV_image_segmentation/out/prediction_x_example_black'# prediction x
'''

init(autoreset=True)  
class Colored(object):  
    def red(self, s):  
        return Fore.RED + s + Fore.RESET  
    
    def yellow(self, s):  
        return Fore.YELLOW + s + Fore.RESET 
color = Colored() 


def main():
    # Get arguments
    parser = argparse.ArgumentParser(description='unet main py')
    parser.add_argument('-pathX', '--pathX', type=str, default='',
            help='data dir for training PS: do not add / in the end ex: /content/gdrive/My Drive/CV_image_segmentation/out/train') 
    parser.add_argument('-pathY', '--pathY', type=str, default='',
                help='data dir for training PS: do not add / in the end ex: /content/gdrive/My Drive/CV_image_segmentation/out/label')
    parser.add_argument('-path_pred', '--path_pred', type=str, default= '',
            help='data dir for pred x ex:/content/gdrive/My Drive/CV_image_segmentation/out/prediction_x_example_black')    
    parser.add_argument('-path_pred_out', '--path_pred_out', type=str, default= '',
            help='out dir of prediction picture ex:/content/gdrive/My Drive/CV_image_segmentation/out/prediction_out')
    parser.add_argument('-pathX_val', '--pathX_val', type=str, default='',
            help='data dir for val PS: do not add / in the end ex: /content/gdrive/My Drive/CV_image_segmentation/out/train')
    parser.add_argument('-pathY_val', '--pathY_val', type=str, default='',
                help='data dir for val PS: do not add / in the end ex: /content/gdrive/My Drive/CV_image_segmentation/out/label')
    parser.add_argument('-is_predict', '--is_predict', type=int, default=0,
                help='whether predict the data, 1-yes 0-no')
    parser.add_argument('-load_weight', '--load_weight', type=str, default='no',
            help='no or yes to load the model weight')
    parser.add_argument('-load_model', '--load_model', type=str, default='no',
            help='no or yes to load the model model')


    args = parser.parse_args()

    print(color.yellow('Start Unet main.py')) 

    PIXEL = 1000    #set your image size
    BATCH_SIZE = 8
    lr = 0.0001
    EPOCH = 1
    X_CHANNEL = 3  # training data channel
    Y_CHANNEL = 1  # test data channel
   # X_NUM = 25  # your traning data number
    smooth = 1
    normalisation = True
    cur_dir=os.getcwd()
    paprameter_file = 'model_parameter'
    if not os.path.isdir(cur_dir+'/'+paprameter_file):
        os.mkdir(cur_dir+'/'+paprameter_file)

    model_structure = 'model_structure'
    if not os.path.isdir(cur_dir+'/'+ model_structure):
        os.mkdir(cur_dir+'/'+model_structure)

    
    images = os.listdir(args.pathX)
    X_NUM = len(images)
    train_steps =X_NUM//BATCH_SIZE
    images_val = os.listdir(args.pathX_val)
    valX_NUM = len(images_val)
    val_steps = valX_NUM // BATCH_SIZE

    network=modelling.network(PIXEL=PIXEL, BATCH_SIZE=BATCH_SIZE, lr=lr, EPOCH = EPOCH, X_CHANNEL = X_CHANNEL, 
            Y_CHANNEL = Y_CHANNEL, smooth = smooth, normalisation = normalisation)
    tf.keras.backend.clear_session()
    
    filename=os.listdir(cur_dir +'/'+ paprameter_file)
    model_weights_name = cur_dir + '/' + paprameter_file + '/'+ filename[0]

    if args.load_weight == 'yes':
        model = network.get_unet(pretrained_weights = model_weights_name)
        print('Load Weight - YES')
    if args.load_weight == 'no':
        model = network.get_unet()
    
    if args.load_model == 'yes':
        model = load_model(cur_dir + '/' + model_structure+'/'+'structure_parameter.h5')


    checkpoint = ModelCheckpoint(os.path.join(cur_dir+'/'+paprameter_file,'_'+str(lr)+'_'+"keras_unet_model.h5"),
                             verbose=1, save_best_only=True)
    early_stop = EarlyStopping(monitor="val_loss", patience=10, verbose=1)
    history = model.fit_generator(network.generator(args.pathX, args.pathY, X_NUM),steps_per_epoch=train_steps , validation_data = network.generator(args.pathX_val, args.pathY_val, valX_NUM),
        validation_steps=val_steps, epochs=EPOCH, use_multiprocessing=True, callbacks=[early_stop, checkpoint])
    
    model.save(cur_dir + '/' + model_structure+'/'+'structure_parameter.h5')

    #network.save_model(history, model)
    #visualisation(path_pred, BATCH_SIZE, normalisation, 20)
    if args.is_predict:
        test_gen = network.test_generator(args.path_pred)
        images = os.listdir(args.path_pred)
        images.sort()
        results = model.predict_generator(test_gen,len(images),verbose=1)
        modelling.save_results(args.path_pred_out, results, images)
    
    modelling.plot(history)


if __name__ == '__main__':
    main()
