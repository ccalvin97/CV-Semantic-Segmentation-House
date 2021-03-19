# -*- coding: utf-8 -*-
#!/usr/bin/env python

import cv_unet_v9 as modelling
# from tools.data import test_generator, save_results
import sys
import os
import numpy as np
################ CPU only ##############
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
########################################

path_pred = '/home/GDDC-CV2/Desktop/data_1000/val_x'
path_pred_out = '/home/GDDC-CV2/Desktop/data_1000/pred_test_y'
#checkpoint_name = 'keras_unet_model.h5'



PIXEL = 1000    #set your image size
BATCH_SIZE = 8
lr = 0.001
EPOCH = 1
X_CHANNEL = 3  # training data channel
Y_CHANNEL = 1  # test data channel
smooth = 1
normalisation =True




if __name__ == "__main__":
    """ Prediction Script
    Fill in path for preduction images dir & output dir
    """

    print('Start Prediction Programme')
    cur_dir=os.getcwd()
    paprameter_file = 'model_parameter'
    filename=os.listdir(cur_dir +'/'+ paprameter_file)
    model_weights_name = cur_dir + '/' + paprameter_file + '/'+ filename[0]
    
    
    network=modelling.network(PIXEL=PIXEL, BATCH_SIZE=BATCH_SIZE, lr=lr, EPOCH = EPOCH, X_CHANNEL = X_CHANNEL,
                              Y_CHANNEL = Y_CHANNEL, smooth = smooth, normalisation = normalisation)

    # build model
    model = network.get_unet(pretrained_weights = model_weights_name)


    test_gen = network.test_generator(path_pred)
    images = os.listdir(path_pred)
    images.sort()
    results = model.predict_generator(test_gen,len(images),verbose=1)
    results = results.astype(np.uint8)*255
    modelling.save_results(path_pred_out, results, images)

  
