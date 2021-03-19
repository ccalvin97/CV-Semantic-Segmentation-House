#!/usr/bin/env python
# coding: utf-8

# -*- coding: utf-8 -*-


from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import datetime
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, AveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.callbacks import ModelCheckpoint
#from keras import backend as K
import tensorflow.keras.backend as K
from tensorflow.keras.layers import LeakyReLU, ReLU
import cv2
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.metrics import MeanIoU


from PIL import Image

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pdb    
import skimage.io as io

from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.compat.v1.keras.backend  import clear_session
from tensorflow.compat.v1.keras.backend  import get_session
#import tensorflow as tf
import tensorflow.compat.v1 as tf #使用1.0版本的方法
tf.disable_v2_behavior() #禁用2.0版本的方法


class network():
    def __init__(self, BATCH_SIZE, normalisation, X_CHANNEL, Y_CHANNEL, PIXEL, lr, EPOCH, smooth):
        # self.pathX = pathX
       # self.pathY = pathY
        self.BATCH_SIZE = BATCH_SIZE
        self.normalisation = normalisation
        self.X_CHANNEL = X_CHANNEL
        self.Y_CHANNEL = Y_CHANNEL
        self.PIXEL = PIXEL
        self.EPOCH = EPOCH
        self.smooth = smooth
        self.lr = lr

    def generator(self, pathX, pathY, NUM):
        while 1:
            X_train_files = os.listdir(pathX)
            X_train_files.sort()
            Y_train_files = os.listdir(pathY)
            Y_train_files.sort()
            a = (np.arange(1, NUM))
            # print(a)
            # cnt = 0
            X = []
            Y = []
            for i in range(self.BATCH_SIZE):
                index = np.random.choice(a)
                img = cv2.imread(pathX + '/' +X_train_files[index], 1)
                if self.normalisation:
                    img = img / 255.0  # normalization
                img = np.array(img).reshape(self.PIXEL, self.PIXEL, self.X_CHANNEL)
                X.append(img)
                img1 = cv2.imread(pathY + '/' + Y_train_files[index], 2)
                #if self.normalisation:
                img1 = img1 / 255.0  # normalization
                img1 = np.array(img1).reshape(self.PIXEL, self.PIXEL, self.Y_CHANNEL)
                Y.append(img1)
                #cnt += 1
            X = np.array(X)
            Y = np.array(Y)
            yield X, Y


    def dice_coef(self, y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        y_true_f = K.cast(y_true_f, dtype='float32')
        intersection = K.sum(y_true_f * y_pred_f)
        res=(2. * intersection + self.smooth) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f) + self.smooth)
        return res

    def dice_coef_loss(self, y_true, y_pred):
        return 1-self.dice_coef(y_true, y_pred)    

    def dice_p_bce(self, in_gt, in_pred):
        return 0.05*binary_crossentropy(in_gt, in_pred) - self.dice_coef(in_gt, in_pred)

    def true_positive_rate(self, y_true, y_pred):
        return K.sum(K.flatten(y_true)*K.flatten(K.round(y_pred)))/K.sum(y_true)


    def focal_loss(self, gamma=2., alpha=.25):
        def focal_loss_fixed(y_true, y_pred):
            pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
            pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
            return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(K.epsilon()+pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
        return focal_loss_fixed

    def binary_accuracy(self, y_true, y_pred):
        return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)

    def get_unet(self, pretrained_weights = None):
        inputs = Input((self.PIXEL, self.PIXEL, 3)) # 1000*1000*3
        conv1 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        pool1 = AveragePooling2D(pool_size=(2, 2))(conv1)  # pool1=500*500*3
     
        conv2 = BatchNormalization(momentum=0.99)(pool1)
        conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        conv2 = BatchNormalization(momentum=0.99)(conv2)
        conv2 = Conv2D(64, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        conv2 = Dropout(0.02)(conv2)
        pool2 = AveragePooling2D(pool_size=(2, 2))(conv2)  # pool2=250*250*3
     
        conv3 = BatchNormalization(momentum=0.99)(pool2)
        conv3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        conv3 = BatchNormalization(momentum=0.99)(conv3)
        conv3 = Conv2D(128, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        conv3 = Dropout(0.02)(conv3)
        pool3 = AveragePooling2D(pool_size=(2, 2))(conv3)  # pool3=125*125*3
     
        conv4 = BatchNormalization(momentum=0.99)(pool3)
        conv4 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        conv4 = BatchNormalization(momentum=0.99)(conv4)
        conv4 = Conv2D(256, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        conv4 = Dropout(0.02)(conv4)
        pool4 = AveragePooling2D(pool_size=(5, 5))(conv4) #pool4=25*25*3
        ''' 
        conv5 = BatchNormalization(momentum=0.99)(pool4)
        conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        conv5 = BatchNormalization(momentum=0.99)(conv5)
        conv5 = Conv2D(512, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        conv5 = Dropout(0.02)(conv5)
        pool3 = AveragePooling2D(pool_size=(5, 5))(conv4) # conv4 = 25*25, pool4=5*5*3
        '''
        # conv5 = Conv2D(35, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        # drop4 = Dropout(0.02)(conv5)

        pool5 = AveragePooling2D(pool_size=(5, 5))(pool3)  # 25*25
       # pool5 = AveragePooling2D(pool_size=(3, 3))(pool4)  # 1
     
        conv6 = BatchNormalization(momentum=0.99)(pool5)
        conv6 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
     
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
       # up7 = (UpSampling2D(size=(5, 5))(conv7))  # up7=25*25*3
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
        merge7 = concatenate([pool4, conv7], axis=3)
     
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        up8 = (UpSampling2D(size=(5, 5))(conv8))  # conv8=25*25*6, up8=125*125*6
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up8)
        merge8 = concatenate([pool3, conv8], axis=3) # 125*125*9
     
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        up9 = (UpSampling2D(size=(2, 2))(conv9))  # 250*250*9
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up9)
        merge9 = concatenate([pool2, conv9], axis=3) # 250*250*12
     
        conv10 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        up10 = (UpSampling2D(size=(2, 2))(conv10))  # 500*500*12
        conv10 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up10)
     
        conv11 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv10)
        up11 = (UpSampling2D(size=(2, 2))(conv11))  # 1000*1000*12
        conv11 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up11)
     
        # conv12 = Conv2D(3, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv11)
        conv12 = Conv2D(3, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv11)
        conv12 = Conv2D(1, 1, 1, activation='sigmoid')(conv12)
        model = Model(inputs, conv12)
        if pretrained_weights:
            model.load_weights(pretrained_weights, by_name=True)

        print(model.summary())
        model.compile(optimizer=Adam(lr=self.lr, decay=1e-6), loss=self.dice_coef_loss,  metrics=[self.dice_coef, self.binary_accuracy, MeanIoU(num_classes=2), self.true_positive_rate])
       #model.compile(optimizer=Adam(lr=self.lr, decay=1e-6), loss=self.dice_coef_loss, metrics=[self.dice_coef, self.binary_accuracy, MeanIoU(num_classes=2)])
       # model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
        return model
    '''
    def save_model(self, history, model):
        #save your training model
        cur_dir=os.getcwd()
        paprameter_file = 'model_parameter'
        if not os.path.isdir(cur_dir+'/'+paprameter_file):
            os.mkdir(cur_dir+'/'+paprameter_file)
        model.save(cur_dir+'/'+paprameter_file+'/'+'test1.h5')
    
        #save your loss data
        loss = np.array((history.history['loss']))
        np.save(cur_dir+'/'+paprameter_file+'/'+'test1.npy', loss)



    def visualisation(path_pred, index):
        pred_x_files = os.listdir(path_pred)
        a = (np.arange(1, self.X_NUM))
        X = []
        X_visialisation = []
        img = cv2.imread(path_pred + '/' + pred_x_files[index], 1)
        if self.normalisation:
            img = img / 255.0  # normalization
        img = np.array(img).reshape(self.PIXEL, self.PIXEL, self.X_CHANNEL)
        X.append(img)
    
        X = np.array(X)
        X_visialisation.append(img)
        X_visialisation = np.array(X_visialisation)
        predd = model.predict(X)
        predd_visualisation = model.predict(X_visialisation)
          
    
       # fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (24, 24))
        predd_visualisation=predd_visualisation.squeeze(axis=0)
        predd_visualisation=predd_visualisation.squeeze(axis=2)
       # ax1.imshow(img)
       # ax2.imshow(predd_visualisation,cmap='binary')
       # ax2.set_title('Prediction')    
        cur_dir=os.getcwd()
        pred_pic_file = 'pred_pic'
        if not os.path.isdir(cur_dir+'/'+pred_pic_file):
            os.mkdir(cur_dir+'/'+pred_pic_file)


        im = Image.fromarray(predd_visualisation)   
        if im.mode != 'RGB':
            im = im.convert('RGB')
        im.save(cur_dir+'/'+pred_pic_file+'/'+pred_x_files[index]+"_pred.jpeg") 
       

        im = Image.fromarray(img)
        if im.mode != 'RGB':
            im = im.convert('RGB')
        im.save(cur_dir+'/'+pred_pic_file+'/'+pred_x_files[index]+"_label.jpeg")
       ''' 


    def test_generator(self, path_pred):
        images = os.listdir(path_pred)
        images.sort()
        total = len(images)
        i = 0
        print('-'*30)
        print('Creating test images')
        print('-'*30)
        for image_name in images:
            img = io.imread(path_pred + '/' + image_name)
            img = img[:, :, :3]
            img = img.reshape(self.PIXEL,self.PIXEL, 3)
            img = np.reshape(img,(1,)+img.shape)
            if self.normalisation:
                img = img / 255.0  # normalization

            yield img
        print('test_generator done')



def save_results(save_path, npyfile, names):
    """ Save Results
    Function that takes predictions from U-Net model
    and saves them to specified folder.
    """

    for i,item in enumerate(npyfile):
        img = normalize_mask(item)
        img = (img * 255.0).astype('uint8')
        io.imsave(os.path.join(save_path,"pred_"+names[i]),img)

def normalize_mask(mask):
    """ Mask Normalization
    Function that returns normalized mask
    Each pixel is either 0 or 1
    """
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    return mask



def plot(history):

    print('Start Plot')
    tp = history.history["true_positive_rate"]
    val_tp = history.history["val_true_positive_rate"]
    MeanIoU = history.history["mean_io_u"]
    val_MeanIoU = history.history["val_mean_io_u"]
    dice_coef = history.history['dice_coef']
    val_dice_coef = history.history["val_dice_coef"]
    loss_bacc = history.history["binary_accuracy"]
    val_bacc = history.history["val_binary_accuracy"]

    epochs = range(1, len(MeanIoU) + 1)
    plt.style.use('ggplot')

    plt.figure()
    plt.plot(epochs, tp, color="yellow", label="tp")
    plt.plot(epochs, val_tp, color="orange", label="val tp")
    plt.plot(epochs, MeanIoU, color="red", label="train MeanIoU")
    plt.plot(epochs, val_MeanIoU, color="darkred", label="val MeanIoU")
    plt.plot(epochs, dice_coef, color="blue", label="train dice coef")
    plt.plot(epochs, val_dice_coef, color="skyblue", label="val dice coef")
    plt.plot(epochs, loss_bacc, color="green", label="train binary acc")
    plt.plot(epochs, val_bacc, color="lightgreen", label="val binary acc")
    plt.title("Training and validation")
    plt.legend()
    plt.show()
    plt.savefig("training_plot.png")
 





