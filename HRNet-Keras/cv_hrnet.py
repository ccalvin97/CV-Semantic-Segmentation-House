#!/usr/bin/env python
# coding: utf-8

# -*- coding: utf-8 -*-


from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import datetime
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, AveragePooling2D, Dropout, BatchNormalization, add, concatenate, Activation
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
    
    def true_positive_rate(self, y_true, y_pred):
        return K.sum(K.flatten(y_true)*K.flatten(K.round(y_pred)))/K.sum(y_true)
    
    def binary_accuracy(self, y_true, y_pred):
        return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)


##############################################################################
    def conv3x3(self, x, out_filters, strides=(1, 1)):
        x = Conv2D(out_filters, 3, padding='same', strides=strides, use_bias=False, kernel_initializer='he_normal')(x)
        return x
    
    
    def basic_Block(self, input, out_filters, strides=(1, 1), with_conv_shortcut=False):
        x = self.conv3x3(input, out_filters, strides)
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)
    
        x = self.conv3x3(x, out_filters)
        x = BatchNormalization(axis=3)(x)
    
        if with_conv_shortcut:
            residual = Conv2D(out_filters, 1, strides=strides, use_bias=False, kernel_initializer='he_normal')(input)
            residual = BatchNormalization(axis=3)(residual)
            x = add([x, residual])
        else:
            x = add([x, input])
    
        x = Activation('relu')(x)
        return x
    
    
    def bottleneck_Block(self, input, out_filters, strides=(1, 1), with_conv_shortcut=False):
        expansion = 4
        de_filters = int(out_filters / expansion)
    
        x = Conv2D(de_filters, 1, use_bias=False, kernel_initializer='he_normal')(input)
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)
    
        x = Conv2D(de_filters, 3, strides=strides, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)
    
        x = Conv2D(out_filters, 1, use_bias=False, kernel_initializer='he_normal')(x)
        x = BatchNormalization(axis=3)(x)
    
        if with_conv_shortcut:
            residual = Conv2D(out_filters, 1, strides=strides, use_bias=False, kernel_initializer='he_normal')(input)
            residual = BatchNormalization(axis=3)(residual)
            x = add([x, residual])
        else:
            x = add([x, input])
    
        x = Activation('relu')(x)
        return x
    
    
    def stem_net(self, input):
        x = Conv2D(64, 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(input)
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)
    
        # x = Conv2D(64, 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
        # x = BatchNormalization(axis=3)(x)
        # x = Activation('relu')(x)
    
        x = self.bottleneck_Block(x, 256, with_conv_shortcut=True)
        x = self.bottleneck_Block(x, 256, with_conv_shortcut=False)
        x = self.bottleneck_Block(x, 256, with_conv_shortcut=False)
        x = self.bottleneck_Block(x, 256, with_conv_shortcut=False)
    
        return x
    
    
    def transition_layer1(self, x, out_filters_list=[32, 64]):
        x0 = Conv2D(out_filters_list[0], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
        x0 = BatchNormalization(axis=3)(x0)
        x0 = Activation('relu')(x0)
    
        x1 = Conv2D(out_filters_list[1], 3, strides=(2, 2),
                    padding='same', use_bias=False, kernel_initializer='he_normal')(x)
        x1 = BatchNormalization(axis=3)(x1)
        x1 = Activation('relu')(x1)
    
        return [x0, x1]
    
    
    def make_branch1_0(self, x, out_filters=32):
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        return x
    
    
    def make_branch1_1(self, x, out_filters=64):
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        return x
    
    
    def fuse_layer1(self, x):
        x0_0 = x[0]
        x0_1 = Conv2D(32, 1, use_bias=False, kernel_initializer='he_normal')(x[1])
        x0_1 = BatchNormalization(axis=3)(x0_1)
        x0_1 = UpSampling2D(size=(2, 2))(x0_1)
        x0 = add([x0_0, x0_1])
    
        x1_0 = Conv2D(64, 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x[0])
        x1_0 = BatchNormalization(axis=3)(x1_0)
        x1_1 = x[1]
        x1 = add([x1_0, x1_1])
        return [x0, x1]
    
    
    def transition_layer2(self, x, out_filters_list=[32, 64, 128]):
        x0 = Conv2D(out_filters_list[0], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x[0])
        x0 = BatchNormalization(axis=3)(x0)
        x0 = Activation('relu')(x0)
    
        x1 = Conv2D(out_filters_list[1], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x[1])
        x1 = BatchNormalization(axis=3)(x1)
        x1 = Activation('relu')(x1)
    
        x2 = Conv2D(out_filters_list[2], 3, strides=(2, 2),
                    padding='same', use_bias=False, kernel_initializer='he_normal')(x[1])
        x2 = BatchNormalization(axis=3)(x2)
        x2 = Activation('relu')(x2)
    
        return [x0, x1, x2]
    
    
    def make_branch2_0(self, x, out_filters=32):
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        return x
    
    
    def make_branch2_1(self, x, out_filters=64):
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        return x
    
    
    def make_branch2_2(self, x, out_filters=128):
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        return x
    
    
    def fuse_layer2(self, x):
        x0_0 = x[0]
        x0_1 = Conv2D(32, 1, use_bias=False, kernel_initializer='he_normal')(x[1])
        x0_1 = BatchNormalization(axis=3)(x0_1)
        x0_1 = UpSampling2D(size=(2, 2))(x0_1)
        x0_2 = Conv2D(32, 1, use_bias=False, kernel_initializer='he_normal')(x[2])
        x0_2 = BatchNormalization(axis=3)(x0_2)
        x0_2 = UpSampling2D(size=(4, 4))(x0_2)
        x0 = add([x0_0, x0_1, x0_2])
    
        x1_0 = Conv2D(64, 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x[0])
        x1_0 = BatchNormalization(axis=3)(x1_0)
        x1_1 = x[1]
        x1_2 = Conv2D(64, 1, use_bias=False, kernel_initializer='he_normal')(x[2])
        x1_2 = BatchNormalization(axis=3)(x1_2)
        x1_2 = UpSampling2D(size=(2, 2))(x1_2)
        x1 = add([x1_0, x1_1, x1_2])
    
        x2_0 = Conv2D(32, 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x[0])
        x2_0 = BatchNormalization(axis=3)(x2_0)
        x2_0 = Activation('relu')(x2_0)
        x2_0 = Conv2D(128, 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x2_0)
        x2_0 = BatchNormalization(axis=3)(x2_0)
        x2_1 = Conv2D(128, 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x[1])
        x2_1 = BatchNormalization(axis=3)(x2_1)
        x2_2 = x[2]
        x2 = add([x2_0, x2_1, x2_2])
        return [x0, x1, x2]
    
    
    def transition_layer3(self, x, out_filters_list=[32, 64, 128, 256]):
        x0 = Conv2D(out_filters_list[0], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x[0])
        x0 = BatchNormalization(axis=3)(x0)
        x0 = Activation('relu')(x0)
    
        x1 = Conv2D(out_filters_list[1], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x[1])
        x1 = BatchNormalization(axis=3)(x1)
        x1 = Activation('relu')(x1)
    
        x2 = Conv2D(out_filters_list[2], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x[2])
        x2 = BatchNormalization(axis=3)(x2)
        x2 = Activation('relu')(x2)
    
        x3 = Conv2D(out_filters_list[3], 3, strides=(2, 2),
                    padding='same', use_bias=False, kernel_initializer='he_normal')(x[2])
        x3 = BatchNormalization(axis=3)(x3)
        x3 = Activation('relu')(x3)
    
        return [x0, x1, x2, x3]
    
    
    def make_branch3_0(self, x, out_filters=32):
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        return x
    
    
    def make_branch3_1(self, x, out_filters=64):
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        return x
    
    
    def make_branch3_2(self, x, out_filters=128):
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        return x
    
    
    def make_branch3_3(self, x, out_filters=256):
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        x = self.basic_Block(x, out_filters, with_conv_shortcut=False)
        return x
    
    
    def fuse_layer3(self, x):
        x0_0 = x[0]
        x0_1 = Conv2D(32, 1, use_bias=False, kernel_initializer='he_normal')(x[1])
        x0_1 = BatchNormalization(axis=3)(x0_1)
        x0_1 = UpSampling2D(size=(2, 2))(x0_1)
        x0_2 = Conv2D(32, 1, use_bias=False, kernel_initializer='he_normal')(x[2])
        x0_2 = BatchNormalization(axis=3)(x0_2)
        x0_2 = UpSampling2D(size=(4, 4))(x0_2)
        x0_3 = Conv2D(32, 1, use_bias=False, kernel_initializer='he_normal')(x[3])
        x0_3 = BatchNormalization(axis=3)(x0_3)
        x0_3 = UpSampling2D(size=(8, 8))(x0_3)
        x0 = concatenate([x0_0, x0_1, x0_2, x0_3], axis=-1)
        return x0
    
    
    def final_layer(self, x):
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(1, 1, use_bias=False, kernel_initializer='he_normal')(x)
        x = BatchNormalization(axis=3)(x)
        x = Activation('sigmoid', name='Classification')(x)
        return x
    
    
    def hrnet(self, pretrained_weights = None):
        inputs = Input((self.PIXEL, self.PIXEL, 3))
    
        x = self.stem_net(inputs)
    
        x = self.transition_layer1(x)
        x0 = self.make_branch1_0(x[0])
        x1 = self.make_branch1_1(x[1])
        x = self.fuse_layer1([x0, x1])
    
        x = self.transition_layer2(x)
        x0 = self.make_branch2_0(x[0])
        x1 = self.make_branch2_1(x[1])
        x2 = self.make_branch2_2(x[2])
        x = self.fuse_layer2([x0, x1, x2])
    
        x = self.transition_layer3(x)
        x0 = self.make_branch3_0(x[0])
        x1 = self.make_branch3_1(x[1])
        x2 = self.make_branch3_2(x[2])
        x3 = self.make_branch3_3(x[3])
        x = self.fuse_layer3([x0, x1, x2, x3])
    
        out = self.final_layer(x)
    
        model = Model(inputs=inputs, outputs=out)


        if pretrained_weights:
            model.load_weights(pretrained_weights)

        print(model.summary())
     
        model.compile(optimizer=Adam(lr=self.lr, decay=1e-6), loss=self.dice_coef_loss, metrics=[self.dice_coef, self.binary_accuracy, MeanIoU(num_classes=2)])
       # model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
        return model
##########################################################################


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
    MeanIoU = history.history["mean_io_u"]
    val_MeanIoU = history.history["val_mean_io_u"]
    dice_coef = history.history['dice_coef']
    val_dice_coef = history.history["val_dice_coef"]
    loss_bacc = history.history["binary_accuracy"]
    val_bacc = history.history["val_binary_accuracy"]

    epochs = range(1, len(MeanIoU) + 1)
    plt.style.use('ggplot')

    plt.figure()
    plt.plot(epochs, MeanIoU, color="red", label="train MeanIoU")
    plt.plot(epochs, val_MeanIoU, color="darkred", label="val MeanIoU")
    plt.plot(epochs, dice_coef, color="blue", label="train dice coef")
    plt.plot(epochs, val_dice_coef, color="skyblue", label="val dice coef")
    plt.plot(epochs, loss_bacc, color="green", label="train binary acc")
    plt.plot(epochs, val_bacc, color="lightgreen", label="val binary acc")
    plt.title("Training and validation ")
    plt.legend()
    plt.show()
    plt.savefig("training_plot.png")
    





