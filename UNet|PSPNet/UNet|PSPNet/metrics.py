  
#!/usr/bin/env python
# coding: utf-8

# -*- coding: utf-8 -*

import numpy as np
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.metrics import MeanIoU, Precision, Recall
import tensorflow.keras.backend as K

EPS = 1e-12


def get_iou(gt, pr, n_classes):
    class_wise = np.zeros(n_classes)
    for cl in range(n_classes):
        intersection = np.sum((gt == cl)*(pr == cl))
        union = np.sum(np.maximum((gt == cl), (pr == cl)))
        iou = float(intersection)/(union + EPS)
        class_wise[cl] = iou
    return class_wise


smooth=1
def dice_coef( y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_true_f = K.cast(y_true_f, dtype='float32')
    intersection = K.sum(y_true_f * y_pred_f)
    res=(2. * intersection + smooth) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f) + smooth)
    return res

def dice_coef_loss( y_true, y_pred):
    return 1-self.dice_coef(y_true, y_pred)

def dice_p_bce( in_gt, in_pred):
    return 0.05*binary_crossentropy(in_gt, in_pred) - self.dice_coef(in_gt, in_pred)


def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(K.epsilon()+pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
    return focal_loss_fixed

def binary_accuracy( y_true, y_pred):
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)

def tp_rate(y_true, y_pred):
    return K.sum(K.flatten(y_true)*K.flatten(K.round(y_pred)))/K.sum(y_true)

def tn_rate(y_true, y_pred):
    """
    param:
    y_pred - Predicted labels
    y_true - True labels 
    Returns:
    Specificity score
    """
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    fp = K.sum(neg_y_true * y_pred)
    tn = K.sum(neg_y_true * neg_y_pred)
    specificity = tn / (tn + fp + K.epsilon())
    return specificity


