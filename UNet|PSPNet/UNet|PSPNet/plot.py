#!/usr/bin/env python
# coding: utf-8

# -*- coding: utf-8 -*-
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pdb



def plot(history):
    pdb.set_trace()
    print('Start Plot')
    tn_rate = history.history["tn_rate"]
    val_tn_rate = history.history["val_tn_rate"]
    dice_coef = history.history['dice_coef']
    val_dice_coef = history.history["val_dice_coef"]
    loss_bacc = history.history["accuracy"]
    val_bacc = history.history["val_accuracy"]
    tp = history.history["tp_rate"]
    val_tp= history.history["val_tp_rate"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = range(1, len(val_bacc) + 1)
    plt.style.use('ggplot')

    plt.figure()
    plt.plot(epochs, val_tn_rate, color="black", label="val tn_rate", alpha=0.7)
    plt.plot(epochs, val_dice_coef, color="blue", label="val dice coef", alpha=0.7)
    plt.plot(epochs, val_bacc, color="green", label="val binary acc", alpha=0.7)
    plt.plot(epochs, val_tp, color="yellow", label="val tp_rate", alpha=0.7)
    plt.plot(epochs, val_loss, color="red", label="val loss")
    plt.title("Validation")
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
    plt.savefig("val_loss.png")

    plt.figure()
    plt.plot(epochs, tn_rate, color="black", label="tn_rate", alpha=0.7)
    plt.plot(epochs, dice_coef, color="blue", label="train dice coef", alpha=0.7)
    plt.plot(epochs, loss_bacc, color="green", label="train binary acc", alpha=0.7)
    plt.plot(epochs, tp, color="yellow", label="tp_rate", alpha=0.7)
    plt.plot(epochs, loss, color="red", label="loss")
    plt.title("Training")
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
    plt.savefig("training_loss.png")

