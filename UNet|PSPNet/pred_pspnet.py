# coding: utf-8
# -*- coding: utf-8 -*-
import os
################ CPU only ##############
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
########################################
from keras_segmentation.models.model_utils import transfer_weights
from keras_segmentation.pretrained import  resnet_pspnet_VOC12_v0_1, pspnet_101_cityscapes
from keras_segmentation.models.pspnet import resnet50_pspnet, pspnet_101

n_classes = 2
#model = vgg_unet(n_classes=n_classes ,  input_height=512, input_width=512)

pred_dir='/home/GDDC-CV1/Desktop/data_1024/pred_x_presentation/'
out_dir='/home/GDDC-CV1/Desktop/pred_out_pred-function'
checkpoints_path='/home/GDDC-CV1/Desktop/CV-Semantic-Segmentation/model_transfer_learning/checkpoint/'

pretrained_model = pspnet_101_cityscapes()
new_model = pspnet_101(n_classes=n_classes)
transfer_weights( pretrained_model , new_model  )
model = pspnet_101(n_classes=n_classes)


out = model.predict_multiple(
    inp_dir=pred_dir,
    out_dir=out_dir,
    checkpoints_path=checkpoints_path
    )


'''
model=None, inps=None, inp_dir=None, out_dir=None,
                     checkpoints_path=None, overlay_img=False,
                     class_names=None, show_legends=False, colors=class_colors,
                     prediction_width=None, prediction_height=None)
'''


