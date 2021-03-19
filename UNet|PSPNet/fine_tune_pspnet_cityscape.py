#!/usr/bin/env python
# coding: utf-8
# -*- coding: utf-8 -*-


from keras_segmentation.models.model_utils import transfer_weights
from keras_segmentation.pretrained import  resnet_pspnet_VOC12_v0_1, pspnet_101_cityscapes
from keras_segmentation.models.pspnet import resnet50_pspnet, pspnet_101
import pdb
from tensorflow.keras.optimizers import Adam


#### local path
epoch=20
checkpoints_path='/home/GDDC-CV1/Desktop/CV-Semantic-Segmentation/model_transfer_learning/checkpoint/pspnet_weight'
batch_size=4
validate=True
train_images='/home/GDDC-CV1/Desktop/data_1024/train_x_png/'
train_annotations='/home/GDDC-CV1/Desktop/data_1024/train_y_png/'
input_height=None
input_width=None
n_classes=2
verify_dataset=True
val_images='/home/GDDC-CV1/Desktop/data_1024/val_x_png'
val_annotations='/home/GDDC-CV1/Desktop/data_1024/val_y_png'
val_batch_size=batch_size
auto_resume_checkpoint=False
load_weights=None
steps_per_epoch=512
val_steps_per_epoch=512
gen_use_multiprocessing=False
ignore_zero_class=False
lr=0.001
optimizer_name=Adam(lr=lr, decay=1e-6)
do_augment=False
augmentation_name="aug_all"
pred_dir='/home/GDDC-CV1/Desktop/data_1024/pred_x_presentation/'
out_dir='/home/GDDC-CV1/Desktop/pred_out/'
patience=10
model_save_path+'/home/GDDC-CV1/Desktop/CV-Semantic-Segmentation/model_transfer_learning/checkpoint/model/'
#### physical GPU (device: 0, name: Tesla V100-PCIE-16GB, pci bus id: f114:00:00.0, compute capability: 7.0) ####
pretrained_model = pspnet_101_cityscapes()

new_model = pspnet_101(n_classes=n_classes)
transfer_weights( pretrained_model , new_model  ) 
new_model.train(
    train_images =  train_images,
    train_annotations = train_annotations,
    checkpoints_path = checkpoints_path,
    epochs=epoch,
    batch_size=batch_size,
    validate=validate, 
    val_images=val_images,
    val_annotations=val_annotations,
    val_batch_size=batch_size,
    auto_resume_checkpoint=auto_resume_checkpoint,
    optimizer_name = optimizer_name,
    patience = patience
)

new_model.save(model_save_path+'model'+str(lr)+str(epoch))

out = new_model.predict_multiple(
    inp_dir=pred_dir,
    out_dir=out_dir)














