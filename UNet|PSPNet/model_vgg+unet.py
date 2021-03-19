# coding: utf-8
# -*- coding: utf-8 -*-

from tensorflow.keras.optimizers import Adam
from keras_segmentation.models.unet import vgg_unet
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from tensorflow.keras.models import save_model

#### local path
epoch=2
checkpoints_path='/home/GDDC-CV2/Desktop/CV-Semantic-Segmentation/model_transfer_learning/checkpoint/unet_weight'
batch_size=4
validate=True
train_images='/home/GDDC-CV2/Desktop/data_1024/train_x_png/'
train_annotations='/home/GDDC-CV2/Desktop/data_1024/train_y_png/'
input_height=1024
input_width=1024
n_classes=2
verify_dataset=True
val_images='/home/GDDC-CV2/Desktop/data_1024/val_x_png'
val_annotations='/home/GDDC-CV2/Desktop/data_1024/val_y_png'
val_batch_size=batch_size
auto_resume_checkpoint=False
load_weights=None
steps_per_epoch=512
val_steps_per_epoch=512
gen_use_multiprocessing=False
ignore_zero_class=False
lr=0.0001
optimizer_name=Adam(lr=lr, decay=1e-6)
do_augment=False
augmentation_name="aug_all"
pred_dir='/home/GDDC-CV2/Desktop/data_1024/pred_x_presentation/'
out_dir='/home/GDDC-CV2/Desktop/data_1024/pred_out/'
patience=10
model_save_path='/home/GDDC-CV2/Desktop/CV-Semantic-Segmentation/model_transfer_learning/checkpoint/whole_model/'

#### physical GPU (device: 0, name: Tesla V100-PCIE-16GB, pci bus id: f114:00:00.0, compute capability: 7.0) ####
model = vgg_unet(n_classes=n_classes ,  input_height=input_height, input_width=input_width)


model.train(
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
    patience=patience,
    steps_per_epoch=steps_per_epoch,
    val_steps_per_epoch=val_steps_per_epoch
    )

model.save(model_save_path+'model'+str(lr)+str(epoch))

out = model.predict_multiple(
    inp_dir=pred_dir,
    out_dir=out_dir)


