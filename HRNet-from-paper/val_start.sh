#!/bin/bash

## Start Validation for HRNet ##
InputStra="/home/GDDC-CV1/Desktop/data_1024_hrnet/data/urbanisation/x/test/"
out_addr="/home/GDDC-CV1/Desktop/data_1024_hrnet/data/list/urbanisation/test.lst"

python -W ignore /home/GDDC-CV1/Desktop/CV-Semantic-Segmentation/HRNet-Semantic-Segmentation/tools/pred_lst_transform.py \
    -path $InputStra -path_out $out_addr
if [ "$?" == "0" ]
then
    python tools/test.py \
--cfg /home/GDDC-CV1/Desktop/CV-Semantic-Segmentation/HRNet-Semantic-Segmentation/experiments/\
seg_hrnet_w18_small_v1_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml \
DATASET.TEST_SET list/urbanisation/test.lst \
TEST.MODEL_FILE /home/GDDC-CV1/Desktop/CV-Semantic-Segmentation/HRNet-Semantic-Segmentation/output/\
urbanisation/ok_production_level/best.pth \
TEST.SCALE_LIST [1] \
TEST.FLIP_TEST False
    echo "pass"
else
    echo "pred_lst_transform.py error"
    exit 1
fi



: << !
For example, evaluating our model on the Cityscapes validation set with multi-scale and flip testing:

python tools/test.py --cfg experiments/cityscapes/seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml \
                     TEST.MODEL_FILE hrnet_w48_cityscapes_cls19_1024x2048_trainset.pth \
                     TEST.SCALE_LIST 0.5,0.75,1.0,1.25,1.5,1.75 \
                     TEST.FLIP_TEST True
Evaluating our model on the Cityscapes test set with multi-scale and flip testing:

python tools/test.py --cfg experiments/cityscapes/seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml \
                     DATASET.TEST_SET list/cityscapes/test.lst \
                     TEST.MODEL_FILE hrnet_w48_cityscapes_cls19_1024x2048_trainset.pth \
                     TEST.SCALE_LIST 0.5,0.75,1.0,1.25,1.5,1.75 \
                     TEST.FLIP_TEST True
!

