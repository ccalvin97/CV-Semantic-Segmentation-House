#!/bin/bash

#### local path
pathX=/home/GDDC-CV1/Desktop/data_1000/train_x
pathY=/home/GDDC-CV1/Desktop/data_1000/train_y
pathX_val=/home/GDDC-CV1/Desktop/data_1000/val_x 
pathY_val=/home/GDDC-CV1/Desktop/data_1000/val_y

#### physical GPU (device: 0, name: Tesla V100-PCIE-16GB, pci bus id: f114:00:00.0, compute capability: 7.0) ####

python -W ignore main_v9.py \
    -pathX ${pathX} \
    -pathY ${pathY} \
    -pathX_val ${pathX_val} \
    -pathY_val ${pathY_val} \
    -is_predict 0 \
    -load_weight no \
    $@

: << !
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
!






