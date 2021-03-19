# Image Segmentation Keras : UNet, PSPNet in House Detection   

## Research Paper  
**U-Net: https://arxiv.org/abs/1505.04597**  
**PSPNet:https://arxiv.org/abs/1612.01105**    
**ResNet:http://arxiv.org/abs/1512.03385, http://image-net.org/challenges/LSVRC/2015/**     
**VGGNet:https://arxiv.org/abs/1409.1556, https://github.com/ethereon/caffe-tensorflow**    

<div align="center">   
<img src="https://github.com/ccalvin97/CV-Semantic-Segmentation/blob/master/model_transfer_learning/graph/unet.png" />
</div>  
<div align="center">   
<img src="https://github.com/ccalvin97/CV-Semantic-Segmentation/blob/master/model_transfer_learning/graph/vggnet.png" />
</div>  

## Contribution   
kuancalvin2016@gmail.com

## Models

Following models are supported and tested:

| model_name       | Base Model        | Segmentation Model |
|------------------|-------------------|--------------------|
| resnet50_pspnet  | Resnet-50         | PSPNet             |
| vgg_unet         | VGG 16            | U-Net              |
| unet             | NA                | U-Net              |


## Getting Started

### Install
1. Instal keras==2.3.1
2. git clone https://github.com/ccalvin97/CV-Semantic-Segmentation 
3. Install dependencies: pip install -r requirements.txt  

## Environment in Azure   
Computer - Standard NC6s_v3   
OS - Ubuntu 18.04  
conda environment - azureml_py36_tensorflow  
CUDNN - 7.6.5  
CUDA - 10.1.243  


### Data preparation  
````bash  
Your directory tree should be look like this:  
$SEG_ROOT/data  
├── urbanisation  
│   ├── test  
│   ├── train  
│   └── val  
````  

### Preparing the data for training  

You need to make two folders

*  Images Folder - For all the training images
* Annotations Folder - For the corresponding ground truth segmentation images

The filenames of the annotation images should be same as the filenames of the RGB images.

The size of the annotation image for the corresponding RGB image should be same.

For each pixel in the RGB image, the class label of that pixel in the annotation image would be the value of the blue pixel.

Example code to generate annotation images :

```python
import cv2
import numpy as np

ann_img = np.zeros((30,30,3)).astype('uint8')
ann_img[ 3 , 4 ] = 1 # this would set the label of pixel 3,4 as 1

cv2.imwrite( "ann_1.png" ,ann_img )
```

Only use bmp or png format for the annotation images.


## Start  
python model_vgg+unet.py   


## Using the python module

You can import keras_segmentation in  your python script and use the API

```python
from keras_segmentation.models.unet import vgg_unet

model = vgg_unet(n_classes=51 ,  input_height=416, input_width=608  )

model.train(
    train_images =  "dataset1/images_prepped_train/",
    train_annotations = "dataset1/annotations_prepped_train/",
    checkpoints_path = "/tmp/vgg_unet_1" , epochs=5
)

out = model.predict_segmentation(
    inp="dataset1/images_prepped_test/0016E5_07965.png",
    out_fname="/tmp/out.png"
)

import matplotlib.pyplot as plt
plt.imshow(out)

# evaluating the model 
print(model.evaluate_segmentation( inp_images_dir="dataset1/images_prepped_test/"  , annotations_dir="dataset1/annotations_prepped_test/" ) )

```



### Training the Model

To train the model run the following command:

```shell
python -m keras_segmentation train \
 --checkpoints_path="path_to_checkpoints" \
 --train_images="dataset1/images_prepped_train/" \
 --train_annotations="dataset1/annotations_prepped_train/" \
 --val_images="dataset1/images_prepped_test/" \
 --val_annotations="dataset1/annotations_prepped_test/" \
 --n_classes=50 \
 --input_height=320 \
 --input_width=640 \
 --model_name="vgg_unet"
```

Choose model_name from the table above



### Getting the predictions

To get the predictions of a trained model

```shell
python -m keras_segmentation predict \
 --checkpoints_path="path_to_checkpoints" \
 --input_path="dataset1/images_prepped_test/" \
 --output_path="path_to_predictions"

```



### Model Evaluation 

To get the IoU scores 

```shell
python -m keras_segmentation evaluate_model \
 --checkpoints_path="path_to_checkpoints" \
 --images_path="dataset1/images_prepped_test/" \
 --segs_path="dataset1/annotations_prepped_test/"
```



## Fine-tuning from existing segmentation model

The following example shows how to fine-tune a model with 10 classes .

```python
from keras_segmentation.models.model_utils import transfer_weights
from keras_segmentation.pretrained import pspnet_50_ADE_20K
from keras_segmentation.models.pspnet import pspnet_50

pretrained_model = pspnet_50_ADE_20K()

new_model = pspnet_50( n_classes=51 )

transfer_weights( pretrained_model , new_model  ) # transfer weights from pre-trained model to your model

new_model.train(
    train_images =  "dataset1/images_prepped_train/",
    train_annotations = "dataset1/annotations_prepped_train/",
    checkpoints_path = "/tmp/vgg_unet_1" , epochs=5
)


```

### Improvement Plan  
| Detail  |
| :--:  |  
| Loss Function: Simple Cross Entropy  | 
| Loss Function: Weighted Cross Entropy  |  
| Loss Function: Dice Loss   |  
| Loss Function: Ohem Loss  | 
| Loss Function: Focal Loss   | 
| Data Augmentation flip & rotate | 
| Data Augmentation vague & other | 
| Metrics Tn, Tp rate mIou        | 
| Metrics Dice loss & Acc         | 
| Multi-scale validation| 
| Data normalisation |  
| Data Cleaning   |    
| LR Adjustment  |    
| WD Adjustment |   
