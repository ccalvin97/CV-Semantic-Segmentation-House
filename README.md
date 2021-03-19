# CV-Semantic-Segmentation in Tf, Keras & Pytorch   
Computer Vision - Semantic Segmentation


| model                | Base Model    | Segmentation Model | Acc  | Mean IoU | Performance on Visualisation |   
| :--:                 | :--:          | :--:               | :--: | :--:     | :--:   |  
| HRNetV2-W18-Small-v1 | ImageNet      | HRNet              | 96.5%| 83%      | Good at small object and boundary pred   | 
| resnet50_pspnet      | Resnet-50     | PSPNet             | 91%  | 43%      | Bad, Pred Imbalanced    | 
| vgg_unet             | VGG 16        | U-Net              | 96%  | 75%      | Overall good    | 
| unet                 | NA            | U-Net              | 91%  | 25%      | Bad, Pred Imbalanced    | 



## Australia Prediction Example, Order Unet+VGGNet - Label - HRNet
<div align="center"><img src="https://github.com/ccalvin97/CV-Semantic-Segmentation/blob/master/Picture/austin16_20_.png" width="250"/><img src="https://github.com/ccalvin97/CV-Semantic-Segmentation/blob/master/Picture/austin16_20_1.png" width="250"/></center><img src="https://github.com/ccalvin97/CV-Semantic-Segmentation/blob/master/Picture/austin16_20__hrnet.png" width="250"/></center> 

<div align="center"><img src="https://github.com/ccalvin97/CV-Semantic-Segmentation/blob/master/Picture/test_215_.png" width="250"/><img src="https://github.com/ccalvin97/CV-Semantic-Segmentation/blob/master/Picture/test_215_1.png" width="250"/></center><img src="https://github.com/ccalvin97/CV-Semantic-Segmentation/blob/master/Picture/test_215__hrnet.png" width="250"/></center>   
