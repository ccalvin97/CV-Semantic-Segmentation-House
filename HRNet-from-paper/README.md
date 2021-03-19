# High-resolution networks - Semantic Segmentation in House Detection  

## Research Paper  
**HRNet: https://arxiv.org/abs/1904.04514**  

<div align="center">   
<img src="https://github.com/ccalvin97/CV-Semantic-Segmentation/blob/master/HRNet-Semantic-Segmentation/graph/HRNet.jpg" />
</div>  
<div align="center">   
<img src="https://github.com/ccalvin97/CV-Semantic-Segmentation/blob/master/HRNet-Semantic-Segmentation/graph/v2-4cfe795071d15b3d32249369534183c3_1440w.jpg" />
</div>  


## Contribution  
Edited by kuancalvin2016@gmail.com  
Original Code from HRNet Official Github  

## Introduction
This is the official code of [high-resolution representations for Semantic Segmentation](https://arxiv.org/abs/1904.04514). 
We augment the HRNet with a very simple segmentation head shown in the figure below. We aggregate the output representations at four different resolutions, and then use a 1x1 convolutions to fuse these representations. The output representations is fed into the classifier. We evaluate our methods on three datasets, Cityscapes, PASCAL-Context and LIP.


## Models  
HRNetV2 Segmentation models are now available. All the results are reproduced by using this repo!!!

The models are initialized by the weights pretrained on the ImageNet. You can download the pretrained models from  https://github.com/HRNet/HRNet-Image-Classification.

### Small models

The models are initialized by the weights pretrained on the ImageNet. You can download the pretrained models from  https://github.com/HRNet/HRNet-Image-Classification.

Performance on the Cityscapes dataset. The models are trained and tested with the input size of 512x1024 and 1024x2048 respectively. The results of other small models are obtained from Structured Knowledge Distillation for Semantic Segmentation(https://arxiv.org/abs/1903.04197). The small model are built based on the code of Pytorch-v1.1 branch.

| model | Train Set | Test Set |#Params | GFLOPs | OHEM | Multi-scale| Flip | Distillation | mIoU | Link |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| HRNetV2-W18-Small-v1 | Train | Val | 1.5M | 31.1 | No | No | No | No | 70.3 | [OneDrive](https://1drv.ms/u/s!Aus8VCZ_C_33gSEsg-2sxTmZL2AT?e=AqHbjh)/[BaiduYun(Access Code:63be)](https://pan.baidu.com/s/17pr-he0HEBycHtUdfqWr3g)|
| HRNetV2-W18-Small-v2 | Train | Val | 3.9M | 71.6 | No | No | No | No | 76.2 | [OneDrive](https://1drv.ms/u/s!Aus8VCZ_C_33gSAL4OurOW0RX4JH?e=ptLwpW)/[BaiduYun(Access Code:p1qf)](https://pan.baidu.com/s/1EHsZhqxWI0KF304Ptcj5-A)|

## Getting start
### Install
1. Instal torch==0.4.1.post2 
2. git clone https://github.com/HRNet/HRNet-Semantic-Segmentation $SEG_ROOT  
3. Install dependencies: pip install -r requirements.txt  

## Environment in Azure   
Computer - Standard NC6s_v3  
OS - Ubuntu 18.04  
conda environment - py37_pytorch  
CUDNN - 7.6.5  
CUDA - 10.1.243  


### Data preparation
Your directory tree should be look like this:
````bash
$SEG_ROOT/data
├── urbanisation
│   ├── y
│   │   ├── test
│   │   ├── train
│   │   └── val
│   └── x
│       ├── test
│       ├── train
│       └── val
├── list
│   ├── urbanisation
│   │   ├── test.lst
│   │   ├── trainval.lst
│   │   └── val.lst
````

### Train and test
Please specify the configuration file.

Train - bash start.sh  

For example, evaluating our model on the Cityscapes validation set with multi-scale and flip testing:
````bash
python tools/test.py --cfg experiments/cityscapes/seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml \
                     TEST.MODEL_FILE hrnet_w48_cityscapes_cls19_1024x2048_trainset.pth \
                     TEST.SCALE_LIST 0.5,0.75,1.0,1.25,1.5,1.75 \
                     TEST.FLIP_TEST True
````
Evaluating our model on the Cityscapes test set with multi-scale and flip testing:
````bash
python tools/test.py --cfg experiments/cityscapes/seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml \
                     DATASET.TEST_SET list/cityscapes/test.lst \
                     TEST.MODEL_FILE hrnet_w48_cityscapes_cls19_1024x2048_trainset.pth \
                     TEST.SCALE_LIST 0.5,0.75,1.0,1.25,1.5,1.75 \
                     TEST.FLIP_TEST True
````


### Improvement Plan  
| Detail  |  
| :--: | :--: |  
| Loss Function: Simple Cross Entropy   | 
| Loss Function: Weighted Cross Entropy |  
| Loss Function: Dice Loss  |  
| Loss Function: Ohem Loss   | 
| Data Augmentation flip & rotate  | 
| Data Augmentation vague & other   | 
| Metrics Tn, Tp rate mIou  | 
| Metrics Dice loss & Acc   | 
| Multi-scale validation  | 
| Data normalisation.   |    
| Data Cleaning   |    
| LR Adjustment   |   
| WD Adjustment   |   

## Citation
If you find this work or code is helpful in your research, please cite:
````
@inproceedings{SunXLW19,
  title={Deep High-Resolution Representation Learning for Human Pose Estimation},
  author={Ke Sun and Bin Xiao and Dong Liu and Jingdong Wang},
  booktitle={CVPR},
  year={2019}
}

@article{WangSCJDZLMTWLX19,
  title={Deep High-Resolution Representation Learning for Visual Recognition},
  author={Jingdong Wang and Ke Sun and Tianheng Cheng and 
          Borui Jiang and Chaorui Deng and Yang Zhao and Dong Liu and Yadong Mu and 
          Mingkui Tan and Xinggang Wang and Wenyu Liu and Bin Xiao},
  journal   = {TPAMI},
  year={2019}
}
````

## Reference
[1] Deep High-Resolution Representation Learning for Visual Recognition. Jingdong Wang, Ke Sun, Tianheng Cheng, 
    Borui Jiang, Chaorui Deng, Yang Zhao, Dong Liu, Yadong Mu, Mingkui Tan, Xinggang Wang, Wenyu Liu, Bin Xiao. Accepted by TPAMI. [download](https://arxiv.org/pdf/1908.07919.pdf)

## Acknowledgement
We adopt sync-bn implemented by [InplaceABN](https://github.com/mapillary/inplace_abn).

We adopt data precosessing on the PASCAL-Context dataset, implemented by [PASCAL API](https://github.com/zhanghang1989/detail-api).
