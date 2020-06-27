# A Deep Multi-modal Explanation Model for Zero-shot Learning

This repo contains the Pytorch source code for the paper [A Deep Multi-modal Explanation Model for Zero-shot Learning](https://ieeexplore.ieee.org/document/9018377) in IEEE Transactions on Image Processing (TIP), 2020. 

In this project, we provide the data, implementation details and Grad-CAM visualization.


## Data

Download the data from [here](https://drive.google.com/drive/folders/1-nLNTRQybMde-NhyCz0IZRqYXmLUq6ii?usp=sharing).

Note that, we extract new visual features from ResNet-101 instead of using the features from previous works. 

For each image, we extract one visual feature without using any data augmentation like crop and flip, because the data augmentation 
will affect the correct alignment of visual explanations afterwards.


## Train and Test

- Run ```DME.py``` to train the visual-semantic embedding module.

- Run ```DME_joint.py``` to train the textual explanation module.

- Run ```.\Grad-CAM\gradcam_resnet101.py``` to generate the visual explanation.

## Notes

More instructions will be provided later.
