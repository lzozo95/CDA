# Camera Data Augmentation (CDA)

## Image-to-image translation form day image to night image

### Introduction

* This repository is released for Camera Data Augmentation (CDA).
* This work translates day image to night image.
* This works on image files (*.png) from the Cityscapes dataset.

### Requirements
Python 3.9.16\
Pytorch 1.12.0\
Numpy 1.23.5

### Demo

1. Make models directory with below codes
```
mkdir models
```
2. Download weights files from this [link](https://).

3. 

```
cd ./lidar_model_da
python main.py
```

## Image-to-Image translation from sunny image to adverse weather image

### Introduction

* This repository is released for Camera Data Augmentation (CDA).
* This work translates sunny image to adverse weather image.
* Adverse weathers in this work contain foggy, rainy, snowy weathers.
* This works on image files (*.png) from the Cityscapes dataset.

### Requirements
Python 3.9.16\
Pytorch 1.12.0\
Numpy 1.23.5

### Demo
```
python ./VRI/voxel_ray_interaction.py
```

## Recent Updates
* (2023.12.15) Image-to-image translation form day image to night image code update
* (2023.12.15) Image-to-Image translation from sunny image to adverse weather image code update

## Background

### Outline
This SW is related to 'data augmentation engine' in 'Development of Driving Environment Data Transformation and Data Verification Technology for the Mutual Utilization of Self-driving Learning Data for Different Vehicles', and in detail, it is SW for Camera data augmentation.
### Data augmentation engine
New data with changed weather, lighting, objects, etc. is generated(augmented) from the acquired real environment information or virtual information.
- Environmental data augmentation: Generating other types of data in which the environment, object elements, etc. is conducted from the collected data and the environmental information at the time of collection
- Main object data augmentation: Inserting main target objects (VRU, Vehicle) on the road into the collected data in using AI
![data_augmentation_image](https://user-images.githubusercontent.com/95835936/147022053-62dd1851-2717-41af-9233-3c5f344dc8cb.png)