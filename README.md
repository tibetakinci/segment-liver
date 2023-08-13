# Unsupervised semantic segmentation using STEGO on liver ultrasound

This is an implementation for unsupervised semantic segmentation task using STEGO model applied on liver ultrasound dataset.
The original implementation for STEGO model can be found [here](https://github.com/mhamilton723/STEGO/tree/master).

## Contents
- [Setup](#setup)
- [Training](#training)
- [Overview of STEGO](#overview)
  - [Extracting feature correspondences](#extracting-feature-correspondences)
  - [Distilling feature correspondences](#distilling-feature-correspondences)
  - [The STEGO architecture](#the-stego-architecture)
- [Results](#results)

## Setup
### Clone this repository:
```
git clone https://github.com/tibetakinci/segment-liver  
cd src
```

### Setup conda environment:
Please visit the [Anaconda install page](https://docs.anaconda.com/anaconda/install/index.html) if you do not already have conda installed.
```
conda env create -f environment.yml  
conda activate stego
```

### Prepare dataset:
To train model please place dataset to pytorch data directory, variable **pytorch_data_dir** in [train_config.yml](src/configs/train_config.yml) with the following structure:    

```
dataset_name
|── imgs
|   ├── train
|   |   |── unique_img_name_1.jpg
|   |   └── unique_img_name_2.jpg
|   └── val
|       |── unique_img_name_3.jpg
|       └── unique_img_name_4.jpg
└── labels
    ├── train
    |   |── unique_img_name_1.png
    |   └── unique_img_name_2.png
    └── val
        |── unique_img_name_3.png
        └── unique_img_name_4.png 
```
> Note: [convert_dataset.py](src/convert_dataset.py) can be used to convert images between **.png** and **.jpg** by adjusting commented variables.  
> Note: If you do not have any labels, disregard **labels** directory from the structure.

### Update train_config.yml:
Adjust variables *pytorch_data_dir* and *dir_dataset_name* according to dataset directory and dataset name respectively.  
Update *dir_dataset_n_classes* variable for desired clustering class number if needed.  
Make sure that *crop_type* variable is set to **"five"**, **"random"** or **None** according to cropping used.

## Training
### Crop dataset:
If necessary, the cropping tool can be employed to enhance the spatial resolution:
```
python crop_datasets.py
```
> Note: *crop_types* and *crop_ratios* variables can be modified based on the intended cropping design.

### Precompute kNNs:
Before training the model, generating kNN indices for the dataset is required:
```
python precompute_knns.py
```

### Train model:
Finally, you can start training by typing:
```
python train_segmentation.py
```

Hyperparameters can be adjusted in [train_config.yml](src/configs/train_config.yml).

To monitor training with tensorboard run the following from root directory:
```
tensorboard --logdir logs
```

## Overview
### Introduction
Semantic segmentation is the task of classifying each pixel in an image into specific labels. Earlier studies primarily emphasized supervised learning through the assignment of ground truth labels, while more recent research suggests the adoption of unsupervised learning to mitigate labor-intensive challenges.
Numerous techniques have been developed to acquire semantically significant features. However in contrast to previous methods, STEGO make use of pre-trained features and focuses on distilling knowledge. This decision is motivated by the observed correlations between unsupervised features being semantically consistent, both within the same image and across images.
Proposed STEGO(**S**elf-supervised **T**ransformer with **E**nergy-based **G**raph **O**ptimization) is a novel transformer-based architecture which has the ability of segmenting objects within an image without supervision by distilling pre-trained unsupervised visual features into clusters using novel contrastive loss.
STEGO basically aims to refine features of pre-trained backbone to yield semantic segmentation predictions when clustered. Embeddings of DINO model is being used because of their quality.

### Extracting feature correspondences
Self-supervised contrastive learning aims to train models to differentiate between similar and dissimilar data points without supervision. This is achieved by computing feature correspondence tensor whose entries represent the cosine similarity between image features. 
The below figure demonstrates the correspondence of three different data points in between same image and it's K-nearest-neighbor with respect to DINO as the feature extractor.
![ultrasound feature correspondence](results/figures/correspondence.png)

### Distilling feature correspondences
In order to compose a high quality semantic segmentation, STEGO distills pre-trained feature correspondences to learn a low-dimensional pixel-wise embedding. This is achieved by utilizing an undirected graphical model, heavily inspired by CRF. 
Undirected graph model utilized to refine noisy or low-resolution class predictions by aligning them with edges and color-related regions in the input image. 
In distillation process, visual backbone is kept frozen and training a segmentation head is focused. The novel contrastive loss function of the STEGO encourages distilled features to form compact clusters.
**S**patial **C**enter operation and 0-Clamping is introduced to the loss function because of such challenges as being unstable sometimes and balancing the learning signal for small objects.
Together with SC and 0-Clamp, the final correlation loss is defined as:  
![contrastive loss](results/figures/loss.png)
![contrastive loss](results/figures/loss.svg)

### The STEGO architecture
First step of STEGO, frozen visual backbone is used as an input to segmentation head for predicting distilled features. Three distinct instantiations are utilized to train the segmentation head, self, KNN and random correspondences. 
STEGO's full loss is formulated as linear combination of weight(to control the balance of the learning signals) and contrastive loss function for each instantiation.  
Prediction pipeline of STEGO includes clustering and CRF as last two steps respectively. Due to feature distillation, STEGO's segmentation features form distinct clusters. Cosine-based minibatch K-Means employed to extract and assign classes from these clusters. Subsequently, the spatial resolution is enhanced through CRF refinement.
![the STEGO architecture](results/figures/stego.svg)

## Results
Results will be posted