# Unsupervised semantic segmentation using STEGO on liver ultrasound

This is an implementation for unsupervised semantic segmentation task using STEGO model applied on liver ultrasound dataset.
The original implementation for STEGO model can be found [here](https://github.com/mhamilton723/STEGO/tree/master).

## Contents
- [Overview of STEGO](#overview)
- [Setup](#setup)
- [Training](#training)
- [Results](#results)

## Overview
Overview of the STEGO model can be found here

## Setup
### Clone this repository:
> git clone https://github.com/tibetakinci/segment-liver
> cd src

### Setup conda environment:
Please visit the [Anaconda install page](https://docs.anaconda.com/anaconda/install/index.html) if you do not already have conda installed

> conda env create -f environment.yml
> conda activate stego

### Prepare dataset:
To train model please place dataset to pytorch data directory, variable **pytorch_data_dir** in [train_config.yml](/src/configs/train_config.yml) with the following structure.  
*Note:* If you do not have any labels, disregard **labels** directory from the structure:
*Note:* You can use [convert_dataset.py](/src/convert_dataset.py) script to convert between **.png** and **.jpg** by adjusting commented variables.

> dataset_name
> |── imgs
> |   ├── train
> |   |   |── unique_img_name_1.jpg
> |   |   └── unique_img_name_2.jpg
> |   └── val
> |       |── unique_img_name_3.jpg
> |       └── unique_img_name_4.jpg
> └── labels
>     ├── train
>     |   |── unique_img_name_1.png
>     |   └── unique_img_name_2.png
>     └── val
>         |── unique_img_name_3.png
>         └── unique_img_name_4.png

## Training:


## Results: