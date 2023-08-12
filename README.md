# Unsupervised semantic segmentation using STEGO on liver ultrasound

This is an implementation for unsupervised semantic segmentation task using STEGO model applied on liver ultrasound dataset.
The original implementation for STEGO model can be found [here](https://github.com/mhamilton723/STEGO/tree/master).

## Contents
- [Overview of STEGO](#overview)
- [Install](#install)
- [Training](#training)
- [Results](#results)

## Overview
Overview of the STEGO model can be found here

## Install
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
Note: [convert_dataset.py](src/convert_dataset.py) can be used to convert images between **.png** and **.jpg** by adjusting commented variables.  
Note: If you do not have any labels, disregard **labels** directory from the structure.

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
Note: *crop_types* and *crop_ratios* variables can be modified based on the intended cropping design.

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

## Results
Results will be posted