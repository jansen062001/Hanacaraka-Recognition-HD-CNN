# Hanacaraka Recognition HD-CNN

Image recognition for Javanese script using YOLOv4 Darknet and HD-CNN. This project use YOLOv4 as object detector, and each detected object will classified by HD-CNN.

# Environment

- CentOS Stream release 9
- CUDA Toolkit 11.8
- cuDNN 8.6
- tensorflow 2.12.0
- opencv 4.6.0

# Table of Contents

- [Hanacaraka Recognition HD-CNN](#hanacaraka-recognition-hd-cnn)
- [Environment](#environment)
- [Table of Contents](#table-of-contents)
- [Usage](#usage)
  - [Installation](#installation)
  - [Training and Testing YOLOv4 Darknet](#training-and-testing-yolov4-darknet)
  - [Training and Testing HD-CNN](#training-and-testing-hd-cnn)
  - [Entire System](#entire-system)

# Usage

## Installation

```bash
git clone https://github.com/jansen062001/Hanacaraka-Recognition-HD-CNN.git
```

## Training and Testing YOLOv4 Darknet

1. Preparing Dataset
   - Download and unzip dataset with YOLO Darknet format from roboflow: [https://universe.roboflow.com/thesis-dicgg/hanacaraka-recognition](https://universe.roboflow.com/thesis-dicgg/hanacaraka-recognition)
      ```bash
      Hanacaraka YOLOv4 Darknet.v14i.darknet
      │   README.dataset.txt
      │   README.roboflow.txt
      │
      └───train
            ...
            9_png.rf.f7a6d330b72103e36cc779f7a2c5d075.jpg
            9_png.rf.f7a6d330b72103e36cc779f7a2c5d075.txt
            _darknet.labels
      ```
   - Because our YOLOv4 model will use 416 width and height, so each image in dataset need to be sliced into 416x416. Use this github [repo](https://github.com/slanj/yolo-tiling) to do this work.
   - Copy all sliced image (train and test) and classes.names into `./dataset/yolov4_darknet/`
   - Rename `classes.names` to `classes.txt`
   - Run these command to re-label and move the dataset into `./yolov4_darknet/dataset/`
      ```bash
      python generate_yolo_dataset.py
      ```
2. Training
    ```bash
    cd ./yolov4_darknet
    python -m src.train
    ```

    Weights file from training process is placed on `./yolov4_darknet/training/`
3. Testing
   - Put image file in the directory `./yolov4_darknet/test_images/`
   - Run these commands
      ```bash
      cd ./yolov4_darknet
      python -m src.test --width=416 --height=416 --filename=example.jpg
      ```

## Training and Testing HD-CNN

## Entire System