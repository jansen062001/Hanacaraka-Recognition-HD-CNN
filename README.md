# Hanacaraka Recognition HD-CNN

Image recognition for Javanese script using YOLOv4 Darknet and HD-CNN. This project uses YOLOv4 as the object detector, and each detected object will be classified by HD-CNN.

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
  - [Run YOLOv4 + HD-CNN](#run-yolov4--hd-cnn)
- [Acknowledgements](#acknowledgements)

# Usage

## Installation

```bash
git clone https://github.com/jansen062001/Hanacaraka-Recognition-HD-CNN.git
```

## Training and Testing YOLOv4 Darknet

1. Preparing The Dataset
   - Download and unzip the dataset with YOLO Darknet format from roboflow: [https://universe.roboflow.com/thesis-dicgg/hanacaraka-recognition](https://universe.roboflow.com/thesis-dicgg/hanacaraka-recognition)
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
   - Because our YOLOv4 model will use 416x416 (WxH), so each image in the dataset needs to be sliced into 416x416. Use this Github [repo](https://github.com/slanj/yolo-tiling) to do this work.
   - Copy all sliced images (train and test) and classes.names into `./dataset/yolov4_darknet/`
   - Rename `classes.names` to `classes.txt`
   - Run these commands to re-label and move the dataset into `./yolov4_darknet/dataset/`
      ```bash
      python generate_yolo_dataset.py
      ```
2. Training
    ```bash
    cd ./yolov4_darknet
    python -m src.train
    ```

    .weights file from the training process will be placed on `./yolov4_darknet/training/`
3. Testing
   - Put the image file in the directory `./yolov4_darknet/test_images/`
   - Run these commands
      ```bash
      cd ./yolov4_darknet
      python -m src.test --width=416 --height=416 --filename=example.jpg
      ```

## Training and Testing HD-CNN

1. Preparing The Dataset
   - Download and unzip the dataset with YOLO Darknet format from roboflow: [https://universe.roboflow.com/thesis-dicgg/hanacaraka-recognition](https://universe.roboflow.com/thesis-dicgg/hanacaraka-recognition)
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
   - Copy all files inside the `train` folder into `./dataset/hd_cnn/`
   - Rename `_darknet.labels` to `classes.txt`
   - Run these commands to re-label and move the dataset into `./hd_cnn/dataset/`
      ```bash
      python generate_hdcnn_dataset.py
      ```
2. Training
    ```bash
    cd ./hd_cnn
    python -m src.train --train=single_layer
    python -m src.train --train=coarse_layer
    python -m src.train --train=fine_layer
    ```

    .weights file from the training process will be placed on `./hd_cnn/weights/`
3. Testing
   - Put the image file in the directory `./hd_cnn/test_img/`
   - Run these commands
      ```bash
      cd ./hd_cnn
      python -m src.test --filename=example.jpg
      ```

## Run YOLOv4 + HD-CNN

- Put the image file in the directory `./`
- Run these commands
   ```bash
   cd ./
   python main.py --filename=example.jpg
   ```
- After the process complete, there is a file called `result.jpg` in the directory `./` as the final output

# Acknowledgements

<details>
  <summary><b>Expand</b></summary>
  
  - [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
  - [https://github.com/slanj/yolo-tiling](https://github.com/slanj/yolo-tiling)
  - [https://github.com/justinessert/hierarchical-deep-cnn](https://github.com/justinessert/hierarchical-deep-cnn)
  - [https://github.com/satyatumati/Hierarchical-Deep-CNN](https://github.com/satyatumati/Hierarchical-Deep-CNN)
</details>
