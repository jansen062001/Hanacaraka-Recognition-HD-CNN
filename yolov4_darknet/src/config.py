import os

YOLO_WEIGHTS_URL = 'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137'
YOLO_WEIGHTS_NAME = 'yolov4.conv.137'
NUMBER_OF_CLASSES = 11
PERCENTAGE_TRAIN = 90
PERCENTAGE_TEST = 100 - PERCENTAGE_TRAIN
IMG_DATASET_EXT = ['jpg', 'png', 'jpeg']
YOLO_CFG_FILENAME = 'yolov4-custom.cfg'
ARCH = '-gencode arch=compute_61,code=[sm_61,compute_61]'

WORKING_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..'))
TRAIN_RESULT_DIRNAME = 'training'
TRAIN_RESULT_DIR = os.path.join(WORKING_DIR, TRAIN_RESULT_DIRNAME)
DARKNET_DIR = os.path.join(WORKING_DIR, 'darknet')
DARKNET_DATA_DIR = os.path.join(DARKNET_DIR, 'data', 'obj')
DATA_DIR = os.path.join(WORKING_DIR, 'dataset')
