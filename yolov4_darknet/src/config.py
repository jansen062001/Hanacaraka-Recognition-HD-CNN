import os

YOLO_IMG_SIZE = 416
YOLO_PRETRAINED_WEIGHTS_URL = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29"
YOLO_PRETRAINED_WEIGHTS_NAME = "yolov4-tiny.conv.29"
YOLO_NUMBER_OF_CLASSES = 24
YOLO_PERCENTAGE_TRAIN = 90
YOLO_PERCENTAGE_TEST = 100 - YOLO_PERCENTAGE_TRAIN
YOLO_IMG_DATASET_EXT = ["jpg", "png", "jpeg"]
YOLO_CFG_FILENAME = "yolov4-tiny.cfg"
YOLO_ARCH = "-gencode arch=compute_61,code=[sm_61,compute_61]"

WORKING_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)
TRAIN_RESULT_DIRNAME = "training"
TRAIN_RESULT_DIR = os.path.join(WORKING_DIR, TRAIN_RESULT_DIRNAME)
DARKNET_DIR = os.path.join(WORKING_DIR, "darknet")
DARKNET_DATA_DIR = os.path.join(DARKNET_DIR, "data", "obj")
DATA_DIR = os.path.join(WORKING_DIR, "dataset")
