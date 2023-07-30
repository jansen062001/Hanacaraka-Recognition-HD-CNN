import os

YOLO_IMG_SIZE = 416
YOLO_BATCH = 64
YOLO_SUBDIVISIONS = 4
YOLO_NUMBER_OF_CLASSES = 24
YOLO_ARCH = "-gencode arch=compute_61,code=[sm_61,compute_61]"
YOLO_PRETRAINED_WEIGHTS_URL = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29"
YOLO_PRETRAINED_WEIGHTS_NAME = "yolov4-tiny.conv.29"
YOLO_CFG_FILENAME = "yolov4-tiny.cfg"

YOLO_DIR = os.path.dirname(os.path.abspath(__file__))

YOLO_DATASET_DIR = os.path.join(YOLO_DIR, "dataset")
RAW_YOLO_DATASET_DIR = os.path.join(YOLO_DATASET_DIR, "raw")
PROCESSED_YOLO_DATASET_DIR = os.path.join(YOLO_DATASET_DIR, "processed")
TRAIN_YOLO_DATASET_DIR = os.path.join(PROCESSED_YOLO_DATASET_DIR, "train")
VALID_YOLO_DATASET_DIR = os.path.join(PROCESSED_YOLO_DATASET_DIR, "valid")
TEST_YOLO_DATASET_DIR = os.path.join(PROCESSED_YOLO_DATASET_DIR, "test")
YOLO_IMG_DATASET_EXT = ["jpg", "png", "jpeg"]

RAW_YOLO_CLASSES_TXT = os.path.join(RAW_YOLO_DATASET_DIR, "classes.txt")
DONE_YOLO_CLASSES_TXT = os.path.join(PROCESSED_YOLO_DATASET_DIR, "classes.txt")

WEIGHTS_FOLDER_NAME = "weights"
WEIGHTS_DIR = os.path.join(YOLO_DIR, WEIGHTS_FOLDER_NAME)
WEIGHTS_RESULT_FILENAME = "yolov4-tiny_final.weights"

DARKNET_DIR = os.path.join(YOLO_DIR, "darknet")
DARKNET_MAKEFILE_PATH = os.path.join(DARKNET_DIR, "Makefile")
DARKNET_DATASET_DIR = os.path.join(DARKNET_DIR, "data", "obj")
DARKNET_OBJ_DATA_PATH = os.path.join(DARKNET_DIR, "data", "obj.data")
DARKNET_OBJ_NAMES_PATH = os.path.join(DARKNET_DIR, "data", "obj.names")
