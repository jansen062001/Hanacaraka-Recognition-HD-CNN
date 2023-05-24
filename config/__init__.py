# HD-CNN
HD_CNN_IMG_WIDTH = 32
HD_CNN_IMG_HEIGHT = 32
HD_CNN_IMG_CHANNEL = 3  # grayscale
COARSE_CLASS_NUM = 24
FINE_CLASS_NUM = 1443

# YOLOv4
YOLO_IMG_SIZE = 416
YOLO_PRETRAINED_WEIGHTS_URL = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29"
YOLO_PRETRAINED_WEIGHTS_NAME = "yolov4-tiny.conv.29"
YOLO_NUMBER_OF_CLASSES = 24
YOLO_PERCENTAGE_TRAIN = 90
YOLO_PERCENTAGE_TEST = 100 - YOLO_PERCENTAGE_TRAIN
YOLO_IMG_DATASET_EXT = ["jpg", "png", "jpeg"]
YOLO_CFG_FILENAME = "yolov4-tiny.cfg"
YOLO_ARCH = "-gencode arch=compute_61,code=[sm_61,compute_61]"
