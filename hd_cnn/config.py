import os

HD_CNN_IMG_WIDTH = 32
HD_CNN_IMG_HEIGHT = 32
HD_CNN_IMG_CHANNEL = 3  # grayscale

BATCH_SIZE = 64

COARSE_CLASS_NUM = 24
FINE_CLASS_NUM = 727

MIN_DATA_EACH_CLASS = 10
MAX_DATA_EACH_CLASS = None

HD_CNN_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(HD_CNN_DIR, "dataset")
RAW_DATASET_DIR = os.path.join(DATASET_DIR, "raw")
PROCESSED_DATASET_DIR = os.path.join(DATASET_DIR, "processed")
TRAIN_DATASET_DIR = os.path.join(PROCESSED_DATASET_DIR, "train")
VALID_DATASET_DIR = os.path.join(PROCESSED_DATASET_DIR, "valid")
TEST_DATASET_DIR = os.path.join(PROCESSED_DATASET_DIR, "test")
LOG_DIR = os.path.join(HD_CNN_DIR, "log")
WEIGHTS_DIR = os.path.join(HD_CNN_DIR, "weights")

IMG_DATASET_EXT = ["jpg", "png", "jpeg"]

RAW_CLASSES_TXT_PATH = os.path.join(RAW_DATASET_DIR, "classes.txt")
COARSE_CLASSES_TXT_PATH = os.path.join(PROCESSED_DATASET_DIR, "coarse_classes.txt")
FINE_CLASSES_TXT_PATH = os.path.join(PROCESSED_DATASET_DIR, "fine_classes.txt")

SINGLE_CLASSIFIER_MODEL_LEARNING_RATE = 0.001
SINGLE_CLASSIFIER_MODEL_WEIGHTS_PATH = os.path.join(
    WEIGHTS_DIR, "single_classifier_model", "single_classifier_model.ckpt"
)

COARSE_CLASSIFIER_MODEL_LEARNING_RATE = 0.005
COARSE_CLASSIFIER_MODEL_WEIGHTS_PATH = os.path.join(
    WEIGHTS_DIR, "coarse_classifier_model", "coarse_classifier_model.ckpt"
)

FINE_CLASSIFIER_MODEL_LEARNING_RATE = 0.01
FINE_CLASSIFIER_MODEL_WEIGHTS_PATH = os.path.join(
    WEIGHTS_DIR, "fine_classifier_model_{num}", "fine_classifier_model_{num}.ckpt"
)

LOG_TIME_FORMAT = "%Y-%m-%d-%H-%M-%S"
