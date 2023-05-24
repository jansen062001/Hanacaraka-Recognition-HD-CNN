import os

WORKING_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)
TRAIN_RESULT_DIRNAME = "training"
TRAIN_RESULT_DIR = os.path.join(WORKING_DIR, TRAIN_RESULT_DIRNAME)
DARKNET_DIR = os.path.join(WORKING_DIR, "darknet")
DARKNET_DATA_DIR = os.path.join(DARKNET_DIR, "data", "obj")
DATA_DIR = os.path.join(WORKING_DIR, "dataset")
