import os

WORKING_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)
DATA_DIR = os.path.join(
    WORKING_DIR,
    "dataset",
)
COARSE_DIR = os.path.join(DATA_DIR, "coarse")
FINE_DIR = os.path.join(DATA_DIR, "fine")
COARSE_FINE_DIR = os.path.join(DATA_DIR, "coarse_fine")
LOG_DIR = os.path.join(WORKING_DIR, "log")
WEIGHTS_DIR = os.path.join(WORKING_DIR, "weights")

SINGLE_CLASSIFIER_MODEL_LEARNING_RATE = 0.001
SINGLE_CLASSIFIER_MODEL_WEIGHTS_PATH = os.path.join(
    WEIGHTS_DIR, "single_classifier_model", "cp.ckpt"
)

COARSE_CLASSIFIER_MODEL_LEARNING_RATE = 0.005
COARSE_CLASSIFIER_MODEL_WEIGHTS_PATH = os.path.join(
    WEIGHTS_DIR, "coarse_classifier_model", "cp.ckpt"
)

FINE_CLASSIFIER_MODEL_LEARNING_RATE = 0.01
FINE_CLASSIFIER_MODEL_WEIGHTS_PATH = os.path.join(
    WEIGHTS_DIR, "fine_classifier_model_{}", "cp.ckpt"
)

BATCH_SIZE = 64
VALIDATION_SPLIT = 0.2
