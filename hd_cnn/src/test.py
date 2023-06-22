import argparse
import cv2
import os
import numpy as np
import uuid

from .train import get_probabilistic_averaging_result
from .config import *


def get_fine_class_name(class_idx):
    with open(os.path.join(DATA_DIR, "fine_classes.txt"), "r") as f:
        idx = 0

        for line in f.readlines():
            if str(idx) != str(class_idx):
                idx += 1
                continue

            class_name = line.strip()
            fine_class_name = class_name.split("_")[1]
            return fine_class_name


def run_test(imgs):
    x = []
    for img in imgs:
        resized_img = cv2.resize(img, (HD_CNN_IMG_WIDTH, HD_CNN_IMG_HEIGHT))
        x.append(resized_img)
    x = np.array(x)

    proba = get_probabilistic_averaging_result(x)

    labels = []
    for i in range(len(proba)):
        predicted = np.argmax(proba[i])
        label = get_fine_class_name(predicted)

        labels.append(label)

    return labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, required=True)
    args = parser.parse_args()

    filename = args.filename
    img = cv2.imread(os.path.join(WORKING_DIR, "test_img", filename), 0)
    img = cv2.resize(img, (HD_CNN_IMG_WIDTH, HD_CNN_IMG_HEIGHT))

    tmp_img_filename = uuid.uuid4().hex + ".jpg"
    tmp_img_path = os.path.join(WORKING_DIR, "test_img", tmp_img_filename)
    cv2.imwrite(tmp_img_path, img)

    img = cv2.imread(tmp_img_path)

    prediction_result = run_test([img])
    print(prediction_result[0])

    os.remove(tmp_img_path)
