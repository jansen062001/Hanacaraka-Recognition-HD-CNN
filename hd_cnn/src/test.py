import argparse
import cv2
import os
import numpy as np
import uuid

from train import get_probabilistic_averaging_result
from config import *


def run_test(arr_img):
    resized_img = cv2.resize(arr_img, (HD_CNN_IMG_WIDTH, HD_CNN_IMG_HEIGHT))
    x = []
    x.append(resized_img)
    x = np.array(x)

    proba = get_probabilistic_averaging_result(x)
    predicted = np.argmax(proba)

    return predicted


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

    prediction_result = run_test(img)
    print(prediction_result)

    os.remove(tmp_img_path)
