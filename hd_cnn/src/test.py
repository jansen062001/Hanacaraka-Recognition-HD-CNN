import argparse
import cv2
from config import *
import os
from train import get_probabilistic_averaging_result
import numpy as np
import uuid

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, required=True)
    args = parser.parse_args()

    filename = args.filename
    img = cv2.imread(os.path.join(WORKING_DIR, "test_img", filename), 0)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

    tmp_img_filename = uuid.uuid4().hex + ".jpg"
    tmp_img_path = os.path.join(WORKING_DIR, "test_img", tmp_img_filename)
    cv2.imwrite(tmp_img_path, img)

    img = cv2.imread(tmp_img_path)

    x = []
    x.append(img)
    x = np.array(x)

    proba = get_probabilistic_averaging_result(x)
    predicted = np.argmax(proba)

    print(predicted)

    os.remove(tmp_img_path)
