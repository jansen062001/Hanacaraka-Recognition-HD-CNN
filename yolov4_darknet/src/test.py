import argparse
import os
import shutil
import uuid
import cv2

from config import *


def create_cfg_test_file(cfg_filename):
    os.chdir(os.path.join(DARKNET_DIR, "cfg"))

    shutil.copyfile(YOLO_CFG_FILENAME, cfg_filename)

    cmd = (
        'sed -i "s/batch=64/batch=1/" '
        + cfg_filename
        + ";"
        + 'sed -i "s/subdivisions=64/subdivisions=1/" '
        + cfg_filename
        + ";"
        + 'sed -i "s/width=416/width='
        + args.width
        + '/" '
        + cfg_filename
        + ";"
        + 'sed -i "s/height=416/height='
        + args.height
        + '/" '
        + cfg_filename
        + ";"
    )
    os.system(cmd)


def run_test(cfg_filename, test_img_filename):
    os.chdir(os.path.join(WORKING_DIR, "test_images"))

    img = cv2.imread(test_img_filename, 0)
    filename = uuid.uuid4().hex + ".jpg"
    cv2.imwrite(filename, img)

    os.chdir(DARKNET_DIR)

    cmd = (
        "./darknet detector test data/obj.data cfg/"
        + cfg_filename
        + " ../"
        + TRAIN_RESULT_DIRNAME
        + "/yolov4-tiny_final.weights -ext_output -dont_show ../test_images/"
        + filename
        + " > ../test_images/result.txt"
    )
    os.system(cmd)

    cmd = "cp predictions.jpg ../test_images/output_" + test_img_filename
    os.system(cmd)

    os.chdir(os.path.join(WORKING_DIR, "test_images"))
    os.remove(filename)


def remove_cfg_test_file(cfg_filename):
    os.chdir(os.path.join(DARKNET_DIR, "cfg"))

    os.remove(cfg_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=str, required=True)
    parser.add_argument("--height", type=str, required=True)
    parser.add_argument("--filename", type=str, required=True)
    args = parser.parse_args()

    cfg_test_filename = uuid.uuid4().hex + ".cfg"

    create_cfg_test_file(cfg_test_filename)
    run_test(cfg_test_filename, args.filename)
    remove_cfg_test_file(cfg_test_filename)
