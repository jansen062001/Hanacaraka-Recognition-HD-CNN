import argparse
import os
import shutil
import uuid
import cv2

from .config import *


def create_cfg_test_file(cfg_filename, width, height):
    darknet_cfg_dir = os.path.join(DARKNET_DIR, "cfg")
    test_cfg_path = os.path.join(darknet_cfg_dir, cfg_filename)

    shutil.copyfile(
        os.path.join(darknet_cfg_dir, YOLO_CFG_FILENAME),
        test_cfg_path,
    )

    cmd = (
        'sed -i "s/batch={batch}/batch={new_batch}/" {cfg_path};'
        + 'sed -i "s/subdivisions={subdivisions}/subdivisions={new_subdivisions}/" {cfg_path};'
        + 'sed -i "s/width={width}/width={new_width}/" {cfg_path};'
        + 'sed -i "s/height={height}/height={new_height}/" {cfg_path};'
    )
    cmd = cmd.format(
        cfg_path=test_cfg_path,
        batch=YOLO_BATCH,
        new_batch=1,
        subdivisions=YOLO_SUBDIVISIONS,
        new_subdivisions=1,
        width=YOLO_IMG_SIZE,
        new_width=width,
        height=YOLO_IMG_SIZE,
        new_height=height,
    )
    os.system(cmd)


def detect(cfg_filename, test_img_filename):
    test_img_dir = os.path.join(WORKING_DIR, "test_images")

    img = cv2.imread(os.path.join(test_img_dir, test_img_filename), 0)
    filename = uuid.uuid4().hex + ".jpg"
    cv2.imwrite(os.path.join(test_img_dir, filename), img)

    os.chdir(DARKNET_DIR)
    cmd = "{} detector test {} {} {} -ext_output -dont_show {} > {}"
    os.system(
        cmd.format(
            os.path.join(DARKNET_DIR, "darknet"),
            os.path.join(DARKNET_DIR, "data", "obj.data"),
            os.path.join(DARKNET_DIR, "cfg", cfg_filename),
            os.path.join(TRAIN_RESULT_DIR, "yolov4-tiny_final.weights"),
            os.path.join(test_img_dir, filename),
            os.path.join(test_img_dir, "result.txt"),
        )
    )

    cmd = "cp {} {}"
    os.system(
        cmd.format(
            os.path.join(DARKNET_DIR, "predictions.jpg"),
            os.path.join(test_img_dir, "output_" + test_img_filename),
        )
    )

    os.remove(os.path.join(WORKING_DIR, "test_images", filename))


def remove_cfg_test_file(cfg_filename):
    os.remove(os.path.join(DARKNET_DIR, "cfg", cfg_filename))


def run_test(width, height, filename, change_working_dir=False, working_dir_path=None):
    cfg_test_filename = uuid.uuid4().hex + ".cfg"

    create_cfg_test_file(cfg_test_filename, width, height)
    detect(cfg_test_filename, filename)
    remove_cfg_test_file(cfg_test_filename)

    if change_working_dir:
        os.chdir(working_dir_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=str, required=True)
    parser.add_argument("--height", type=str, required=True)
    parser.add_argument("--filename", type=str, required=True)
    args = parser.parse_args()

    run_test(args.width, args.height, args.filename)
