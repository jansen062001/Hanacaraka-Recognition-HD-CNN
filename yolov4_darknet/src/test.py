import argparse
import os
import shutil
import uuid

from config import *


def create_cfg_test_file(cfg_filename):
    os.chdir(os.path.join(DARKNET_DIR, 'cfg'))

    shutil.copyfile(YOLO_CFG_FILENAME, cfg_filename)

    cmd = 'sed -i "s/batch=64/batch=1/" ' + cfg_filename + ';' + \
        'sed -i "s/subdivisions=64/subdivisions=1/" ' + cfg_filename + ';' + \
        'sed -i "s/width=416/width=' + args.width + '/" ' + cfg_filename + ';' + \
        'sed -i "s/height=416/height=' + args.height + '/" ' + cfg_filename + ';'
    os.system(cmd)


def run_test(cfg_filename, test_img_filename):
    os.chdir(DARKNET_DIR)

    cmd = './darknet detector test data/obj.data cfg/' + cfg_filename + \
        ' ../' + TRAIN_RESULT_DIRNAME + \
        '/yolov4-custom_best.weights ../test_images/' + test_img_filename
    os.system(cmd)

    cmd = 'cp predictions.jpg ../test_images/output_' + test_img_filename
    os.system(cmd)


def remove_cfg_test_file(cfg_filename):
    os.chdir(os.path.join(DARKNET_DIR, 'cfg'))

    os.remove(cfg_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=str, required=True)
    parser.add_argument('--height', type=str, required=True)
    parser.add_argument('--filename', type=str, required=True)
    args = parser.parse_args()

    cfg_test_filename = uuid.uuid4().hex + '.cfg'

    create_cfg_test_file(cfg_test_filename)
    run_test(cfg_test_filename, args.filename)
    remove_cfg_test_file(cfg_test_filename)
