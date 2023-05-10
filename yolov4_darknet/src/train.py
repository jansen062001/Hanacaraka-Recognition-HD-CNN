import os
import glob
import shutil
import random

from config import *


def make_train_dir():
    if os.path.exists(TRAIN_RESULT_DIR) == False:
        os.mkdir(TRAIN_RESULT_DIR)


def build_darknet():
    os.chdir(WORKING_DIR)

    cmd = 'git clone https://github.com/AlexeyAB/darknet.git'
    os.system(cmd)

    os.chdir(DARKNET_DIR)

    cmd = 'sed -i "s/OPENCV=0/OPENCV=1/" Makefile;' + \
        'sed -i "s/GPU=0/GPU=1/" Makefile;' + \
        'sed -i "s/CUDNN=0/CUDNN=1/" Makefile;' + \
        'sed -i "s/CUDNN_HALF=0/CUDNN_HALF=1/" Makefile;' + \
        'sed -i "s/ARCH= -gencode arch=compute_35,code=sm_35/ARCH= ' + ARCH + '/" Makefile;' + \
        'make'
    os.system(cmd)

    cmd = 'wget ' + YOLO_WEIGHTS_URL
    os.system(cmd)


def move_dataset_folder():
    if os.path.exists(DARKNET_DATA_DIR) == False:
        os.mkdir(DARKNET_DATA_DIR)

    for ext_name in IMG_DATASET_EXT:
        for path_and_filename in glob.iglob(os.path.join(DATA_DIR, '*.' + ext_name)):
            file_title, file_ext = os.path.splitext(
                os.path.basename(path_and_filename))

            shutil.move(path_and_filename,
                        os.path.join(DARKNET_DATA_DIR, file_title + file_ext))

            shutil.move(os.path.join(DATA_DIR, file_title + '.txt'),
                        os.path.join(DARKNET_DATA_DIR, file_title + '.txt'))


def create_obj_data():
    with open(os.path.join(DARKNET_DIR, 'data', 'obj.data'), 'w') as f:
        f.write('classes = ' + str(NUMBER_OF_CLASSES) + '\n')
        f.write('train = data/train.txt' + '\n')
        f.write('valid = data/test.txt' + '\n')
        f.write('names = data/obj.names' + '\n')
        f.write('backup = ../' + TRAIN_RESULT_DIRNAME)


def create_obj_names():
    shutil.move(os.path.join(DATA_DIR, 'classes.txt'),
                os.path.join(DARKNET_DIR, 'data', 'obj.names'))


def split_train_valid():
    file_train = open(os.path.join(DARKNET_DIR, 'data', 'train.txt'), 'w')
    file_test = open(os.path.join(DARKNET_DIR, 'data', 'test.txt'), 'w')
    dataset = []
    test_data_idx = []

    for ext_name in IMG_DATASET_EXT:
        for path_and_filename in glob.iglob(os.path.join(DARKNET_DATA_DIR, '*.' + ext_name)):
            dataset.append(path_and_filename)

    while True:
        random_idx = random.randint(0, len(dataset) - 1)

        if random_idx not in test_data_idx:
            test_data_idx.append(random_idx)

        if len(test_data_idx) == int(len(dataset) * PERCENTAGE_TEST / 100):
            break

    for i in range(len(dataset)):
        file_title, file_ext = os.path.splitext(
            os.path.basename(dataset[i]))

        if i in test_data_idx:
            file_test.write("data/obj" + "/" + file_title + file_ext + "\n")
        else:
            file_train.write("data/obj" + "/" + file_title + file_ext + "\n")

    file_train.close()
    file_test.close()


def move_cfg_file():
    shutil.move(os.path.join(WORKING_DIR, YOLO_CFG_FILENAME),
                os.path.join(DARKNET_DIR, 'cfg', YOLO_CFG_FILENAME))


def run_training():
    os.chdir(DARKNET_DIR)

    cmd = './darknet detector train data/obj.data cfg/' + \
        YOLO_CFG_FILENAME + ' ' + YOLO_WEIGHTS_NAME + ' -dont_show -map'
    os.system(cmd)


if __name__ == "__main__":
    make_train_dir()
    build_darknet()
    move_dataset_folder()
    create_obj_data()
    create_obj_names()
    split_train_valid()
    move_cfg_file()
    run_training()
