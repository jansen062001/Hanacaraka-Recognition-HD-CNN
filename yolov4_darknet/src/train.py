import os
import glob
import shutil
import random

from .config import *


def make_train_dir():
    if os.path.exists(TRAIN_RESULT_DIR) == False:
        os.mkdir(TRAIN_RESULT_DIR)


def build_darknet():
    if os.path.exists(DARKNET_DIR) == False:
        os.mkdir(DARKNET_DIR)

    cmd = "git clone https://github.com/AlexeyAB/darknet.git {}"
    os.system(cmd.format(DARKNET_DIR))

    os.chdir(DARKNET_DIR)
    makefile_path = os.path.join(DARKNET_DIR, "Makefile")
    cmd = (
        'sed -i "s/OPENCV=0/OPENCV=1/" {};'
        + 'sed -i "s/GPU=0/GPU=1/" {};'
        + 'sed -i "s/CUDNN=0/CUDNN=1/" {};'
        + 'sed -i "s/CUDNN_HALF=0/CUDNN_HALF=1/" {};'
        + 'sed -i "s/ARCH= -gencode arch=compute_35,code=sm_35/ARCH= {}/" {};'
        + "make"
    )
    os.system(
        cmd.format(
            makefile_path,
            makefile_path,
            makefile_path,
            makefile_path,
            YOLO_ARCH,
            makefile_path,
        )
    )

    cmd = "wget -P {} {}"
    os.system(cmd.format(DARKNET_DIR, YOLO_PRETRAINED_WEIGHTS_URL))


def move_dataset_folder():
    if os.path.exists(DARKNET_DATA_DIR) == False:
        os.mkdir(DARKNET_DATA_DIR)

    for ext_name in YOLO_IMG_DATASET_EXT:
        for path_and_filename in glob.iglob(os.path.join(DATA_DIR, "*." + ext_name)):
            file_title, file_ext = os.path.splitext(os.path.basename(path_and_filename))

            shutil.move(
                path_and_filename, os.path.join(DARKNET_DATA_DIR, file_title + file_ext)
            )

            shutil.move(
                os.path.join(DATA_DIR, file_title + ".txt"),
                os.path.join(DARKNET_DATA_DIR, file_title + ".txt"),
            )


def create_obj_data():
    with open(os.path.join(DARKNET_DIR, "data", "obj.data"), "w") as f:
        f.write("classes = " + str(YOLO_NUMBER_OF_CLASSES) + "\n")
        f.write("train = data/train.txt" + "\n")
        f.write("valid = data/test.txt" + "\n")
        f.write("names = data/obj.names" + "\n")
        f.write("backup = ../" + TRAIN_RESULT_DIRNAME)


def create_obj_names():
    shutil.move(
        os.path.join(DATA_DIR, "classes.txt"),
        os.path.join(DARKNET_DIR, "data", "obj.names"),
    )


def split_train_valid():
    file_train = open(os.path.join(DARKNET_DIR, "data", "train.txt"), "w")
    file_test = open(os.path.join(DARKNET_DIR, "data", "test.txt"), "w")
    dataset = []
    test_data_idx = []

    for ext_name in YOLO_IMG_DATASET_EXT:
        for path_and_filename in glob.iglob(
            os.path.join(DARKNET_DATA_DIR, "*." + ext_name)
        ):
            dataset.append(path_and_filename)

    test_data_amount = int(len(dataset) * YOLO_PERCENTAGE_TEST / 100)
    while True:
        if test_data_amount <= 0:
            break

        random_idx = random.randint(0, len(dataset) - 1)

        if random_idx not in test_data_idx:
            test_data_idx.append(random_idx)

        if len(test_data_idx) == test_data_amount:
            break

    for i in range(len(dataset)):
        file_title, file_ext = os.path.splitext(os.path.basename(dataset[i]))

        if i in test_data_idx:
            file_test.write("data/obj" + "/" + file_title + file_ext + "\n")
        else:
            file_train.write("data/obj" + "/" + file_title + file_ext + "\n")

    file_train.close()
    file_test.close()


def move_cfg_file():
    shutil.copyfile(
        os.path.join(WORKING_DIR, YOLO_CFG_FILENAME),
        os.path.join(DARKNET_DIR, "cfg", YOLO_CFG_FILENAME),
    )


def check_train_test_data():
    with open(os.path.join(DARKNET_DIR, "data", "train.txt"), "r") as f:
        for line in f.readlines():
            if os.path.exists(os.path.join(DARKNET_DIR, line.strip())) == False:
                return False

    with open(os.path.join(DARKNET_DIR, "data", "test.txt"), "r") as f:
        for line in f.readlines():
            if os.path.exists(os.path.join(DARKNET_DIR, line.strip())) == False:
                return False

    return True


def run_training():
    darknet_exe_path = os.path.join(DARKNET_DIR, "darknet")
    os.chdir(DARKNET_DIR)

    cmd = "{} detector train {} {} {} -dont_show -map"
    os.system(
        cmd.format(
            darknet_exe_path,
            os.path.join(DARKNET_DIR, "data", "obj.data"),
            os.path.join(DARKNET_DIR, "cfg", YOLO_CFG_FILENAME),
            os.path.join(DARKNET_DIR, YOLO_PRETRAINED_WEIGHTS_NAME),
        )
    )


if __name__ == "__main__":
    make_train_dir()
    build_darknet()
    move_dataset_folder()
    create_obj_data()
    create_obj_names()
    split_train_valid()
    move_cfg_file()

    if check_train_test_data():
        run_training()
    else:
        print("Cannot run training because train/test data doesn't exist")
