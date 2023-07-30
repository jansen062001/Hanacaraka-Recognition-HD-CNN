import os
import glob
import shutil

from .config import *


def make_weights_dir():
    if not os.path.exists(WEIGHTS_DIR):
        os.mkdir(WEIGHTS_DIR)


def build_darknet():
    if not os.path.exists(DARKNET_DIR):
        os.mkdir(DARKNET_DIR)

    cmd = "git clone https://github.com/AlexeyAB/darknet.git {}"
    os.system(cmd.format(DARKNET_DIR))

    os.chdir(DARKNET_DIR)
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
            DARKNET_MAKEFILE_PATH,
            DARKNET_MAKEFILE_PATH,
            DARKNET_MAKEFILE_PATH,
            DARKNET_MAKEFILE_PATH,
            YOLO_ARCH,
            DARKNET_MAKEFILE_PATH,
        )
    )

    cmd = "wget -P {} {}"
    os.system(cmd.format(DARKNET_DIR, YOLO_PRETRAINED_WEIGHTS_URL))


def copy_processed_dataset():
    if not os.path.exists(DARKNET_DATASET_DIR):
        os.mkdir(DARKNET_DATASET_DIR)

    list_dir = [TRAIN_YOLO_DATASET_DIR, VALID_YOLO_DATASET_DIR, TEST_YOLO_DATASET_DIR]
    for directory in list_dir:
        for ext_name in YOLO_IMG_DATASET_EXT:
            for path_and_filename in glob.iglob(
                os.path.join(directory, "*." + ext_name)
            ):
                file_title, file_ext = os.path.splitext(
                    os.path.basename(path_and_filename)
                )

                shutil.copyfile(
                    path_and_filename,
                    os.path.join(DARKNET_DATASET_DIR, file_title + file_ext),
                )

                shutil.copyfile(
                    os.path.join(directory, file_title + ".txt"),
                    os.path.join(DARKNET_DATASET_DIR, file_title + ".txt"),
                )


def create_obj_data():
    with open(DARKNET_OBJ_DATA_PATH, "w") as f:
        f.write("classes = " + str(YOLO_NUMBER_OF_CLASSES) + "\n")
        f.write("train = data/train.txt" + "\n")
        f.write("valid = data/valid.txt" + "\n")
        f.write("names = data/obj.names" + "\n")
        f.write("backup = ../" + WEIGHTS_FOLDER_NAME)


def create_obj_names():
    shutil.copyfile(DONE_YOLO_CLASSES_TXT, DARKNET_OBJ_NAMES_PATH)


def create_train_valid_txt():
    darknet_data_dir = os.path.join(DARKNET_DIR, "data")
    train_txt_path = os.path.join(darknet_data_dir, "train.txt")
    valid_txt_path = os.path.join(darknet_data_dir, "valid.txt")

    list_txt_path = [train_txt_path, valid_txt_path]
    list_dataset_dir = [TRAIN_YOLO_DATASET_DIR, VALID_YOLO_DATASET_DIR]

    for i in range(len(list_txt_path)):
        with open(list_txt_path[i], "w") as f:
            for ext_name in YOLO_IMG_DATASET_EXT:
                for path_and_filename in glob.iglob(
                    os.path.join(list_dataset_dir[i], "*." + ext_name)
                ):
                    file_title, file_ext = os.path.splitext(
                        os.path.basename(path_and_filename)
                    )

                    f.write("data/obj/" + file_title + file_ext + "\n")


def copy_cfg_file():
    shutil.copyfile(
        os.path.join(YOLO_DIR, YOLO_CFG_FILENAME),
        os.path.join(DARKNET_DIR, "cfg", YOLO_CFG_FILENAME),
    )


def check_train_valid_data():
    list_txt_path = [
        os.path.join(DARKNET_DIR, "data", "train.txt"),
        os.path.join(DARKNET_DIR, "data", "valid.txt"),
    ]

    for txt in list_txt_path:
        with open(txt, "r") as f:
            for line in f.readlines():
                if not os.path.exists(os.path.join(DARKNET_DIR, line.strip())):
                    return False

    return True


def run_training():
    darknet_exe_path = os.path.join(DARKNET_DIR, "darknet")
    os.chdir(DARKNET_DIR)

    cmd = "{} detector train {} {} {} -dont_show -map"
    os.system(
        cmd.format(
            darknet_exe_path,
            DARKNET_OBJ_DATA_PATH,
            os.path.join(DARKNET_DIR, "cfg", YOLO_CFG_FILENAME),
            os.path.join(DARKNET_DIR, YOLO_PRETRAINED_WEIGHTS_NAME),
        )
    )


if __name__ == "__main__":
    make_weights_dir()
    build_darknet()
    copy_processed_dataset()
    create_obj_data()
    create_obj_names()
    create_train_valid_txt()
    copy_cfg_file()

    if check_train_valid_data():
        run_training()
    else:
        print("Cannot run training because train/valid data doesn't exist")
