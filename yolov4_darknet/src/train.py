import os
import glob
import shutil
import random

YOLO_WEIGHTS_URL = 'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137'
YOLO_WEIGHTS_NAME = 'yolov4.conv.137'
NUMBER_OF_CLASSES = 9
PERCENTAGE_TRAIN = 90
PERCENTAGE_TEST = 100 - PERCENTAGE_TRAIN
IMG_DATASET_EXT = ['jpg', 'png', 'jpeg']
YOLO_CFG_FILENAME = 'yolov4-custom.cfg'


def make_train_dir():
    if os.path.exists(os.getcwd() + '/training') == False:
        os.mkdir(os.getcwd() + '/training')


def build_darknet():
    current_dir = os.getcwd()

    cmd = 'git clone https://github.com/AlexeyAB/darknet.git'
    os.system(cmd)

    os.chdir(os.path.join(current_dir, 'darknet'))

    cmd = 'sed -i "s/OPENCV=0/OPENCV=1/" Makefile;' + \
        'sed -i "s/GPU=0/GPU=1/" Makefile;' + \
        'sed -i "s/CUDNN=0/CUDNN=1/" Makefile;' + \
        'sed -i "s/CUDNN_HALF=0/CUDNN_HALF=1/" Makefile;' + \
        'sed -i "s/ARCH= -gencode arch=compute_35,code=sm_35/ARCH= -gencode arch=compute_61,code=[sm_61,compute_61]/" Makefile;' + \
        'make'
    os.system(cmd)

    cmd = 'wget ' + YOLO_WEIGHTS_URL
    os.system(cmd)

    os.chdir(current_dir)


def move_dataset_folder():
    current_dir = os.getcwd()
    dataset_dir = os.path.join(current_dir, 'dataset')
    target_dir = os.path.join(current_dir, 'darknet', 'data', 'obj')

    if os.path.exists(target_dir) == False:
        os.mkdir(target_dir)

    for ext_name in IMG_DATASET_EXT:
        for path_and_filename in glob.iglob(dataset_dir + '/' + '*.' + ext_name):
            file_title, file_ext = os.path.splitext(
                os.path.basename(path_and_filename))

            shutil.move(path_and_filename,
                        target_dir + '/' + file_title + file_ext)

            shutil.move(dataset_dir + '/' + file_title + '.txt',
                        target_dir + '/' + file_title + '.txt')


def create_obj_data():
    with open(os.path.join(os.getcwd(), 'darknet', 'data', 'obj.data'), 'w') as f:
        f.write('classes = ' + str(NUMBER_OF_CLASSES) + '\n')
        f.write('train = data/train.txt' + '\n')
        f.write('valid = data/test.txt' + '\n')
        f.write('names = data/obj.names' + '\n')
        f.write('backup = ../training')


def create_obj_names():
    dataset_dir = os.path.join(os.getcwd(), 'dataset')
    data_dir = os.path.join(os.getcwd(), 'darknet', 'data')

    shutil.move(dataset_dir + '/classes.txt',
                data_dir + '/obj.names')


def split_train_valid():
    data_dir = os.path.join(os.getcwd(), 'darknet', 'data')
    file_train = open(os.path.join(data_dir, 'train.txt'), 'w')
    file_test = open(os.path.join(data_dir, 'test.txt'), 'w')
    dataset = []
    test_data_idx = []

    for ext_name in IMG_DATASET_EXT:
        for path_and_filename in glob.iglob(data_dir + '/obj/' + '*.' + ext_name):
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
    cfg_dir = os.path.join(os.getcwd(), 'darknet', 'cfg')

    shutil.move(os.path.join(os.getcwd(), YOLO_CFG_FILENAME),
                cfg_dir + '/' + YOLO_CFG_FILENAME)


def run_training():
    current_dir = os.getcwd()

    os.chdir(os.path.join(current_dir, 'darknet'))

    cmd = './darknet detector train data/obj.data cfg/' + \
        YOLO_CFG_FILENAME + ' ' + YOLO_WEIGHTS_NAME + ' -dont_show -map'
    os.system(cmd)

    os.chdir(current_dir)


if __name__ == "__main__":
    # Change working directory
    working_dir = os.path.abspath(
        os.path.dirname(os.path.abspath(__file__)) + '/../')
    os.chdir(working_dir)

    make_train_dir()
    build_darknet()
    move_dataset_folder()
    create_obj_data()
    create_obj_names()
    split_train_valid()
    move_cfg_file()
    run_training()
