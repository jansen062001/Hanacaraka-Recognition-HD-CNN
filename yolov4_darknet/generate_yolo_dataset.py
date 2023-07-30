import os
import glob
import shutil
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .config import (
    RAW_YOLO_CLASSES_TXT,
    RAW_YOLO_DATASET_DIR,
    YOLO_IMG_DATASET_EXT,
    TRAIN_YOLO_DATASET_DIR,
    VALID_YOLO_DATASET_DIR,
    TEST_YOLO_DATASET_DIR,
    DONE_YOLO_CLASSES_TXT,
)


def get_coarse_class_name(class_id):
    with open(RAW_YOLO_CLASSES_TXT, "r") as f:
        idx = 0

        for line in f.readlines():
            if str(idx) != class_id:
                idx += 1
                continue

            box = "{}".format(line.strip())
            class_name = box.strip().split()[0]
            coarse_class_name = class_name.split("_")[0]

            return coarse_class_name


def split_dataset(list_img_path, train_size, valid_size, test_size):
    df = pd.DataFrame(list_img_path)
    train_size /= 100
    valid_size /= 100
    test_size /= 100

    x_train, x_test, y_train, y_test = train_test_split(
        df[0], df.index, test_size=1 - train_size, random_state=42
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_test, y_test, test_size=test_size / (test_size + valid_size), random_state=42
    )

    return x_train, x_val, x_test, y_train, y_val, y_test


def copy_dataset(x, target_path, class_list, desc):
    idx = list(x.index)
    for i in tqdm(range(len(idx)), desc=desc):
        title, ext = os.path.splitext(os.path.basename(x[idx[i]]))

        with open(os.path.join(target_path, title + ".txt"), "w") as new_yolo_txt:
            with open(os.path.join(RAW_YOLO_DATASET_DIR, title + ".txt"), "r") as f:
                for line in f.readlines():
                    box = "{}".format(line.strip())
                    class_id, x_center, y_center, w, h = box.strip().split()

                    coarse_class_name = get_coarse_class_name(class_id)
                    new_class_id = class_list.index(coarse_class_name)

                    new_yolo_txt.write(
                        str(new_class_id)
                        + " "
                        + x_center
                        + " "
                        + y_center
                        + " "
                        + w
                        + " "
                        + h
                        + "\n"
                    )

        shutil.copyfile(
            x[idx[i]],
            os.path.join(target_path, title + ext),
        )


def main(args):
    train_size = int(args.train_size)
    valid_size = int(args.valid_size)
    test_size = int(args.test_size)

    if train_size + valid_size + test_size != 100:
        print("INVALID SPLIT SIZE")
        return
    if not os.path.exists(RAW_YOLO_CLASSES_TXT):
        print("classes.txt doesn't exist")
        return

    list_img_path = []
    class_list = []
    for ext in YOLO_IMG_DATASET_EXT:
        path = os.path.join(RAW_YOLO_DATASET_DIR, "*." + ext)
        for path_and_filename in glob.iglob(path):
            list_img_path.append(path_and_filename)

            title, _ = os.path.splitext(os.path.basename(path_and_filename))
            txt_path = os.path.join(RAW_YOLO_DATASET_DIR, title + ".txt")
            with open(txt_path, "r") as f:
                for line in f.readlines():
                    box = "{}".format(line.strip())
                    class_id, _, _, _, _ = box.strip().split()

                    coarse_class_name = get_coarse_class_name(class_id)
                    if coarse_class_name not in class_list:
                        class_list.append(coarse_class_name)

    x_train, x_val, x_test, _, _, _ = split_dataset(
        list_img_path, train_size, valid_size, test_size
    )

    copy_dataset(
        x_train,
        TRAIN_YOLO_DATASET_DIR,
        class_list,
        "Copy & Re-Label Train Data",
    )
    copy_dataset(
        x_val,
        VALID_YOLO_DATASET_DIR,
        class_list,
        "Copy & Re-Label Validation Data",
    )
    copy_dataset(
        x_test,
        TEST_YOLO_DATASET_DIR,
        class_list,
        "Copy & Re-Label Test Data",
    )

    with open(os.path.join(DONE_YOLO_CLASSES_TXT), "w") as f:
        for class_name in class_list:
            f.write(class_name + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_size", type=str, required=True)
    parser.add_argument("--valid_size", type=str, required=True)
    parser.add_argument("--test_size", type=str, required=True)
    args = parser.parse_args()

    main(args)
