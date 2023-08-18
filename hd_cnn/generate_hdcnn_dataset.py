import os
import glob
import cv2
import operator
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
from tqdm import tqdm

from . import config as hdcnn_config


def crop_img(img, x_center, y_center, w, h):
    dh, dw, _ = img.shape

    x_center, y_center, w, h = float(x_center), float(y_center), float(w), float(h)
    x_center = round(x_center * dw)
    y_center = round(y_center * dh)
    w = round(w * dw)
    h = round(h * dh)
    x = round(x_center - w / 2)
    y = round(y_center - h / 2)

    cropped_img = img[y : y + h, x : x + w]

    return cropped_img


def get_class_name(class_id):
    with open(hdcnn_config.RAW_CLASSES_TXT_PATH, "r") as f:
        idx = 0

        for line in f.readlines():
            if str(idx) != class_id:
                idx += 1
                continue

            box = "{}".format(line.strip())
            class_name = box.strip().split()[0]

            return class_name


def generate_filtered_dataset(x, y):
    list_uniq_label = np.unique(np.array(y))
    new_x = []
    new_y = []
    new_list_label = {}

    for label in list_uniq_label:
        if (
            hdcnn_config.MIN_DATA_EACH_CLASS is not None
            and y.count(label) < hdcnn_config.MIN_DATA_EACH_CLASS
        ):
            continue

        new_list_label[label] = {"count": 0}

    for i in range(len(x)):
        if y[i] not in new_list_label:
            continue
        if (
            hdcnn_config.MAX_DATA_EACH_CLASS is not None
            and new_list_label[y[i]]["count"] + 1 > hdcnn_config.MAX_DATA_EACH_CLASS
        ):
            continue

        new_list_label[y[i]]["count"] += 1

        new_x.append(x[i])
        new_y.append(y[i])

    return new_x, new_y


def get_list_coarse_fine_class(y):
    coarse = []
    fine = []
    uniq_class = np.unique(np.array(y))

    for class_name in uniq_class:
        coarse_class_name = class_name.split("_")[0]
        if coarse_class_name not in coarse:
            coarse.append(coarse_class_name)

        fine.append(class_name)

    return coarse, fine


def split_dataset(x, y, train_size, valid_size, test_size):
    train_size /= 100
    valid_size /= 100
    test_size /= 100

    x_train, x_val_test, y_train, y_val_test = train_test_split(
        x, y, stratify=y, random_state=42, test_size=1 - train_size
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_val_test,
        y_val_test,
        stratify=y_val_test,
        random_state=42,
        test_size=test_size / (test_size + valid_size),
    )

    return x_train, x_val, x_test, y_train, y_val, y_test


def get_processed_dataset():
    x = []
    y = []

    for dataset_ext in hdcnn_config.IMG_DATASET_EXT:
        path = os.path.join(hdcnn_config.RAW_DATASET_DIR, "*." + dataset_ext)
        for path_and_filename in glob.iglob(path):
            title, _ = os.path.splitext(os.path.basename(path_and_filename))
            img = cv2.imread(path_and_filename)

            txt_path = os.path.join(hdcnn_config.RAW_DATASET_DIR, title + ".txt")
            with open(txt_path, "r") as f:
                for line in f.readlines():
                    box = "{}".format(line.strip())
                    class_id, x_center, y_center, w, h = box.strip().split()

                    cropped_img = crop_img(img, x_center, y_center, w, h)
                    h, w, _ = cropped_img.shape
                    if h == 0 or w == 0:
                        continue

                    resized_img = cv2.resize(
                        cropped_img,
                        (
                            hdcnn_config.HD_CNN_IMG_WIDTH,
                            hdcnn_config.HD_CNN_IMG_HEIGHT,
                        ),
                    )

                    class_name = get_class_name(class_id)

                    x.append(resized_img)
                    y.append(class_name)

    return x, y


def save_dataset(
    x_train_or_val, y_train_or_val, coarse_class, fine_class, is_train_dataset
):
    dataset_dir = (
        hdcnn_config.TRAIN_DATASET_DIR
        if is_train_dataset
        else hdcnn_config.VALID_DATASET_DIR
    )
    coarse_dir = os.path.join(dataset_dir, "coarse")
    fine_dir = os.path.join(dataset_dir, "fine")
    coarse_fine_dir = os.path.join(dataset_dir, "coarse_fine")

    for idx_coarse in range(len(coarse_class)):
        first_dir = os.path.join(coarse_fine_dir, str(idx_coarse))
        if not os.path.exists(first_dir):
            os.mkdir(first_dir)

        for idx_fine in range(len(fine_class)):
            second_dir = os.path.join(first_dir, str(idx_fine))
            if not os.path.exists(second_dir):
                os.mkdir(second_dir)

    for idx in tqdm(range(len(x_train_or_val))):
        x = x_train_or_val[idx]
        y = y_train_or_val[idx]

        coarse_class_idx = coarse_class.index(y.split("_")[0])
        fine_class_idx = fine_class.index(y)

        coarse_class_dir = os.path.join(coarse_dir, str(coarse_class_idx))
        if not os.path.exists(coarse_class_dir):
            os.mkdir(coarse_class_dir)

        fine_class_dir = os.path.join(fine_dir, str(fine_class_idx))
        if not os.path.exists(fine_class_dir):
            os.mkdir(fine_class_dir)

        coarse_fine_class_dir = os.path.join(
            coarse_fine_dir, str(coarse_class_idx), str(fine_class_idx)
        )

        filename = str(idx + 1) + ".jpg"
        cv2.imwrite(os.path.join(coarse_class_dir, filename), x)
        cv2.imwrite(os.path.join(fine_class_dir, filename), x)
        cv2.imwrite(os.path.join(coarse_fine_class_dir, filename), x)


def save_test_dataset(x_test, y_test, fine_class):
    for idx in tqdm(range(len(x_test))):
        x = x_test[idx]
        y = y_test[idx]

        fine_class_idx = fine_class.index(y)

        fine_class_dir = os.path.join(
            hdcnn_config.TEST_DATASET_DIR, str(fine_class_idx)
        )
        if not os.path.exists(fine_class_dir):
            os.mkdir(fine_class_dir)

        filename = str(idx + 1) + ".jpg"
        cv2.imwrite(os.path.join(fine_class_dir, filename), x)


def write_sorted_coarse_class(coarse_class):
    with open(hdcnn_config.COARSE_CLASSES_TXT_PATH, "w") as f:
        coarse_class_dict = []
        for i in range(len(coarse_class)):
            coarse_class_dict.append({"idx": str(i), "class_name": coarse_class[i]})
        coarse_class_dict.sort(key=operator.itemgetter("idx"))

        for i in range(len(coarse_class_dict)):
            f.write(coarse_class_dict[i]["class_name"] + "\n")


def write_sorted_fine_class(fine_class):
    with open(hdcnn_config.FINE_CLASSES_TXT_PATH, "w") as f:
        fine_class_dict = []
        for i in range(len(fine_class)):
            fine_class_dict.append({"idx": str(i), "class_name": fine_class[i]})
        fine_class_dict.sort(key=operator.itemgetter("idx"))

        for i in range(len(fine_class_dict)):
            f.write(fine_class_dict[i]["class_name"] + "\n")


def main(args):
    train_size = int(args.train_size)
    valid_size = int(args.valid_size)
    test_size = int(args.test_size)

    if train_size + valid_size + test_size != 100:
        print("INVALID SPLIT SIZE")
        return
    if not os.path.exists(hdcnn_config.RAW_CLASSES_TXT_PATH):
        print("classes.txt doesn't exist")
        return
    if (
        hdcnn_config.MIN_DATA_EACH_CLASS is not None
        and hdcnn_config.MIN_DATA_EACH_CLASS <= 0
    ):
        print("MIN_DATA_EACH_CLASS must be > 0")
        return
    if (
        hdcnn_config.MAX_DATA_EACH_CLASS is not None
        and hdcnn_config.MAX_DATA_EACH_CLASS < hdcnn_config.MIN_DATA_EACH_CLASS
    ):
        print("MAX_DATA_EACH_CLASS must be >= MIN_DATA_EACH_CLASS")
        return

    x, y = get_processed_dataset()
    x, y = generate_filtered_dataset(x, y)
    x_train, x_val, x_test, y_train, y_val, y_test = split_dataset(
        x, y, train_size, valid_size, test_size
    )

    coarse_class, fine_class = get_list_coarse_fine_class(y)

    save_dataset(x_train, y_train, coarse_class, fine_class, True)
    save_dataset(x_val, y_val, coarse_class, fine_class, False)
    save_test_dataset(x_test, y_test, fine_class)

    write_sorted_coarse_class(coarse_class)
    write_sorted_fine_class(fine_class)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_size", type=str, required=True)
    parser.add_argument("--valid_size", type=str, required=True)
    parser.add_argument("--test_size", type=str, required=True)
    args = parser.parse_args()

    main(args)
