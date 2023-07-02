import os
import glob
import cv2
import operator
import numpy as np
from sklearn.model_selection import train_test_split

from hd_cnn.src import config as hd_cnn_config
from yolov4_darknet.src import config as yolo_config

MIN_DATA_EACH_CLASS = 10
MAX_DATA_EACH_CLASS = None
TEST_SIZE = 0.1


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


def get_class_name(class_id, classes_txt_path):
    with open(classes_txt_path, "r") as f:
        idx = 0

        for line in f.readlines():
            if str(idx) != class_id:
                idx += 1
                continue

            box = "{}".format(line.strip())
            class_name = box.strip().split()[0]

            return class_name


def generate_filtered_dataset(x, y):
    dataset_idx_to_delete = []

    if MIN_DATA_EACH_CLASS is not None:
        list_label = np.unique(np.array(y))

        for i in range(len(list_label)):
            count = 0
            label_items_idx = []

            for j in range(len(y)):
                if y[j] == list_label[i]:
                    count += 1
                    label_items_idx.append(j)
                if count >= MIN_DATA_EACH_CLASS:
                    break

            if count < MIN_DATA_EACH_CLASS:
                for j in range(len(label_items_idx)):
                    dataset_idx_to_delete.append(label_items_idx[j])

    new_x = []
    new_y = []
    for i in range(len(x)):
        if i in dataset_idx_to_delete:
            continue
        if (
            MAX_DATA_EACH_CLASS is not None
            and new_y.count(y[i]) + 1 > MAX_DATA_EACH_CLASS
        ):
            continue

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


def main():
    if MIN_DATA_EACH_CLASS is not None and MIN_DATA_EACH_CLASS <= 0:
        print("MIN_DATA_EACH_CLASS must be > 0")
        return
    if MAX_DATA_EACH_CLASS is not None and MAX_DATA_EACH_CLASS < MIN_DATA_EACH_CLASS:
        print("MAX_DATA_EACH_CLASS must be >= MIN_DATA_EACH_CLASS")
        return

    current_file_path = os.path.dirname(os.path.abspath(__file__))
    raw_hdcnn_dataset_dir = os.path.join(current_file_path, "dataset", "hd_cnn")
    hdcnn_dataset_dir = os.path.join(current_file_path, "hd_cnn", "dataset")
    ext = yolo_config.YOLO_IMG_DATASET_EXT
    classes_txt_path = os.path.join(raw_hdcnn_dataset_dir, "classes.txt")
    coarse_dir = os.path.join(hdcnn_dataset_dir, "coarse")
    fine_dir = os.path.join(hdcnn_dataset_dir, "fine")
    coarse_fine_dir = os.path.join(hdcnn_dataset_dir, "coarse_fine")
    test_dir = os.path.join(hdcnn_dataset_dir, "test")

    if os.path.exists(coarse_dir) == False:
        os.mkdir(coarse_dir)
    if os.path.exists(fine_dir) == False:
        os.mkdir(fine_dir)
    if os.path.exists(coarse_fine_dir) == False:
        os.mkdir(coarse_fine_dir)
    if os.path.exists(test_dir) == False:
        os.mkdir(test_dir)

    x = []
    y = []
    for ext in ext:
        path = os.path.join(raw_hdcnn_dataset_dir, "*" + ext)
        for path_and_filename in glob.iglob(path):
            title, ext = os.path.splitext(os.path.basename(path_and_filename))
            img = cv2.imread(path_and_filename)

            with open(os.path.join(raw_hdcnn_dataset_dir, title + ".txt"), "r") as f:
                for line in f.readlines():
                    box = "{}".format(line.strip())
                    class_id, x_center, y_center, w, h = box.strip().split()

                    cropped_img = crop_img(img, x_center, y_center, w, h)
                    resized_img = cv2.resize(
                        cropped_img,
                        (
                            hd_cnn_config.HD_CNN_IMG_WIDTH,
                            hd_cnn_config.HD_CNN_IMG_HEIGHT,
                        ),
                    )

                    class_name = get_class_name(class_id, classes_txt_path)

                    x.append(resized_img)
                    y.append(class_name)

    x, y = generate_filtered_dataset(x, y)
    coarse_class, fine_class = get_list_coarse_fine_class(y)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, stratify=y, test_size=TEST_SIZE
    )

    for i in range(len(x_train)):
        coarse_class_name = y_train[i].split("_")[0]
        coarse_class_dir = os.path.join(
            coarse_dir, str(coarse_class.index(coarse_class_name))
        )
        if os.path.exists(coarse_class_dir) == False:
            os.mkdir(coarse_class_dir)

        fine_class_dir = os.path.join(fine_dir, str(fine_class.index(y_train[i])))
        if os.path.exists(fine_class_dir) == False:
            os.mkdir(fine_class_dir)

        coarse_fine_class_dir = os.path.join(
            coarse_fine_dir, str(coarse_class.index(coarse_class_name))
        )
        if os.path.exists(coarse_fine_class_dir) == False:
            os.mkdir(coarse_fine_class_dir)

        coarse_fine_class_dir = os.path.join(
            coarse_fine_class_dir, str(fine_class.index(y_train[i]))
        )
        if os.path.exists(coarse_fine_class_dir) == False:
            os.mkdir(coarse_fine_class_dir)

        filename = str(i + 1) + ".jpg"
        cv2.imwrite(os.path.join(coarse_class_dir, filename), x_train[i])
        cv2.imwrite(os.path.join(fine_class_dir, filename), x_train[i])
        cv2.imwrite(os.path.join(coarse_fine_class_dir, filename), x_train[i])

        if i == len(x_train) - 1:
            print("Done generate " + str(i + 1) + " train data")

    for i in range(len(x_test)):
        fine_class_dir = os.path.join(test_dir, str(fine_class.index(y_test[i])))
        if os.path.exists(fine_class_dir) == False:
            os.mkdir(fine_class_dir)

        filename = str(i + 1) + ".jpg"
        cv2.imwrite(os.path.join(fine_class_dir, filename), x_test[i])

        if i == len(x_test) - 1:
            print("Done generate " + str(i + 1) + " test data")

    for i in range(len(coarse_class)):
        for j in range(len(fine_class)):
            path = os.path.join(coarse_fine_dir, str(i), str(j))
            if os.path.exists(path) == False:
                os.mkdir(path)

    with open(os.path.join(hdcnn_dataset_dir, "coarse_classes.txt"), "w") as f:
        coarse_class_dict = []
        for i in range(len(coarse_class)):
            coarse_class_dict.append({"idx": str(i), "class_name": coarse_class[i]})
        coarse_class_dict.sort(key=operator.itemgetter("idx"))

        for i in range(len(coarse_class_dict)):
            f.write(coarse_class_dict[i]["class_name"] + "\n")

    with open(os.path.join(hdcnn_dataset_dir, "fine_classes.txt"), "w") as f:
        fine_class_dict = []
        for i in range(len(fine_class)):
            fine_class_dict.append({"idx": str(i), "class_name": fine_class[i]})
        fine_class_dict.sort(key=operator.itemgetter("idx"))

        for i in range(len(fine_class_dict)):
            f.write(fine_class_dict[i]["class_name"] + "\n")


if __name__ == "__main__":
    main()
