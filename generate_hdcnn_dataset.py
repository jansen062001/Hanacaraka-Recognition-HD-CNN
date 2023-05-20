import os
import glob
import cv2

IMG_RESIZE_WIDTH = 32
IMG_RESIZE_HEIGHT = 32


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


current_file_path = os.path.dirname(os.path.abspath(__file__))
raw_hdcnn_dataset_dir = os.path.join(current_file_path, "dataset", "hd_cnn")
hdcnn_dataset_dir = os.path.join(current_file_path, "hd_cnn", "dataset")
ext = ["jpg", "png", "jpeg"]
classes_txt_path = os.path.join(raw_hdcnn_dataset_dir, "classes.txt")
coarse_dir = os.path.join(hdcnn_dataset_dir, "coarse")
fine_dir = os.path.join(hdcnn_dataset_dir, "fine")
coarse_fine_dir = os.path.join(hdcnn_dataset_dir, "coarse_fine")
coarse_class = []
fine_class = []

if os.path.exists(coarse_dir) == False:
    os.mkdir(coarse_dir)
if os.path.exists(fine_dir) == False:
    os.mkdir(fine_dir)
if os.path.exists(coarse_fine_dir) == False:
    os.mkdir(coarse_fine_dir)

count = 1
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
                    cropped_img, (IMG_RESIZE_WIDTH, IMG_RESIZE_HEIGHT)
                )

                class_name = get_class_name(class_id, classes_txt_path)
                coarse_class_name = class_name.split("_")[0]

                if coarse_class_name not in coarse_class:
                    coarse_class.append(coarse_class_name)
                if class_name not in fine_class:
                    fine_class.append(class_name)

                coarse_class_dir = os.path.join(
                    coarse_dir, str(coarse_class.index(coarse_class_name))
                )
                if os.path.exists(coarse_class_dir) == False:
                    os.mkdir(coarse_class_dir)

                fine_class_dir = os.path.join(
                    fine_dir, str(fine_class.index(class_name))
                )
                if os.path.exists(fine_class_dir) == False:
                    os.mkdir(fine_class_dir)

                coarse_fine_class_dir = os.path.join(
                    coarse_fine_dir, str(coarse_class.index(coarse_class_name))
                )
                if os.path.exists(coarse_fine_class_dir) == False:
                    os.mkdir(coarse_fine_class_dir)
                coarse_fine_class_dir = os.path.join(
                    coarse_fine_class_dir, str(fine_class.index(class_name))
                )
                if os.path.exists(coarse_fine_class_dir) == False:
                    os.mkdir(coarse_fine_class_dir)

                filename = str(count) + ".jpg"
                cv2.imwrite(os.path.join(coarse_class_dir, filename), resized_img)
                cv2.imwrite(os.path.join(fine_class_dir, filename), resized_img)
                cv2.imwrite(os.path.join(coarse_fine_class_dir, filename), resized_img)
                count += 1

                print("Done generate " + filename)

for i in range(len(coarse_class)):
    for j in range(len(fine_class)):
        path = os.path.join(coarse_fine_dir, str(i), str(j))
        if os.path.exists(path) == False:
            os.mkdir(path)

with open(os.path.join(hdcnn_dataset_dir, "coarse_classes.txt"), "w") as f:
    for class_name in coarse_class:
        f.write(class_name + "\n")
with open(os.path.join(hdcnn_dataset_dir, "fine_classes.txt"), "w") as f:
    for class_name in fine_class:
        f.write(class_name + "\n")
