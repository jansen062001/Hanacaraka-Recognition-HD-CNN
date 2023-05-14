import os
import glob
import shutil


def get_coarse_class_name(class_id, classes_txt_path):
    with open(classes_txt_path, "r") as f:
        idx = 0

        for line in f.readlines():
            if str(idx) != class_id:
                idx += 1
                continue

            box = "{}".format(line.strip())
            class_name = box.strip().split()[0]
            coarse_class_name = class_name.split("_")[0]

            return coarse_class_name


current_file_path = os.path.dirname(os.path.abspath(__file__))
yolo_dataset_dir = os.path.join(current_file_path, "yolov4_darknet", "dataset")
raw_yolo_dataset_dir = os.path.join(current_file_path, "dataset", "yolov4_darknet")
ext = ["jpg", "png", "jpeg"]
class_list = []
classes_txt_path = os.path.join(raw_yolo_dataset_dir, "classes.txt")

for ext in ext:
    path = os.path.join(raw_yolo_dataset_dir, "*" + ext)
    for path_and_filename in glob.iglob(path):
        title, ext = os.path.splitext(os.path.basename(path_and_filename))

        with open(os.path.join(yolo_dataset_dir, title + ".txt"), "w") as new_yolo_txt:
            with open(os.path.join(raw_yolo_dataset_dir, title + ".txt"), "r") as f:
                for line in f.readlines():
                    box = "{}".format(line.strip())
                    class_id, x_center, y_center, w, h = box.strip().split()

                    coarse_class_name = get_coarse_class_name(
                        class_id, classes_txt_path
                    )
                    if coarse_class_name not in class_list:
                        class_list.append(coarse_class_name)

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

        target_path = os.path.join(yolo_dataset_dir, title + ext)
        shutil.copyfile(path_and_filename, target_path)
        print("Done generate " + title + ext)

with open(os.path.join(yolo_dataset_dir, "classes.txt"), "w") as f:
    for class_name in class_list:
        f.write(class_name + "\n")
    print("Done generate classes.txt")
