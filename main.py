import argparse
import cv2
import os
import numpy as np
from typing import Tuple
import uuid
import time

from hd_cnn import test as hd_cnn_test
from yolov4_darknet import config as yolo_config
from yolov4_darknet import test as yolo_test


def draw_bounding_box(img, list_bb_and_label):
    for bounding_box in list_bb_and_label:
        x0 = bounding_box["left_x"]
        x1 = bounding_box["left_x"] + bounding_box["width"]
        y0 = bounding_box["top_y"]
        y1 = bounding_box["top_y"] + bounding_box["height"]

        start_point = (int(x0), int(y0))
        end_point = (int(x1), int(y1))
        img = cv2.rectangle(img, start_point, end_point, color=(0, 255, 0), thickness=1)

        img = cv2.putText(
            img,
            bounding_box["label"],
            (int(x0), int(y0) - 2),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.4,
            color=(0, 255, 0),
            thickness=1,
        )

    return img


def resize_with_pad(
    image: np.array,
    new_shape: Tuple[int, int],
    padding_color: Tuple[int] = (255, 255, 255),
) -> np.array:
    """Maintains aspect ratio and resizes with padding.
    Params:
        image: Image to be resized.
        new_shape: Expected (width, height) of new image.
        padding_color: Tuple in BGR of padding color
    Returns:
        image: Resized image with padding
    """
    original_shape = (image.shape[1], image.shape[0])
    ratio = float(max(new_shape)) / max(original_shape)
    new_size = tuple([int(x * ratio) for x in original_shape])
    image = cv2.resize(image, new_size)
    delta_w = new_shape[0] - new_size[0]
    delta_h = new_shape[1] - new_size[1]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color
    )
    return image


if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, required=True)
    args = parser.parse_args()

    current_file_path = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(current_file_path, args.filename)
    img = cv2.imread(img_path)
    height, width, channel = img.shape

    resize_to = 0
    size = width if width > height else height
    while True:
        if size % yolo_config.YOLO_IMG_SIZE == 0:
            resize_to = size
            break
        size += 1
    new_img = resize_with_pad(img, (resize_to, resize_to), (114, 114, 114))
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

    list_bb_and_label = []
    for h in range(0, resize_to, yolo_config.YOLO_IMG_SIZE):
        for w in range(0, resize_to, yolo_config.YOLO_IMG_SIZE):
            cropped_img = new_img[
                h : h + yolo_config.YOLO_IMG_SIZE, w : w + yolo_config.YOLO_IMG_SIZE
            ]

            yolo_test_img_dir = os.path.join(yolo_config.YOLO_DIR, "data")
            filename = uuid.uuid4().hex + ".jpg"
            yolo_test_img_path = os.path.join(yolo_test_img_dir, filename)
            cv2.imwrite(yolo_test_img_path, cropped_img)

            yolo_test.run_test(
                str(yolo_config.YOLO_IMG_SIZE),
                str(yolo_config.YOLO_IMG_SIZE),
                filename,
                True,
                current_file_path,
            )

            with open(os.path.join(yolo_test_img_dir, "result.txt"), "r") as f:
                for line in f.readlines():
                    line = line.strip()

                    if (
                        "left_x" not in line
                        or "top_y" not in line
                        or "width" not in line
                        or "height" not in line
                    ):
                        continue

                    str_left_x = "left_x:"
                    str_left_x_idx = line.index(str_left_x)

                    str_top_y = "top_y:"
                    str_top_y_idx = line.index(str_top_y)

                    str_width = "width:"
                    str_width_idx = line.index(str_width)

                    str_height = "height:"
                    str_height_idx = line.index(str_height)

                    left_x = int(
                        line[
                            str_left_x_idx + len(str_left_x) : str_top_y_idx - 1
                        ].strip()
                    )
                    top_y = int(
                        line[str_top_y_idx + len(str_top_y) : str_width_idx - 1].strip()
                    )
                    width = int(
                        line[
                            str_width_idx + len(str_width) : str_height_idx - 1
                        ].strip()
                    )
                    height = int(
                        line[str_height_idx + len(str_height) : len(line) - 1].strip()
                    )

                    if left_x < 0:
                        left_x = 0
                    if top_y < 0:
                        top_y = 0
                    if width < 0:
                        width = 0
                    if height < 0:
                        height = 0

                    list_bb_and_label.append(
                        {
                            "left_x": left_x + w,
                            "top_y": top_y + h,
                            "width": width,
                            "height": height,
                            "label": "",
                        }
                    )

    gray_3_ch = np.zeros((resize_to, resize_to, 3))
    gray_3_ch[:, :, 0] = new_img
    gray_3_ch[:, :, 1] = new_img
    gray_3_ch[:, :, 2] = new_img

    x = []
    for i in range(len(list_bb_and_label)):
        top_y = list_bb_and_label[i]["top_y"]
        left_x = list_bb_and_label[i]["left_x"]
        width = list_bb_and_label[i]["width"]
        height = list_bb_and_label[i]["height"]
        predicted_img = gray_3_ch[top_y : top_y + height, left_x : left_x + width, :]

        x.append(predicted_img)

    y = hd_cnn_test.run_test(x)
    for i in range(len(list_bb_and_label)):
        list_bb_and_label[i]["label"] = y[i]

    img_with_bb = resize_with_pad(img, (resize_to, resize_to), (114, 114, 114))
    img_with_bb = draw_bounding_box(img_with_bb, list_bb_and_label)
    cv2.imwrite("result.jpg", img_with_bb)

    end_time = time.time()
    elapsed = end_time - start_time
    print("Elapsed time is %f seconds." % elapsed)
