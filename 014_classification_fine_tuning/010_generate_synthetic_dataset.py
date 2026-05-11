"""===========================================================================================
Required Libraries
==========================================================================================="""
import math
import os
import shutil

import cv2
import numpy as np
from tqdm import tqdm


"""===========================================================================================
Settings
==========================================================================================="""
classes = ["circle", "triangle", "square", "star"]
image_size = 224
num_images = 1000

dir_proj = os.getcwd()
dir_dataset = os.path.join(
    dir_proj, "014_classification_fine_tuning", "dataset")
rng = np.random.default_rng(seed=14)


"""===========================================================================================
Draw Synthetic Shapes
==========================================================================================="""


def draw_shape(img, name):
    cx, cy = rng.integers(low=70, high=image_size - 70, size=2)
    radius = int(rng.integers(low=38, high=62))

    if name == "circle":
        cv2.circle(img=img, center=(cx, cy), radius=radius,
                   color=(0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)

    elif name == "triangle":
        pts = np.array(object=[[cx, cy - radius],
                               [cx - radius, cy + radius],
                               [cx + radius, cy + radius]], dtype=np.int32)
        cv2.fillPoly(img=img, pts=[pts], color=(0, 0, 0),
                     lineType=cv2.LINE_AA)

    elif name == "square":
        cv2.rectangle(img=img, pt1=(cx - radius, cy - radius),
                      pt2=(cx + radius, cy + radius), color=(0, 0, 0),
                      thickness=-1, lineType=cv2.LINE_AA)

    elif name == "star":
        pts = []
        for i in range(10):
            r = radius if i % 2 == 0 else radius * 0.45
            angle = -math.pi / 2 + i * math.pi / 5
            pts.append([int(cx + r * math.cos(angle)),
                        int(cy + r * math.sin(angle))])
        cv2.fillPoly(img=img, pts=[np.array(object=pts, dtype=np.int32)],
                     color=(0, 0, 0), lineType=cv2.LINE_AA)


def make_image(class_name):
    img = np.full(shape=(image_size, image_size, 3),
                  fill_value=255, dtype=np.uint8)
    draw_shape(img=img, name=class_name)
    return img


"""===========================================================================================
Create Dataset
==========================================================================================="""
if os.path.exists(dir_dataset):
    shutil.rmtree(dir_dataset)

num_train = int(num_images * 0.6)
num_valid = int(num_images * 0.2)
num_test = num_images - num_train - num_valid

for split_name, count in [("train", num_train),
                          ("valid", num_valid),
                          ("test", num_test)]:
    dir_split = os.path.join(dir_dataset, split_name)
    os.makedirs(dir_split, exist_ok=True)

    labels = [classes[i % len(classes)] for i in range(count)]
    rng.shuffle(labels)

    for idx, class_name in enumerate(tqdm(iterable=labels,
                                          desc=f"Creating {split_name}")):
        file_name = f"{idx:04d}_{class_name}.png"
        cv2.imwrite(filename=os.path.join(dir_split, file_name),
                    img=make_image(class_name=class_name))

print("Saved dataset to:", dir_dataset)
print("Train / valid / test:", num_train, num_valid, num_test)
