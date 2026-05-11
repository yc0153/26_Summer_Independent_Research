"""===========================================================================================
Required Libraries
==========================================================================================="""
import glob
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


"""===========================================================================================
Settings
==========================================================================================="""
classes = ["circle", "triangle", "square", "star"]
image_size = 224

imagenet_mean = np.array(object=[0.485, 0.456, 0.406], dtype=np.float32)
imagenet_std = np.array(object=[0.229, 0.224, 0.225], dtype=np.float32)


"""===========================================================================================
Shape Dataset
==========================================================================================="""
class ShapeDataset(Dataset):
    def __init__(self, dir_split, return_path=False):
        self.paths = sorted(glob.glob(pathname=os.path.join(dir_split, "*.png")))
        self.return_path = return_path

        if not self.paths:
            raise FileNotFoundError("Run 010_generate_synthetic_dataset.py first.")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        class_name = os.path.splitext(os.path.basename(path))[0].split("_")[-1]

        img = cv2.imread(filename=path)
        img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
        img = cv2.resize(src=img, dsize=(image_size, image_size))
        img = img.astype(np.float32) / 255.0
        img = (img - imagenet_mean) / imagenet_std
        img = np.transpose(a=img, axes=(2, 0, 1))

        image = torch.tensor(data=img, dtype=torch.float32)
        label = torch.tensor(data=classes.index(class_name), dtype=torch.long)

        if self.return_path:
            return image, label, path

        return image, label
