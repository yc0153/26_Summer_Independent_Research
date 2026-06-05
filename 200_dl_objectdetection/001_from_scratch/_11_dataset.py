"""==============================================================
# 필요한 라이브러리 불러오기
=============================================================="""
import json
import os

import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from _00_config import Config

"""==============================================================
# 기본 설정
=============================================================="""

config = Config()

"""==============================================================
# PyTorch Dataset 클래스 정의
=============================================================="""


class DataSet(Dataset):
    def __init__(self, split_name):
        """==============================================================
        ## split별 directory와 label path 설정
        =============================================================="""
        # split_name에 따라 image directory 선택
        split_dir = {
            "train": config.TRAIN_DIR,
            "valid": config.VALID_DIR,
            "test": config.TEST_DIR,
        }

        # split_name에 따라 label json file 선택
        label_path = {
            "train": config.TRAIN_LABEL_PATH,
            "valid": config.VALID_LABEL_PATH,
            "test": config.TEST_LABEL_PATH,
        }

        self.split_dir = split_dir[split_name]

        # label json file 불러오기
        with open(file=label_path[split_name], mode="r", encoding="utf-8") as f:
            self.labels = json.load(fp=f)

    def __len__(self):
        # Dataset 전체 sample 개수 반환
        return len(self.labels)

    def __getitem__(self, index):
        """==============================================================
        ## image, class label, bbox tensor 만들기
        =============================================================="""
        # index에 해당하는 label record 선택
        label_info = self.labels[index]

        # OpenCV로 이미지를 불러온 뒤 BGR에서 RGB로 변환
        img_path = os.path.join(self.split_dir, label_info["file_name"])
        img_bgr = cv2.imread(filename=img_path, flags=cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(src=img_bgr, code=cv2.COLOR_BGR2RGB)

        # image는 float tensor, label은 long tensor, bbox는 float tensor로 변환
        image_tensor = TF.to_tensor(img_rgb)
        label_tensor = torch.tensor(label_info["label"], dtype=torch.long)
        bbox_tensor = torch.tensor(label_info["bbox"], dtype=torch.float32)

        return image_tensor, label_tensor, bbox_tensor
