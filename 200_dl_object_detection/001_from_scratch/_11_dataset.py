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
# 기본 설정값 불러오기
=============================================================="""
config = Config()

"""==============================================================
# 디텍션 데이터셋 클래스 정의
=============================================================="""


class DataSet(Dataset):
    def __init__(self, split_name):
        """==============================================================
        ## 데이터셋 directory와 label path 선택
        =============================================================="""
        # split 이름에 따라 image directory와 label json path 선택
        if split_name == "train":
            self.split_dir = config.TRAIN_DIR
            self.label_path = config.TRAIN_LABEL_PATH
        elif split_name == "valid":
            self.split_dir = config.VALID_DIR
            self.label_path = config.VALID_LABEL_PATH
        elif split_name == "test":
            self.split_dir = config.TEST_DIR
            self.label_path = config.TEST_LABEL_PATH
        else:
            raise ValueError(f"Invalid split name: {split_name}")

        # labels.json에서 image file name, class label, bbox 정보 읽기
        with open(file=self.label_path, mode="r", encoding="utf-8") as label_file:
            self.labels = json.load(fp=label_file)

    def __len__(self):
        """==============================================================
        ## 데이터셋 길이 반환
        =============================================================="""
        return len(self.labels)

    def __getitem__(self, index):
        """==============================================================
        ## index에 해당하는 이미지, 클래스 라벨, 바운딩 박스 반환
        =============================================================="""
        # 현재 index의 label record 선택
        record = self.labels[index]

        # 이미지 파일 읽기
        img_path = os.path.join(self.split_dir, record["file_name"])
        img_bgr = cv2.imread(filename=str(img_path), flags=cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(img_path)

        # OpenCV BGR 이미지를 모델 입력용 RGB 이미지로 변환
        img_rgb = cv2.cvtColor(src=img_bgr, code=cv2.COLOR_BGR2RGB)
        img_height, img_width = img_rgb.shape[:2]

        # 이미지와 class label을 tensor로 변환
        image_tensor = TF.to_tensor(img_rgb)
        label_tensor = torch.tensor(int(record["label"]), dtype=torch.long)

        # pixel bbox를 이미지 크기로 나누어 0~1 범위로 정규화
        bbox = record["bbox"]
        bbox_tensor = torch.tensor(
            [
                bbox["x_min"] / img_width,
                bbox["y_min"] / img_height,
                bbox["x_max"] / img_width,
                bbox["y_max"] / img_height,
            ],
            dtype=torch.float32,
        ).clamp(0.0, 1.0)

        return image_tensor, label_tensor, bbox_tensor
