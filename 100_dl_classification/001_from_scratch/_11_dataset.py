"""==============================================================
# 필요한 라이브러리 불러오기
=============================================================="""
import os
import json
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
# 데이터셋 클래스 정의
=============================================================="""


class DataSet(Dataset):
    def __init__(self, split_name):
        """==============================================================
        ## 데이터셋 디렉토리 정리 및 라벨 읽기
        =============================================================="""
        # 디렉도리 정리
        if split_name == 'train':
            self.split_dir = config.TRAIN_DIR
            self.label_path = config.TRAIN_LABEL_PATH
        elif split_name == 'valid':
            self.split_dir = config.VALID_DIR
            self.label_path = config.VALID_LABEL_PATH
        elif split_name == 'test':
            self.split_dir = config.TEST_DIR
            self.label_path = config.TEST_LABEL_PATH
        else:
            raise ValueError(f"ValueError")

        # 라벨 읽기
        with open(file=self.label_path, mode="r", encoding="utf-8") as label_file:
            self.labels = json.load(fp=label_file)

    def __len__(self):
        """==============================================================
        ## 데이터셋의 길이 반환
        =============================================================="""
        return len(self.labels)

    def __getitem__(self, index):
        """==============================================================
        ## index에 해당하는 이미지와 데이터 반환 (Tensor 자료형으로)
        =============================================================="""
        # 이미지 데이터 선택
        label = self.labels[index]['label']
        img_dir = os.path.join(self.split_dir, self.labels[index]['file_name'])
        img_bgr = cv2.imread(filename=str(img_dir), flags=cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(
            src=img_bgr, code=cv2.COLOR_BGR2RGB)  # RGB기준으로 모델 훈련

        # 텐서 변환
        image_tensor = TF.to_tensor(img_rgb)  # min-max 정규화까지 포함
        label_tensor = torch.tensor(int(label), dtype=torch.long)

        # index에 해당하는 이미지와 라벨 반환 (텐서 자료형)
        return image_tensor, label_tensor
