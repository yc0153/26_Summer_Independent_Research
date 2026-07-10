"""==============================================================
# 필요한 라이브러리 불러오기
=============================================================="""
import os

"""==============================================================
# 설정값 클래스 정의
=============================================================="""


class Config:
    def __init__(self):
        """==============================================================
        ## 일반 설정
        =============================================================="""
        # 실험을 같은 조건으로 반복하기 위한 random seed
        self.RANDOM_SEED = 10

        """==============================================================
        ## 경로 설정
        =============================================================="""
        # 현재 예제 파일이 들어 있는 root directory
        self.ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        # 합성 데이터 생성을 위한 배경/전경 reference image directory
        self.REFERENCE_BG_DIR = os.path.join(
            self.ROOT_DIR, "data", "reference_bg")
        self.REFERENCE_FG_DIR = os.path.join(
            self.ROOT_DIR, "data", "reference_fg")

        # 생성된 train/valid/test image directory
        self.TRAIN_DIR = os.path.join(self.ROOT_DIR, "data", "train")
        self.VALID_DIR = os.path.join(self.ROOT_DIR, "data", "valid")
        self.TEST_DIR = os.path.join(self.ROOT_DIR, "data", "test")

        # 각 split의 label json file path
        self.TRAIN_LABEL_PATH = os.path.join(self.TRAIN_DIR, "labels.json")
        self.VALID_LABEL_PATH = os.path.join(self.VALID_DIR, "labels.json")
        self.TEST_LABEL_PATH = os.path.join(self.TEST_DIR, "labels.json")

        # 학습된 model checkpoint를 저장할 경로와 파일 이름
        self.CHECKPOINT_DIR = os.path.join(self.ROOT_DIR, "checkpoint")
        self.CHECKPOINT_NAME = "detection_model.pt"

        """==============================================================
        ## 클래스 설정
        =============================================================="""
        # reference_fg 하위 폴더 이름에서 class name과 label mapping 생성
        self.class_name_to_label = self.get_class_names_and_labels()

        """==============================================================
        ## 입력 이미지와 합성 데이터 설정
        =============================================================="""
        # 모델 입력 이미지 크기
        self.MODEL_IMAGE_HEIGHT = 240
        self.MODEL_IMAGE_WIDTH = 320

        # 생성 이미지 확장자
        self.IMAGE_EXTENSIONS = ".jpg"

        # train/valid/test 합성 이미지 생성 개수
        self.TRAIN_IMAGE_COUNT = 7000
        self.VALID_IMAGE_COUNT = 1500
        self.TEST_IMAGE_COUNT = 1500

        # background class 비율과 foreground 크기 random scale 범위
        self.BG_RATIO = 0.1
        self.FG_SCALE_MIN = 0.25
        self.FG_SCALE_MAX = 1.0

        # foreground alpha channel에서 실제 물체로 볼 threshold
        self.FG_ALPHA_THRESHOLD = 64

        # 기존 생성 데이터를 지우고 다시 만들지 여부
        self.RECREATE_DATASET = True

        """==============================================================
        ## object detection model 설정
        =============================================================="""
        # convolution feature extractor의 channel 수
        self.NUM_CONV_HIDDEN_CHANNELS = [64, 128, 256, 512]

        # convolution layer 공통 설정
        self.CONV_KERNEL_SIZE = 3
        self.CONV_STRIDE = 1
        self.CONV_PADDING = 1

        # feature map downsampling을 위한 max pooling 설정
        self.MAX_POOL_KERNEL_SIZE = 2
        self.MAX_POOL_STRIDE = 2

        # classification 앞쪽 shared hidden layer 개수와 node 수
        self.NUM_HEAD_HIDDEN_LAYERS = 3
        self.HEAD_HIDDEN_NODES = 64

        # class loss에 비해 bbox loss를 얼마나 크게 반영할지 결정
        self.BBOX_LOSS_WEIGHT = 10.0

        """==============================================================
        ## 학습 설정
        =============================================================="""
        self.BATCH_SIZE = 32
        self.NUM_EPOCHS = 10
        self.LEARNING_RATE = 1e-3
        self.NUM_WORKERS = 0

        """==============================================================
        ## 추론 설정
        =============================================================="""
        # webcam index와 bounding box를 그릴 confidence threshold
        self.CAMERA_INDEX = 0
        self.INFERENCE_PROB_THRESHOLD = 0.1

    """==============================================================
    ## 함수: 클래스 이름과 label mapping 만들기
    =============================================================="""

    def get_class_names_and_labels(self):
        class_name_to_label = {}

        # 예: "1_mouse" 폴더명에서 label=1, class_name="mouse" 추출
        for folder_name in os.listdir(self.REFERENCE_FG_DIR):
            label_text, class_name = folder_name.split("_", 1)
            class_name_to_label[class_name] = int(label_text)

        return class_name_to_label
