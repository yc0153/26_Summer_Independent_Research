"""==============================================================
# 라이브러리 불러오기
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
        # 실험 재현을 위한 random seed
        self.RANDOM_SEED = 10

        """==============================================================
        ## 경로 설정
        =============================================================="""
        # 현재 예제의 root directory
        self.ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        # 합성 데이터 생성을 위한 reference image directory
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

        # model checkpoint directory
        self.CHECKPOINT_DIR = os.path.join(self.ROOT_DIR, "checkpoint")

        """==============================================================
        ## 클래스 설정
        =============================================================="""
        # reference_fg 하위 폴더 이름에서 class label mapping 생성
        self.class_name_to_label = self.get_class_names_and_labels()

        """==============================================================
        ## 입력 이미지 설정
        =============================================================="""
        # 모델 입력 이미지 크기
        self.MODEL_IMAGE_HEIGHT = 240
        self.MODEL_IMAGE_WIDTH = 320

        # 생성 이미지와 mask 확장자
        self.IMAGE_EXTENSIONS = ".jpg"
        self.MASK_EXTENSIONS = ".png"

        # train/valid/test 합성 이미지 생성 개수
        self.TRAIN_IMAGE_COUNT = 700
        self.VALID_IMAGE_COUNT = 150
        self.TEST_IMAGE_COUNT = 150

        # 세그멘테이션 학습을 위한 합성 데이터 비율과 전경 크기
        self.BG_RATIO = 0.2
        self.FG_SCALE_MIN = 0.25
        self.FG_SCALE_MAX = 0.75
        self.FG_ALPHA_THRESHOLD = 64

        # 기존 생성 데이터를 지운 뒤 다시 생성할지 여부
        self.RECREATE_DATASET = True

        """==============================================================
        ## 세그멘테이션 모델 학습 설정
        =============================================================="""
        # 새 학습 또는 이어 학습 여부와 checkpoint file name
        self.TRAIN_SETTING = {
            "train_mode": ["fresh", "resume"][0],
            "ckp_name": "segmodel.pt",
        }

        # convolution block 공통 설정
        self.CONV_KERNEL_SIZE = 3
        self.CONV_STRIDE = 1
        self.CONV_PADDING = 1

        # stem block 설정
        self.NUM_STEM_CONV_BLOCKS = 2
        self.STEM_OUT_CHANNELS = 32

        # encoder block 설정
        self.ENCODER_OUT_CHANNELS = [self.STEM_OUT_CHANNELS, 64, 128, 256]
        self.NUM_ENCODER_BLOCKS = len(self.ENCODER_OUT_CHANNELS)

        # encoder downsampling max pooling 설정
        self.MAX_POOL_KERNEL_SIZE = 2
        self.MAX_POOL_STRIDE = 2

        # bottleneck block 설정
        self.BOTTLENECK_OUT_CHANNELS = self.ENCODER_OUT_CHANNELS[-1]
        self.NUM_BOTTLENECK_CONV_BLOCKS = 2

        # decoder block 설정
        self.DECODER_OUT_CHANNELS = self.ENCODER_OUT_CHANNELS[::-1]
        self.NUM_DECODER_BLOCKS = len(self.DECODER_OUT_CHANNELS)

        # segmentation output head 설정
        self.OUTPUT_HEAD_HIDDEN_CHANNELS = self.DECODER_OUT_CHANNELS[-1]

        # loss weight 설정
        self.BACKGROUND_LOSS_WEIGHT = 0.2
        self.DICE_LOSS_WEIGHT = 1.0

        # 학습 설정
        self.BATCH_SIZE = 32
        self.NUM_EPOCHS = 50
        self.LEARNING_RATE = 1e-3
        self.NUM_WORKERS = 0

        """==============================================================
        ## 세그멘테이션 모델 추론 설정
        =============================================================="""
        # webcam index와 추론 checkpoint file name
        self.CAMERA_INDEX = 0
        self.INFERENCE_CKPT_NAME = "segmodel.pt"
        self.INFERENCE_PROB_THRESHOLD = 0.25

    """==============================================================
    ## 함수: 클래스 이름과 라벨 매핑 읽기
    =============================================================="""

    def get_class_names_and_labels(self):
        class_name_to_label = {}
        for fg_dir_name in os.listdir(self.REFERENCE_FG_DIR):
            label_string, class_name = fg_dir_name.split("_", 1)
            class_name_to_label[class_name] = int(label_string)
        return class_name_to_label
