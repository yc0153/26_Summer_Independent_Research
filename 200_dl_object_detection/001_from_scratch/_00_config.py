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

        # 생성 이미지 확장자
        self.IMAGE_EXTENSIONS = ".jpg"

        # train/valid/test 합성 이미지 생성 개수
        self.TRAIN_IMAGE_COUNT = 700
        self.VALID_IMAGE_COUNT = 150
        self.TEST_IMAGE_COUNT = 150

        # 기존 생성 데이터 삭제 후 다시 생성할지 여부
        self.RECREATE_DATASET = True

        """==============================================================
        ## 디텍션 모델 학습 설정
        =============================================================="""
        # 새 학습 또는 이어 학습 여부와 checkpoint file name
        self.TRAIN_SETTING = {
            "train_mode": ["fresh", "resume"][0],
            "ckp_name": "detector_1.pt",
        }

        # CNN feature extractor 설정
        self.NUM_CONV_BLOCKS = 5
        self.NUM_CONV_HIDDEN_CHANNELS = 32
        self.CONV_KERNEL_SIZE = 3
        self.CONV_STRIDE = 2
        self.CONV_PADDING = 1

        # class head 설정
        self.NUM_CLASS_FC_BLOCKS = 2
        self.NUM_CLASS_FC_HIDDEN_NODES = 64

        # bbox head 설정
        self.NUM_BBOX_FC_BLOCKS = 2
        self.NUM_BBOX_FC_HIDDEN_NODES = 64

        # loss와 optimizer hyperparameter 설정
        self.BBOX_LOSS_WEIGHT = 10.0
        self.BATCH_SIZE = 32
        self.NUM_EPOCHS = 50
        self.LEARNING_RATE = 1e-3
        self.NUM_WORKERS = 0

        """==============================================================
        ## 디텍션 모델 추론 설정
        =============================================================="""
        # webcam index와 추론 checkpoint file name
        self.CAMERA_INDEX = 0
        self.INFERENCE_CKPT_NAME = "detector_1.pt"

    """==============================================================
    ## 함수: 클래스 이름과 라벨 매핑 읽기
    =============================================================="""

    def get_class_names_and_labels(self):
        # reference_fg 폴더가 아직 없으면 빈 mapping 반환
        if not os.path.isdir(self.REFERENCE_FG_DIR):
            return {}

        # "1_mouse" 형식의 폴더 이름에서 label과 class name 추출
        class_name_to_label = {}
        for fg_dir_name in os.listdir(self.REFERENCE_FG_DIR):
            if "_" not in fg_dir_name:
                continue

            label_string, class_name = fg_dir_name.split("_", 1)
            class_name_to_label[class_name] = int(label_string)

        return class_name_to_label
