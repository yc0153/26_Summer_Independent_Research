"""==============================================================
# 필요한 라이브러리 불러오기
=============================================================="""
# 표준 라이브러리 모듈
import os  # 운영체제 관련기능

"""==============================================================
# 설정값 클래스 정의
=============================================================="""


class Config:
    def __init__(self):
        """==============================================================
        ## 일반 설정
        =============================================================="""
        self.RANDOM_SEED = 10

        """==============================================================
        ## 경로 설정
        =============================================================="""
        # 프로젝트 루트 디렉토리
        self.ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        # 입력 이미지 데이터 디렉토리
        self.REFERENCE_BG_DIR = os.path.join(
            self.ROOT_DIR, "data", "reference_bg")
        self.REFERENCE_FG_DIR = os.path.join(
            self.ROOT_DIR, "data", "reference_fg")

        self.TRAIN_DIR = os.path.join(self.ROOT_DIR, "data", "train")
        self.VALID_DIR = os.path.join(self.ROOT_DIR, "data", "valid")
        self.TEST_DIR = os.path.join(self.ROOT_DIR, "data", "test")

        # 입력 이미지 데이터 라벨 파일 경로
        self.TRAIN_LABEL_PATH = os.path.join(self.TRAIN_DIR, "labels.json")
        self.VALID_LABEL_PATH = os.path.join(self.VALID_DIR, "labels.json")
        self.TEST_LABEL_PATH = os.path.join(self.TEST_DIR, "labels.json")

        # 모델 체크포인트 저장 디렉토리
        self.CHECKPOINT_DIR = os.path.join(self.ROOT_DIR, "checkpoint")

        """==============================================================
        ## 클래스 설정
        =============================================================="""
        # 클래스 이름과 레이블 매핑
        self.class_name_to_label = self.get_class_names_and_labels()

        """==============================================================
        ## 입력 이미지 설정
        =============================================================="""
        # 입력 이미지 크기
        self.MODEL_IMAGE_HEIGHT = 240
        self.MODEL_IMAGE_WIDTH = 320

        # 입력 이미지 확장자 (생성될 이미지의 확장자)
        self.IMAGE_EXTENSIONS = ".jpg"

        # 훈련/검증/테스트 이미지 생성 개수
        self.TRAIN_IMAGE_COUNT = 700
        self.VALID_IMAGE_COUNT = 150
        self.TEST_IMAGE_COUNT = 150

        # 이미지 재생성 시, 기존 이미지 삭제 여부
        self.RECREATE_DATASET = True

        # 배경 이미지 비율
        self.BG_SCALE_RATIO = 0.2

        # 전경 이미지 크기 비율
        self.FG_SCALE_RATIO_MIN = 0.25
        self.FG_SCALE_RATIO_MAX = 1.0

        """==============================================================
        ## 딥 뉴럴 네트워크 학습 설정
        =============================================================="""
        # 학습 이어하기 여부 및 체크포인트 파일명 설정
        self.TRAIN_SETTING = {
            "train_mode": ["fresh", "resume"][0],
            "ckp_name": "classmodel.pt",
        }

        # 뉴럴 네트워크 아키텍쳐 설정: Feature Extractor
        self.NUM_CONV_HIDDEN_CHANNELS = [8, 16, 32, 64, 128]
        self.NUM_CONV_BLOCKS = len(self.NUM_CONV_HIDDEN_CHANNELS)
        self.CONV_KERNEL_SIZE = 3
        self.CONV_STRIDE = 1
        self.CONV_PADDING = 1

        # 뉴럴 네트워크 아키텍쳐 설정: Feature Extractor Pooling
        self.MAX_POOL_KERNEL_SIZE = 2
        self.MAX_POOL_STRIDE = 2

        # 뉴럴 네트워크 아키텍쳐 설정: Header
        self.NUM_FC_BLOCKS = 2
        self.NUM_FC_HIDDEN_NODES = 32

        # 학습 하이퍼파라미터 설정
        self.BATCH_SIZE = 32
        self.NUM_EPOCHS = 50
        self.LEARNING_RATE = 1e-3
        self.NUM_WORKERS = 0

        """==============================================================
        ## 딥 뉴럴 네트워크 추론 설정
        =============================================================="""
        # 웹캠 번호
        self.CAMERA_INDEX = 0

        # 추론에 사용할 checkpoint 파일명
        self.INFERENCE_CKPT_NAME = "classmodel.pt"

    """==============================================================
    ## 함수: 사용되는 클래스 이름 및 라벨 매핑규칙 가져오기
    =============================================================="""

    def get_class_names_and_labels(self):
        # 전경 이미지 폴더 이름 목록 가져오기
        fg_list_raw = os.listdir(self.REFERENCE_FG_DIR)

        # 클래스 이름 및 라벨 매핑 딕셔너리 생성
        class_name_to_label = {}
        for fg_i_raw in fg_list_raw:
            label_string, class_name = fg_i_raw.split("_")
            class_name_to_label[class_name] = int(label_string)

        # 클래스 이름 및 라벨 매핑 딕셔너리 반환
        return class_name_to_label
