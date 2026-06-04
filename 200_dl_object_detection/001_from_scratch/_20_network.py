"""==============================================================
# 라이브러리 불러오기
=============================================================="""
import torch
from torch import nn

from _00_config import Config

"""==============================================================
# 기본설정 불러오기
=============================================================="""
config = Config()


def make_conv_block(num_ch_in, num_ch_out, k, s, p):
    """==============================================================
    ## Convolutional Layer 블록 정의
    =============================================================="""
    # 합성곱 레이어와 활성화 함수 추가
    conv_block = nn.Sequential(
        nn.Conv2d(
            in_channels=num_ch_in,
            out_channels=num_ch_out,
            kernel_size=k,
            stride=s,
            padding=p,
        ),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
    )

    # 완성된 nn.Sequential 객체 반환
    return conv_block


def make_fc_block(num_node_in, num_node_out, is_last_block):
    """==============================================================
    ## Fully Connected Layer 블록 정의
    =============================================================="""
    # FC Layer 추가
    layers = [
        nn.Linear(in_features=num_node_in, out_features=num_node_out),
    ]

    # 마지막 블록이 아니면 활성화 함수 추가
    if is_last_block is False:
        layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

    # 완성된 레이어 리스트로 nn.Sequential 객체 생성하여 반환
    return nn.Sequential(*layers)


def make_fc_head(num_node_in, num_hidden_nodes, num_node_out, num_fc_blocks):
    """==============================================================
    ## Fully Connected Head 정의
    =============================================================="""
    # head 레이어 컨테이너 생성
    head_layers = []

    # head 전용 완전 연결 블록 조립 루프
    for fc_block_idx in range(num_fc_blocks):
        # 마지막 FC Layer인지 여부 확인
        is_last_block = (fc_block_idx == num_fc_blocks - 1)

        # 마지막 블록이 아니면 hidden node 수로 설정
        if is_last_block is False:
            num_node_out_current = num_hidden_nodes
        # 마지막 블록이면 head 출력 node 수로 설정
        else:
            num_node_out_current = num_node_out

        # 완전 연결 블록 추가
        head_layers.append(make_fc_block(num_node_in=num_node_in,
                                         num_node_out=num_node_out_current,
                                         is_last_block=is_last_block))

        # 다음 FC Block의 입력 노드 수 업데이트
        num_node_in = num_node_out_current

    # 완성된 head 레이어 리스트로 nn.Sequential 객체 생성하여 반환
    return nn.Sequential(*head_layers)


class CNN_Model(nn.Module):
    def __init__(self, img_ch):
        """==============================================================
        ### 준비작업
        =============================================================="""
        # 상속된 nn.Module의 생성자 실행
        super().__init__()

        # 입력 이미지 구조 파악
        img_height = config.MODEL_IMAGE_HEIGHT
        img_width = config.MODEL_IMAGE_WIDTH

        # 디텍션 출력 구성을 위한 클래스 개수 확인
        num_classes = len(config.class_name_to_label)

        """==============================================================
        ### 아키텍쳐 구성
        =============================================================="""
        # 레이어 컨테이너 생성
        layers = []

        # 합성곱 블록 조립을 위한 준비 작업
        ch_in = img_ch  # 이미지 채널 수로, 최초 입력 채널 수 초기화
        feature_map_height = img_height  # 이미지 높이로 최초 피쳐맵 높이 초기화
        feature_map_width = img_width  # 이미지 너비로 최초 피쳐맵 너비 초기화
        feature_map_sizes = []  # 피쳐맵 크기 저장 리스트 초기화

        # 합성곱 블록 조립 루프
        for idx in range(config.NUM_CONV_BLOCKS):
            # 합성곱 블록 추가
            layers.append(make_conv_block(num_ch_in=ch_in,
                                          num_ch_out=config.NUM_CONV_HIDDEN_CHANNELS,
                                          k=config.CONV_KERNEL_SIZE,
                                          s=config.CONV_STRIDE,
                                          p=config.CONV_PADDING))

            # 추가된 합성곱 블록을 지났을 때 출력되는 Feature Map의 크기 계산
            feature_map_height = ((feature_map_height + 2 * config.CONV_PADDING -
                                  config.CONV_KERNEL_SIZE) // config.CONV_STRIDE) + 1
            feature_map_width = ((feature_map_width + 2 * config.CONV_PADDING -
                                 config.CONV_KERNEL_SIZE) // config.CONV_STRIDE) + 1
            feature_map_sizes.append([idx, feature_map_height, feature_map_width])

            # 다음 Conv Block의 입력 채널 수 업데이트
            ch_in = config.NUM_CONV_HIDDEN_CHANNELS

        # 마지막 합성곱 블록 통과한 피쳐맵 픽셀 수 계산
        num_pixels_last_feature_map = feature_map_sizes[-1][1] * \
            feature_map_sizes[-1][2] * config.NUM_CONV_HIDDEN_CHANNELS

        # 마지막 합성곱 블록 통과한 피처맵을 1차원 벡터로 펼치기
        layers.append(nn.Flatten())

        # 완성된 레이어 리스트로 nn.Sequential 객체 생성하여 self.feature_extractor에 할당
        self.feature_extractor = nn.Sequential(*layers)

        # 클래스 분류와 bbox 예측을 위한 head 분리
        self.class_head = make_fc_head(num_node_in=num_pixels_last_feature_map,
                                       num_hidden_nodes=config.NUM_CLASS_FC_HIDDEN_NODES,
                                       num_node_out=num_classes,
                                       num_fc_blocks=config.NUM_CLASS_FC_BLOCKS)
        self.bbox_head = make_fc_head(num_node_in=num_pixels_last_feature_map,
                                      num_hidden_nodes=config.NUM_BBOX_FC_HIDDEN_NODES,
                                      num_node_out=num_classes * 4,
                                      num_fc_blocks=config.NUM_BBOX_FC_BLOCKS)

        # bbox head 출력을 클래스별 bbox로 나누기 위한 클래스 개수 저장
        self.num_classes = num_classes

    def forward(self, images):
        """==============================================================
        ### Forward Propagation 정의
        =============================================================="""
        # 네트워크에 입력 이미지 통과시켜 공통 특징 추출
        features = self.feature_extractor(images)

        # class head로 class logits 계산
        class_logits = self.class_head(features)

        # bbox head로 클래스별 bbox 좌표 예측값 계산, sigmoid 함수로 0~1 사이 값으로 변환
        bbox_pred = torch.sigmoid(self.bbox_head(features))
        bbox_pred = bbox_pred.view(-1, self.num_classes, 4)

        # class logits와 클래스별 bbox 예측값 함께 반환
        return class_logits, bbox_pred
