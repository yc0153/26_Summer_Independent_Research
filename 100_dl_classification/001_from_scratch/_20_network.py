"""==============================================================
# 라이브러리 불러오기
=============================================================="""
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
    conv_block = nn.Sequential(
        nn.Conv2d(
            in_channels=num_ch_in,
            out_channels=num_ch_out,
            kernel_size=k,
            stride=s,
            padding=p,
        ),
        nn.LeakyReLU(
            negative_slope=0.2, inplace=True
        ),
    )
    return conv_block


def make_fc_block(num_node_in, num_node_out, is_last_block):
    """==============================================================
    ## Fully Connected Layer 블록 정의
    =============================================================="""
    if is_last_block is False:
        fc_block = nn.Sequential(
            nn.Linear(in_features=num_node_in,
                      out_features=num_node_out),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
    else:
        fc_block = nn.Sequential(
            nn.Linear(in_features=num_node_in,
                      out_features=num_node_out),
        )
    return fc_block


class CNN_Model(nn.Module):
    def __init__(self, img_ch):
        """==============================================================
        ### 준비작업
        =============================================================="""
        # 상속된 nn.Module의 생성자 실행
        super().__init__()

        # 입력 이미지 구조 파악
        img_ch = img_ch
        img_height = config.MODEL_IMAGE_HEIGHT
        img_width = config.MODEL_IMAGE_WIDTH

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
        conv_hidden_channels = config.NUM_CONV_HIDDEN_CHANNELS
        if isinstance(conv_hidden_channels, int):
            conv_hidden_channels = [conv_hidden_channels] * config.NUM_CONV_BLOCKS
        elif len(conv_hidden_channels) != config.NUM_CONV_BLOCKS:
            raise ValueError(
                "NUM_CONV_HIDDEN_CHANNELS의 길이는 NUM_CONV_BLOCKS와 같아야 합니다."
            )

        # 합성곱 블록 조립 루프
        for idx, num_ch_out in enumerate(conv_hidden_channels):
            # 합성곱 블록 추가
            layers.append(make_conv_block(num_ch_in=ch_in,
                                          num_ch_out=num_ch_out,
                                          k=config.CONV_KERNEL_SIZE,
                                          s=config.CONV_STRIDE,
                                          p=config.CONV_PADDING))

            # Max Pooling 레이어 추가 및 출력되는 Feature Map 크기 계산
            layers.append(nn.MaxPool2d(kernel_size=config.MAX_POOL_KERNEL_SIZE,
                                       stride=config.MAX_POOL_STRIDE))
            feature_map_height = ((feature_map_height -
                                  config.MAX_POOL_KERNEL_SIZE) //
                                  config.MAX_POOL_STRIDE) + 1
            feature_map_width = ((feature_map_width -
                                 config.MAX_POOL_KERNEL_SIZE) //
                                 config.MAX_POOL_STRIDE) + 1

            feature_map_sizes.append(
                [idx, feature_map_height, feature_map_width])

            # 다음 Conv Block의 입력 채널 수 업데이트
            ch_in = num_ch_out

        # 마지막 합성곱 블록 통과한 피쳐맵 픽셀 수 계산
        num_pixels_last_feature_map = feature_map_sizes[-1][1] * \
            feature_map_sizes[-1][2] * conv_hidden_channels[-1]

        # 마지막 합성곱 블록 통과한 피처맵을 1차원 벡터로 펼치기
        layers.append(nn.Flatten())


        # 완전 연결 블록 조립을 위한 준비작업
        num_node_in = num_pixels_last_feature_map  # 마지막 합성곱 블록 통과한 피쳐맵 픽셀 수로, 최초 입력 노드 수 초기화

        # 완전 연결 블록 조립 루프
        for conv_block_idx in range(config.NUM_FC_BLOCKS):
            # 마지막 FC Layer인지 여부 확인
            is_last_block = (conv_block_idx == config.NUM_FC_BLOCKS - 1)

            # 마지막 블록이 아니면 추가되는 FC 레이어의 출력 노드 수는 config.NUM_FC_HIDDEN_NODES로 설정
            if is_last_block is False:
                num_node_out = config.NUM_FC_HIDDEN_NODES
            # 마지막 블록이면 추가되는 FC 레이어의 출력 노드 수는 클래스 수로 설정
            else:
                num_node_out = len(config.class_name_to_label)

            # 완전 연결 블록 추가
            layers.append(make_fc_block(num_node_in=num_node_in,
                                        num_node_out=num_node_out,
                                        is_last_block=is_last_block))

            # 다음 FC Block의 입력 노드 수 업데이트
            num_node_in = num_node_out

        # 완성된 레이어 리스트로 nn.Sequential 객체 생성하여 self.network에 할당
        self.network = nn.Sequential(*layers)

    def forward(self, images):
        """==============================================================
        ### Forward Propagation 정의
        =============================================================="""
        logits = self.network(images)
        return logits
