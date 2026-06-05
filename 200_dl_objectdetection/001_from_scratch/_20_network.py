"""==============================================================
# 필요한 라이브러리 불러오기
=============================================================="""
from torch import nn

from _00_config import Config

"""==============================================================
# 기본 설정
=============================================================="""

config = Config()

"""==============================================================
# 함수: convolution block 만들기
=============================================================="""


def make_conv_block(num_ch_in, num_ch_out, k, s, p, pool_k, pool_s):
    # Conv2d, LeakyReLU, MaxPool2d를 묶은 기본 feature extraction block
    conv_block = nn.Sequential(
        nn.Conv2d(in_channels=num_ch_in,
                  out_channels=num_ch_out,
                  kernel_size=k,
                  stride=s,
                  padding=p),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.MaxPool2d(kernel_size=pool_k,
                     stride=pool_s),
    )

    return conv_block

"""==============================================================
# 함수: fully connected block 만들기
=============================================================="""


def make_fc_block(num_node_in, num_node_out):
    # Linear 뒤에 LeakyReLU를 붙인 head용 block
    fc_block = nn.Sequential(
        nn.Linear(in_features=num_node_in,
                  out_features=num_node_out),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
    )

    return fc_block

"""==============================================================
# object detection CNN model 정의
=============================================================="""


class CNN_Model(nn.Module):
    def __init__(self, img_ch):
        super().__init__()

        # 예측해야 할 class 개수
        num_classes = len(config.class_name_to_label)

        """==============================================================
        ## feature extractor 만들기
        =============================================================="""
        layers = []
        ch_in = img_ch
        feature_map_height = config.MODEL_IMAGE_HEIGHT
        feature_map_width = config.MODEL_IMAGE_WIDTH

        # conv block을 반복해서 feature map 크기를 줄임
        for ch_out in config.NUM_CONV_HIDDEN_CHANNELS:
            layers.append(make_conv_block(num_ch_in=ch_in,
                                          num_ch_out=ch_out,
                                          k=config.CONV_KERNEL_SIZE,
                                          s=config.CONV_STRIDE,
                                          p=config.CONV_PADDING,
                                          pool_k=config.MAX_POOL_KERNEL_SIZE,
                                          pool_s=config.MAX_POOL_STRIDE))

            feature_map_height = feature_map_height // 2
            feature_map_width = feature_map_width // 2
            ch_in = ch_out

        # convolution 결과를 flatten했을 때의 feature 개수
        num_features = feature_map_height * feature_map_width * ch_in

        """==============================================================
        ## classification head와 bbox regression head 만들기
        =============================================================="""
        # image에서 공통 feature를 뽑는 CNN backbone
        self.feature_extractor = nn.Sequential(*layers)

        # classification과 bbox prediction이 함께 사용하는 shared head
        head_layers = [nn.Flatten()]
        num_node_in = num_features

        for _ in range(config.NUM_HEAD_HIDDEN_LAYERS):
            head_layers.append(make_fc_block(
                num_node_in=num_node_in,
                num_node_out=config.HEAD_HIDDEN_NODES,
            ))
            num_node_in = config.HEAD_HIDDEN_NODES

        self.shared_head = nn.Sequential(*head_layers)

        # class를 예측하는 head
        self.class_head = nn.Linear(in_features=num_node_in,
                                    out_features=num_classes)

        # bbox 좌표 4개를 예측하는 head
        self.bbox_head = nn.Linear(in_features=num_node_in,
                                   out_features=4)

    def forward(self, images):
        # CNN feature 추출 후 shared head 통과
        features = self.feature_extractor(images)
        features = self.shared_head(features)

        # class logits와 0~1 범위 bbox 좌표 예측
        class_logits = self.class_head(features)
        bbox_pred = self.bbox_head(features).sigmoid()

        return class_logits, bbox_pred
