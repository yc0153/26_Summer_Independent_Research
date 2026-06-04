"""==============================================================
# 라이브러리 불러오기
=============================================================="""
import torch
from torch import nn
from torch.nn import functional as F

from _00_config import Config

"""==============================================================
# 기본설정 불러오기
=============================================================="""
config = Config()


def make_conv_block(ch_in, ch_out, k, s, p):
    """==============================================================
    ## Convolutional Layer 블록 정의
    =============================================================="""
    # 합성곱 레이어, 활성화 함수 추가
    conv_block = nn.Sequential(
        nn.Conv2d(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=k,
            stride=s,
            padding=p,
            bias=True,
        ),
        nn.ReLU(inplace=True),
    )

    # 완성된 nn.Sequential 객체 반환
    return conv_block


class CNN_Model(nn.Module):
    def __init__(self, img_ch):
        """==============================================================
        ### 준비작업
        =============================================================="""
        # 상속된 nn.Module의 생성자 실행
        super().__init__()

        # 입력 이미지 구조와 segmentation class 개수 파악
        num_classes = len(config.class_name_to_label)
        stem_ch = config.STEM_OUT_CHANNELS
        encoder_channels = config.ENCODER_OUT_CHANNELS
        bottleneck_ch = config.BOTTLENECK_OUT_CHANNELS
        decoder_channels = config.DECODER_OUT_CHANNELS

        """==============================================================
        ### 아키텍쳐 구성
        =============================================================="""
        # 원본 해상도의 feature를 만드는 stem block
        stem_layers = []
        ch_in = img_ch
        for _ in range(config.NUM_STEM_CONV_BLOCKS):
            stem_layers.append(make_conv_block(ch_in=ch_in,
                                               ch_out=stem_ch,
                                               k=config.CONV_KERNEL_SIZE,
                                               s=config.CONV_STRIDE,
                                               p=config.CONV_PADDING))
            ch_in = stem_ch
        self.stem = nn.Sequential(*stem_layers)

        # downsampling encoder block 생성
        encoder_layers = []
        skip_channels = []
        for encoder_ch in encoder_channels:
            encoder_layers.append(make_conv_block(ch_in=ch_in,
                                                  ch_out=encoder_ch,
                                                  k=config.CONV_KERNEL_SIZE,
                                                  s=config.CONV_STRIDE,
                                                  p=config.CONV_PADDING))
            encoder_layers.append(
                nn.MaxPool2d(kernel_size=config.MAX_POOL_KERNEL_SIZE,
                             stride=config.MAX_POOL_STRIDE)
            )
            ch_in = encoder_ch
            skip_channels.append(ch_in)
        self.encoder_blocks = nn.Sequential(*encoder_layers)

        # 가장 작은 bottleneck block
        bottleneck_layers = []
        for _ in range(config.NUM_BOTTLENECK_CONV_BLOCKS):
            bottleneck_layers.append(make_conv_block(ch_in=ch_in,
                                                     ch_out=bottleneck_ch,
                                                     k=config.CONV_KERNEL_SIZE,
                                                     s=config.CONV_STRIDE,
                                                     p=config.CONV_PADDING))
            ch_in = bottleneck_ch
        self.bottleneck = nn.Sequential(*bottleneck_layers)

        # upsampling decoder block 생성
        decoder_layers = []
        for decoder_ch, skip_ch in zip(decoder_channels,
                                       reversed(skip_channels)):
            decoder_layers.append(
                make_conv_block(ch_in=ch_in + skip_ch,
                                ch_out=decoder_ch,
                                k=config.CONV_KERNEL_SIZE,
                                s=config.CONV_STRIDE,
                                p=config.CONV_PADDING)
            )
            ch_in = decoder_ch
        self.decoder_blocks = nn.Sequential(*decoder_layers)

        # 픽셀마다 class logits를 예측하는 segmentation head 정의
        self.segmentation_head = nn.Sequential(
            make_conv_block(ch_in=ch_in,
                            ch_out=config.OUTPUT_HEAD_HIDDEN_CHANNELS,
                            k=config.CONV_KERNEL_SIZE,
                            s=config.CONV_STRIDE,
                            p=config.CONV_PADDING),
            nn.Conv2d(in_channels=config.OUTPUT_HEAD_HIDDEN_CHANNELS,
                      out_channels=num_classes,
                      kernel_size=1,
                      stride=1,
                      padding=0),
        )

    def forward(self, images):
        """==============================================================
        ### Forward Propagation 정의
        =============================================================="""
        # 원본 해상도 feature 생성
        features = self.stem(images)
        skip_features = []

        # encoder를 통과하며 skip connection용 feature 저장
        for i in range(0, len(self.encoder_blocks), 2):
            features = self.encoder_blocks[i](features)
            skip_features.append(features)
            features = self.encoder_blocks[i + 1](features)

        # bottleneck에서 가장 작은 feature map 보강
        features = self.bottleneck(features)

        # decoder에서 feature를 키우며 skip feature와 결합
        for i in range(len(self.decoder_blocks)):
            skip_feature = skip_features[-i - 1]
            features = F.interpolate(input=features,
                                     size=skip_feature.shape[-2:],
                                     mode="bilinear",
                                     align_corners=False)
            features = torch.cat([features, skip_feature], dim=1)
            features = self.decoder_blocks[i](features)

        # full resolution feature에서 pixel logits 계산
        mask_logits = self.segmentation_head(features)

        return mask_logits
