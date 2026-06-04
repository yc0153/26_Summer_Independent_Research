"""==============================================================
# 필요한 라이브러리 불러오기
=============================================================="""
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from _00_config import Config
from _11_dataset import DataSet
from _20_network import CNN_Model

"""==============================================================
# 준비 작업
=============================================================="""
# 기본 설정값과 클래스 개수 확인
config = Config()
num_classes = len(config.class_name_to_label)

# 평가 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 평가에 사용할 checkpoint file path
checkpoint_path = os.path.join(
    config.CHECKPOINT_DIR, config.TRAIN_SETTING["ckp_name"])

"""==============================================================
# 테스트 데이터셋과 dataloader 생성
=============================================================="""
# test dataset 생성
test_dataset = DataSet(split_name="test")

# 첫 번째 이미지 tensor에서 입력 채널 수 확인
first_img_tensor = test_dataset[0][0]
img_ch = int(first_img_tensor.shape[0])

# test dataloader 생성
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=config.BATCH_SIZE,
                         shuffle=False,
                         num_workers=config.NUM_WORKERS)

"""==============================================================
# 디텍션 모델 생성과 checkpoint 불러오기
=============================================================="""
# 모델 생성 후 평가 장치로 이동
model = CNN_Model(img_ch=img_ch)
model = model.to(device=device)

# 저장된 checkpoint의 모델 weight 불러오기
checkpoint = torch.load(f=checkpoint_path, map_location=device)
model.load_state_dict(state_dict=checkpoint["model_state_dict"])

"""==============================================================
# 손실 함수 설정
=============================================================="""
# class prediction loss와 bbox regression loss 설정
class_loss_fn = nn.CrossEntropyLoss()
bbox_loss_fn = nn.SmoothL1Loss()

"""==============================================================
# 바운딩 박스 평가 함수 정의
=============================================================="""


def order_bboxes(bboxes):
    """==============================================================
    ## bbox 좌표 순서 보정
    =============================================================="""
    # x/y min/max가 뒤집혀도 올바른 좌표 순서로 정렬
    x_min = torch.minimum(bboxes[:, 0], bboxes[:, 2])
    y_min = torch.minimum(bboxes[:, 1], bboxes[:, 3])
    x_max = torch.maximum(bboxes[:, 0], bboxes[:, 2])
    y_max = torch.maximum(bboxes[:, 1], bboxes[:, 3])
    return torch.stack([x_min, y_min, x_max, y_max], dim=1)


def bbox_iou(bboxes_a, bboxes_b):
    """==============================================================
    ## bbox IoU 계산
    =============================================================="""
    # 예측 bbox와 정답 bbox를 0~1 범위의 정렬된 좌표로 변환
    bboxes_a = order_bboxes(bboxes_a).clamp(0.0, 1.0)
    bboxes_b = order_bboxes(bboxes_b).clamp(0.0, 1.0)

    # 교집합 영역 좌표 계산
    inter_x_min = torch.maximum(bboxes_a[:, 0], bboxes_b[:, 0])
    inter_y_min = torch.maximum(bboxes_a[:, 1], bboxes_b[:, 1])
    inter_x_max = torch.minimum(bboxes_a[:, 2], bboxes_b[:, 2])
    inter_y_max = torch.minimum(bboxes_a[:, 3], bboxes_b[:, 3])

    # 교집합 면적 계산
    inter_w = (inter_x_max - inter_x_min).clamp(min=0.0)
    inter_h = (inter_y_max - inter_y_min).clamp(min=0.0)
    inter_area = inter_w * inter_h

    # 각 bbox 면적과 합집합 면적 계산
    area_a = ((bboxes_a[:, 2] - bboxes_a[:, 0]).clamp(min=0.0) *
              (bboxes_a[:, 3] - bboxes_a[:, 1]).clamp(min=0.0))
    area_b = ((bboxes_b[:, 2] - bboxes_b[:, 0]).clamp(min=0.0) *
              (bboxes_b[:, 3] - bboxes_b[:, 1]).clamp(min=0.0))
    union_area = area_a + area_b - inter_area

    return inter_area / union_area.clamp(min=1e-7)

"""==============================================================
# 모델 테스트
=============================================================="""
# 모델을 평가 모드로 전환
model.eval()

# test metric 누적 변수 초기화
test_loss_sum = 0.0
test_iou_sum = 0.0
test_data_count = 0
confusion_matrices = torch.zeros(size=(num_classes, 2, 2),
                                 dtype=torch.long,
                                 device=device)

# test progress bar 생성
test_progress = tqdm(test_loader, desc="Test")

# gradient 계산 없이 test batch loop 실행
with torch.no_grad():
    for images, labels, bboxes in test_progress:
        # batch data를 평가 장치로 이동
        images = images.to(device=device)
        labels = labels.to(device=device)
        bboxes = bboxes.to(device=device)

        # class logits와 bbox 예측 후 loss 계산
        class_logits, bbox_pred = model(images)
        class_loss = class_loss_fn(input=class_logits, target=labels)
        batch_indices = torch.arange(labels.size(0), device=device)
        bbox_pred_for_loss = bbox_pred[batch_indices, labels]
        bbox_loss = bbox_loss_fn(input=bbox_pred_for_loss, target=bboxes)
        loss = class_loss + config.BBOX_LOSS_WEIGHT * bbox_loss

        # class prediction과 bbox IoU 계산
        predictions = torch.argmax(input=class_logits, dim=1)
        bbox_pred_for_iou = bbox_pred[batch_indices, predictions]
        ious = bbox_iou(bbox_pred_for_iou, bboxes)

        # 전체 test loss와 IoU 누적
        batch_size = labels.size(0)
        test_loss_sum = test_loss_sum + loss.item() * batch_size
        test_iou_sum = test_iou_sum + ious.sum().item()
        test_data_count = test_data_count + batch_size

        # 클래스별 2x2 confusion matrix 누적
        for class_idx in range(num_classes):
            actual_is_class = (labels == class_idx)
            predicted_is_class = (predictions == class_idx)

            confusion_matrices[class_idx, 1, 1] += \
                (actual_is_class & predicted_is_class).sum()
            confusion_matrices[class_idx, 0, 1] += \
                (~actual_is_class & predicted_is_class).sum()
            confusion_matrices[class_idx, 1, 0] += \
                (actual_is_class & ~predicted_is_class).sum()
            confusion_matrices[class_idx, 0, 0] += \
                (~actual_is_class & ~predicted_is_class).sum()

        # 진행바에 현재 batch loss와 mean IoU 표시
        test_progress.set_postfix(loss=f"{loss.item():.4f}",
                                  miou=f"{ious.mean().item():.4f}")

"""==============================================================
# 테스트 결과 계산
=============================================================="""
# 전체 평균 loss와 mean IoU 계산
test_loss = test_loss_sum / test_data_count
mean_iou = test_iou_sum / test_data_count

# class별 precision, recall, f1 계산을 위한 값 분리
true_positive = confusion_matrices[:, 1, 1].float()
false_positive = confusion_matrices[:, 0, 1].float()
false_negative = confusion_matrices[:, 1, 0].float()

precision_per_class = true_positive / \
    (true_positive + false_positive).clamp(min=1)
recall_per_class = true_positive / \
    (true_positive + false_negative).clamp(min=1)
f1_per_class = (2 * precision_per_class * recall_per_class) / \
    (precision_per_class + recall_per_class).clamp(min=1e-12)

# macro average metric 계산
macro_precision = precision_per_class.mean().item()
macro_recall = recall_per_class.mean().item()
macro_f1 = f1_per_class.mean().item()

"""==============================================================
# 테스트 결과 표시
=============================================================="""
# test loss, bbox IoU, class macro metric 출력
tqdm.write(f"Test Loss: {test_loss:.4f} | "
           f"Mean IoU: {mean_iou:.4f} | "
           f"Macro Precision: {macro_precision:.4f} | "
           f"Macro Recall: {macro_recall:.4f} | "
           f"Macro F1: {macro_f1:.4f}")
