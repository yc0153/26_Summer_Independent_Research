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
# 기본 설정
=============================================================="""

config = Config()
class_name_to_label = config.class_name_to_label
num_classes = len(class_name_to_label)

# GPU가 있으면 cuda, 없으면 cpu 사용
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = os.path.join(config.CHECKPOINT_DIR, config.CHECKPOINT_NAME)

"""==============================================================
# test Dataset과 DataLoader 만들기
=============================================================="""
# test dataset 생성
test_dataset = DataSet(split_name="test")

# image channel 수 확인
img_ch = int(test_dataset[0][0].shape[0])

# mini-batch 평가를 위한 DataLoader 생성
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=config.BATCH_SIZE,
                         shuffle=False,
                         num_workers=config.NUM_WORKERS)

"""==============================================================
# model과 checkpoint 불러오기
=============================================================="""
# object detection model 생성
model = CNN_Model(img_ch=img_ch)
model = model.to(device=device)

# 저장된 checkpoint에서 model weight 불러오기
checkpoint = torch.load(f=checkpoint_path, map_location=device)
model.load_state_dict(state_dict=checkpoint["model_state_dict"])

# test loss 계산을 위한 loss function
class_loss_fn = nn.CrossEntropyLoss()
bbox_loss_fn = nn.L1Loss()

"""==============================================================
# 함수: bbox 좌표 순서 정리
=============================================================="""


def order_bboxes(bboxes):
    # 예측값이 x_min/x_max 순서로 나오지 않아도 IoU 계산이 가능하게 정렬
    x_min = torch.minimum(bboxes[:, 0], bboxes[:, 2])
    y_min = torch.minimum(bboxes[:, 1], bboxes[:, 3])
    x_max = torch.maximum(bboxes[:, 0], bboxes[:, 2])
    y_max = torch.maximum(bboxes[:, 1], bboxes[:, 3])

    return torch.stack([x_min, y_min, x_max, y_max], dim=1)

"""==============================================================
# 함수: bbox IoU 계산
=============================================================="""


def bbox_iou(bboxes_a, bboxes_b):
    # 두 bbox set의 좌표 순서 정리
    bboxes_a = order_bboxes(bboxes_a)
    bboxes_b = order_bboxes(bboxes_b)

    # intersection 영역 좌표 계산
    inter_x_min = torch.maximum(bboxes_a[:, 0], bboxes_b[:, 0])
    inter_y_min = torch.maximum(bboxes_a[:, 1], bboxes_b[:, 1])
    inter_x_max = torch.minimum(bboxes_a[:, 2], bboxes_b[:, 2])
    inter_y_max = torch.minimum(bboxes_a[:, 3], bboxes_b[:, 3])

    # intersection area 계산
    inter_w = (inter_x_max - inter_x_min).clamp(min=0.0)
    inter_h = (inter_y_max - inter_y_min).clamp(min=0.0)
    inter_area = inter_w * inter_h

    # 각 bbox area와 union area 계산
    area_a = (bboxes_a[:, 2] - bboxes_a[:, 0]) * \
        (bboxes_a[:, 3] - bboxes_a[:, 1])
    area_b = (bboxes_b[:, 2] - bboxes_b[:, 0]) * \
        (bboxes_b[:, 3] - bboxes_b[:, 1])
    union_area = area_a + area_b - inter_area

    # IoU = intersection / union
    return inter_area / union_area

"""==============================================================
# test 평가 준비
=============================================================="""

model.eval()

test_loss_sum = 0.0
test_data_count = 0
correct_class_count = 0
object_iou_sum = 0.0
object_iou_count = 0
confusion_matrix = torch.zeros(size=(num_classes, num_classes),
                               dtype=torch.long,
                               device=device)

"""==============================================================
# test set 평가
=============================================================="""

with torch.no_grad():
    test_progress = tqdm(test_loader, desc="Test")

    for images, labels, bboxes in test_progress:
        # mini-batch를 device로 이동
        images = images.to(device=device)
        labels = labels.to(device=device)
        bboxes = bboxes.to(device=device)

        # forward
        class_logits, bbox_pred = model(images)

        # classification loss와 object sample의 bbox loss 계산
        class_loss = class_loss_fn(input=class_logits, target=labels)
        object_mask = labels != class_name_to_label["background"]
        bbox_loss = bbox_loss_fn(input=bbox_pred[object_mask],
                                 target=bboxes[object_mask])
        loss = class_loss + config.BBOX_LOSS_WEIGHT * bbox_loss

        # class prediction과 object sample의 IoU 계산
        predictions = torch.argmax(input=class_logits, dim=1)
        object_ious = bbox_iou(bbox_pred[object_mask],
                               bboxes[object_mask])

        # loss, accuracy, IoU 누적
        batch_size = labels.size(0)
        test_loss_sum = test_loss_sum + loss.item() * batch_size
        test_data_count = test_data_count + batch_size
        correct_class_count = correct_class_count + \
            (predictions == labels).sum().item()
        object_iou_sum = object_iou_sum + object_ious.sum().item()
        object_iou_count = object_iou_count + int(object_mask.sum().item())

        # confusion matrix 누적
        indices = labels * num_classes + predictions
        batch_matrix = torch.bincount(input=indices,
                                      minlength=num_classes * num_classes)
        batch_matrix = batch_matrix.reshape(num_classes, num_classes)
        confusion_matrix = confusion_matrix + batch_matrix

        # progress bar에 현재까지의 평균 metric 표시
        running_acc = correct_class_count / test_data_count
        running_iou = object_iou_sum / object_iou_count
        test_progress.set_postfix(loss=f"{loss.item():.4f}",
                                  acc=f"{running_acc:.4f}",
                                  obj_iou=f"{running_iou:.4f}")

"""==============================================================
# 최종 metric 계산
=============================================================="""

# 전체 test loss, class accuracy, object mean IoU 계산
test_loss = test_loss_sum / test_data_count
class_accuracy = correct_class_count / test_data_count
object_mean_iou = object_iou_sum / object_iou_count

# confusion matrix에서 class별 precision, recall, F1 계산
true_positive = torch.diag(input=confusion_matrix).float()
actual_count = confusion_matrix.sum(dim=1).float()
predicted_count = confusion_matrix.sum(dim=0).float()

precision_per_class = true_positive / predicted_count
recall_per_class = true_positive / actual_count
f1_per_class = (2 * precision_per_class * recall_per_class) / \
    (precision_per_class + recall_per_class)

macro_precision = precision_per_class.mean()
macro_recall = recall_per_class.mean()
macro_f1 = f1_per_class.mean()

"""==============================================================
# 평가 결과 출력
=============================================================="""

tqdm.write(f"Test Loss: {test_loss:.4f} | "
           f"Class Accuracy: {class_accuracy:.4f} | "
           f"Object Mean IoU: {object_mean_iou:.4f} | "
           f"Macro Precision: {macro_precision.item():.4f} | "
           f"Macro Recall: {macro_recall.item():.4f} | "
           f"Macro F1: {macro_f1.item():.4f}")

label_to_class_name = {label: class_name
                       for class_name, label in class_name_to_label.items()}

# class별 F1 출력
for class_idx in range(num_classes):
    class_name = label_to_class_name[class_idx]
    tqdm.write(f"{class_name}: F1 {f1_per_class[class_idx].item():.4f}")
