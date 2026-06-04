"""==============================================================
# 필요한 라이브러리 불러오기
=============================================================="""
import os

import torch
from torch import nn
from torch.nn import functional as F
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
# 세그멘테이션 모델 생성과 checkpoint 불러오기
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
# train과 같은 class weight를 사용해 pixel-wise loss 계산
class_weights = torch.ones(size=(num_classes,), dtype=torch.float32)
background_label = config.class_name_to_label["background"]
class_weights[background_label] = config.BACKGROUND_LOSS_WEIGHT
class_weights = class_weights.to(device=device)
cross_entropy_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

"""==============================================================
# 모델 테스트
=============================================================="""
# 모델을 평가 모드로 전환
model.eval()

# test metric 누적 변수 초기화
test_loss_sum = 0.0
test_data_count = 0
confusion_matrix = torch.zeros(size=(num_classes, num_classes),
                               dtype=torch.long,
                               device=device)

# test progress bar 생성
test_progress = tqdm(test_loader, desc="Test")

# gradient 계산 없이 test batch loop 실행
with torch.no_grad():
    for images, masks in test_progress:
        # batch data를 평가 장치로 이동
        images = images.to(device=device)
        masks = masks.to(device=device)

        # pixel logits와 loss 계산
        mask_logits = model(images)
        ce_loss = cross_entropy_loss_fn(input=mask_logits, target=masks)

        predicted_mask = F.softmax(input=mask_logits, dim=1)
        masks_one_hot = F.one_hot(input=masks, num_classes=num_classes)
        masks_one_hot = masks_one_hot.permute(0, 3, 1, 2).float()

        intersection = (predicted_mask * masks_one_hot).sum(dim=(0, 2, 3))
        predicted_area = predicted_mask.sum(dim=(0, 2, 3))
        target_area = masks_one_hot.sum(dim=(0, 2, 3))
        dice_score = (2.0 * intersection + 1.0) / \
            (predicted_area + target_area + 1.0)

        # 0번 background는 제외하고 1번부터 foreground class만 반영
        dice_loss = (1.0 - dice_score[1:]).mean()

        loss = ce_loss + config.DICE_LOSS_WEIGHT * dice_loss
        predictions = torch.argmax(input=mask_logits, dim=1)

        # 전체 test loss 누적
        batch_size = images.size(0)
        test_loss_sum = test_loss_sum + loss.item() * batch_size
        test_data_count = test_data_count + batch_size

        # pixel 단위 confusion matrix 누적
        indices = masks.reshape(-1) * num_classes + predictions.reshape(-1)
        batch_matrix = torch.bincount(input=indices,
                                      minlength=num_classes * num_classes)
        batch_matrix = batch_matrix.reshape(num_classes, num_classes)
        confusion_matrix = confusion_matrix + batch_matrix

        # 진행바에 현재 batch loss 표시
        test_progress.set_postfix(loss=f"{loss.item():.4f}")

"""==============================================================
# 테스트 결과 계산
=============================================================="""
# 전체 평균 loss 계산
test_loss = test_loss_sum / test_data_count

# class별 precision, recall, f1, IoU 계산
true_positive = torch.diag(input=confusion_matrix).float()
actual_pixels = confusion_matrix.sum(dim=1).float()
predicted_pixels = confusion_matrix.sum(dim=0).float()
total_pixels = confusion_matrix.sum().float()

precision_per_class = true_positive / predicted_pixels.clamp(min=1)
recall_per_class = true_positive / actual_pixels.clamp(min=1)
f1_per_class = (2 * precision_per_class * recall_per_class) / \
    (precision_per_class + recall_per_class).clamp(min=1e-12)
iou_per_class = true_positive / \
    (actual_pixels + predicted_pixels - true_positive).clamp(min=1)

# 전체 pixel accuracy와 macro metric 계산
pixel_accuracy = true_positive.sum() / total_pixels.clamp(min=1)
present_classes = actual_pixels > 0
mean_iou = iou_per_class[present_classes].mean()
macro_precision = precision_per_class[present_classes].mean()
macro_recall = recall_per_class[present_classes].mean()
macro_f1 = f1_per_class[present_classes].mean()

"""==============================================================
# 테스트 결과 표시
=============================================================="""
# test loss, pixel accuracy, 세그멘테이션 metric 출력
tqdm.write(f"Test Loss: {test_loss:.4f} | "
           f"Pixel Acc: {pixel_accuracy.item():.4f} | "
           f"Mean IoU: {mean_iou.item():.4f} | "
           f"Macro Precision: {macro_precision.item():.4f} | "
           f"Macro Recall: {macro_recall.item():.4f} | "
           f"Macro F1: {macro_f1.item():.4f}")

# class별 IoU 출력
label_to_class_name = {label: class_name
                       for class_name, label in config.class_name_to_label.items()}
for class_idx in range(num_classes):
    class_name = label_to_class_name.get(class_idx, f"class_{class_idx}")
    tqdm.write(f"{class_name}: IoU {iou_per_class[class_idx].item():.4f}")
