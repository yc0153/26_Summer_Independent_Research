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
# 기본 설정값 불러오기
config = Config()

# 클래스 수 확인
num_classes = len(config.class_name_to_label)

# 학습 장치 설정 (cpu or gpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 체크포인트 파일 경로
checkpoint_path = os.path.join(
    config.CHECKPOINT_DIR, config.TRAIN_SETTING["ckp_name"])

"""==============================================================
# 테스트를 위한 데이터셋 생성
=============================================================="""
# 테스트 데이터셋
test_dataset = DataSet(split_name="test")

# 입력 이미지 채널 수 확인
first_img_tensor = test_dataset[0][0]
img_ch = int(first_img_tensor.shape[0])

"""==============================================================
# 테스트를 위한 데이터로더 생성
=============================================================="""
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=config.BATCH_SIZE,
                         shuffle=False,
                         num_workers=config.NUM_WORKERS)

"""==============================================================
# 딥 뉴럴 네트워크 모델 생성
=============================================================="""
# 정의된 CNN_Model 클래스를 이용하여 모델 객체 생성
model = CNN_Model(img_ch=img_ch)

# 모델을 설정된 학습장치로 이동
model = model.to(device=device)

"""==============================================================
# 체크포인트 불러오기
=============================================================="""
# 체크포인트 불러오기
checkpoint = torch.load(f=checkpoint_path, map_location=device)

# 모델 웨이트 불러오기
model.load_state_dict(state_dict=checkpoint["model_state_dict"])

"""==============================================================
# 손실함수 설정
=============================================================="""
# 손실함수 설정
loss_fn = nn.CrossEntropyLoss()

"""==============================================================
# 모델 테스트
=============================================================="""
# 모델을 평가 모드로 전환
model.eval()

# 테스트 변수 초기화
test_loss_sum = 0.0
test_data_count = 0
confusion_matrices = torch.zeros(size=(num_classes, 2, 2),
                                 dtype=torch.long,
                                 device=device)

# 테스트 진행률 바 생성
test_progress = tqdm(test_loader, desc="Test")

with torch.no_grad():
    for images, labels in test_progress:
        # 테스트 데이터 준비
        images = images.to(device=device)
        labels = labels.to(device=device)

        # 예측 및 손실 계산
        logits = model(images)
        loss = loss_fn(input=logits, target=labels)
        predictions = torch.argmax(input=logits, dim=1)

        # 테스트 진행 중, 평균 로스 계산을 위한 데이터 누적
        batch_size = labels.size(0)
        test_loss_sum = test_loss_sum + loss.item() * batch_size
        test_data_count = test_data_count + batch_size

        # 테스트 진행 중, 클래스별 2x2 Confusion Matrix 누적
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

"""==============================================================
# 테스트 결과 표시
=============================================================="""
# 테스트가 끝난 후, 테스트 손실 및 Macro 성능 지표 계산
test_loss = test_loss_sum / test_data_count

true_positive = confusion_matrices[:, 1, 1].float()
false_positive = confusion_matrices[:, 0, 1].float()
false_negative = confusion_matrices[:, 1, 0].float()

precision_per_class = true_positive / \
    (true_positive + false_positive).clamp(min=1)
recall_per_class = true_positive / \
    (true_positive + false_negative).clamp(min=1)
f1_per_class = (2 * precision_per_class * recall_per_class) / \
    (precision_per_class + recall_per_class).clamp(min=1e-12)

macro_precision = precision_per_class.mean().item()
macro_recall = recall_per_class.mean().item()
macro_f1 = f1_per_class.mean().item()

# 테스트가 끝난 후, 전체 테스트 결과 표시 (Macro Average)
tqdm.write(f"Test Loss: {test_loss:.4f} | "
           f"Macro Precision: {macro_precision:.4f} | "
           f"Macro Recall: {macro_recall:.4f} | "
           f"Macro F1: {macro_f1:.4f}")
