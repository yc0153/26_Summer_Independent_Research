"""==============================================================
# 필요한 라이브러리 불러오기
=============================================================="""
import os

import matplotlib.pyplot as plt
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

# random seed 고정
torch.manual_seed(config.RANDOM_SEED)

# checkpoint directory 생성
os.makedirs(name=config.CHECKPOINT_DIR, exist_ok=True)

# 학습 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""==============================================================
# 학습과 검증 데이터셋 생성
=============================================================="""
# train/valid dataset 생성
train_dataset = DataSet(split_name="train")
valid_dataset = DataSet(split_name="valid")

# 첫 번째 이미지 tensor에서 입력 채널 수 확인
first_img_tensor = train_dataset[0][0]
img_ch = int(first_img_tensor.shape[0])

"""==============================================================
# 학습과 검증 dataloader 생성
=============================================================="""
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=config.BATCH_SIZE,
                          shuffle=True,
                          num_workers=config.NUM_WORKERS)

valid_loader = DataLoader(dataset=valid_dataset,
                          batch_size=config.BATCH_SIZE,
                          shuffle=False,
                          num_workers=config.NUM_WORKERS)

"""==============================================================
# 디텍션 모델 생성
=============================================================="""
# CNN_Model 객체 생성 후 학습 장치로 이동
model = CNN_Model(img_ch=img_ch)
model = model.to(device=device)

"""==============================================================
# 손실 함수와 optimizer 설정
=============================================================="""
# class prediction loss와 bbox regression loss 설정
class_loss_fn = nn.CrossEntropyLoss()
bbox_loss_fn = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(
    params=model.parameters(), lr=config.LEARNING_RATE)

"""==============================================================
# 학습 상태 초기화와 checkpoint 설정
=============================================================="""
# 학습 시작 epoch와 best valid loss 초기화
start_epoch = 1
best_valid_loss = float("inf")

# checkpoint file path 생성
checkpoint_path = os.path.join(
    config.CHECKPOINT_DIR, config.TRAIN_SETTING["ckp_name"])

# 이어 학습 모드이면 checkpoint에서 모델과 optimizer 상태 복원
if config.TRAIN_SETTING["train_mode"] == "resume" and os.path.exists(checkpoint_path):
    checkpoint = torch.load(f=checkpoint_path, map_location=device)
    model.load_state_dict(state_dict=checkpoint["model_state_dict"])
    optimizer.load_state_dict(state_dict=checkpoint["optimizer_state_dict"])
    start_epoch = int(checkpoint["epoch"]) + 1
    best_valid_loss = float(checkpoint["best_valid_loss"])

"""==============================================================
# 디텍션 손실 계산 함수
=============================================================="""


def get_detection_loss(class_logits, bbox_pred, labels, bboxes):
    """==============================================================
    ## class loss와 bbox loss를 합산
    =============================================================="""
    # class 분류 손실과 bbox 좌표 회귀 손실 계산
    class_loss = class_loss_fn(input=class_logits, target=labels)

    # 정답 클래스에 해당하는 bbox 예측값만 선택
    batch_indices = torch.arange(labels.size(0), device=labels.device)
    bbox_pred_selected = bbox_pred[batch_indices, labels]

    # 선택된 bbox 예측값으로 bbox 좌표 회귀 손실 계산
    bbox_loss = bbox_loss_fn(input=bbox_pred_selected, target=bboxes)

    # bbox loss weight를 곱해 최종 디텍션 손실 계산
    loss = class_loss + config.BBOX_LOSS_WEIGHT * bbox_loss
    return loss, class_loss, bbox_loss

"""==============================================================
# 모델 학습과 검증
=============================================================="""
train_loss_history = []
valid_loss_history = []
for epoch in range(start_epoch, config.NUM_EPOCHS + 1):
    """==============================================================
    ## 학습
    =============================================================="""
    # 모델을 학습 모드로 전환
    model.train()

    # train loss 누적 변수 초기화
    train_loss_sum = 0.0
    train_data_count = 0
    train_progress = tqdm(train_loader,
                          desc=f"Train {epoch}/{config.NUM_EPOCHS}")

    # train batch loop
    for images, labels, bboxes in train_progress:
        # batch data를 학습 장치로 이동
        images = images.to(device=device)
        labels = labels.to(device=device)
        bboxes = bboxes.to(device=device)

        # class logits와 bbox 예측 후 loss 계산
        class_logits, bbox_pred = model(images)
        loss, class_loss, bbox_loss = get_detection_loss(
            class_logits=class_logits,
            bbox_pred=bbox_pred,
            labels=labels,
            bboxes=bboxes,
        )

        # gradient 계산과 model parameter update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 진행 중 평균 train loss 계산
        batch_size = labels.size(0)
        train_loss_sum = train_loss_sum + loss.item() * batch_size
        train_data_count = train_data_count + batch_size
        train_loss = train_loss_sum / train_data_count

        # 진행바에 전체 loss, class loss, bbox loss 표시
        train_progress.set_postfix(loss=f"{train_loss:.4f}",
                                   cls=f"{class_loss.item():.4f}",
                                   box=f"{bbox_loss.item():.4f}")

    # epoch 단위 train loss 저장
    train_loss = train_loss_sum / train_data_count
    train_loss_history.append(train_loss)

    """==============================================================
    ## 검증
    =============================================================="""
    # 모델을 평가 모드로 전환
    model.eval()

    # valid loss 누적 변수 초기화
    valid_loss_sum = 0.0
    valid_data_count = 0
    valid_progress = tqdm(valid_loader,
                          desc=f"Valid {epoch}/{config.NUM_EPOCHS}")

    # gradient 계산 없이 valid batch loop 실행
    with torch.no_grad():
        for images, labels, bboxes in valid_progress:
            # batch data를 학습 장치로 이동
            images = images.to(device=device)
            labels = labels.to(device=device)
            bboxes = bboxes.to(device=device)

            # class logits와 bbox 예측 후 loss 계산
            class_logits, bbox_pred = model(images)
            loss, class_loss, bbox_loss = get_detection_loss(
                class_logits=class_logits,
                bbox_pred=bbox_pred,
                labels=labels,
                bboxes=bboxes,
            )

            # 진행 중 평균 valid loss 계산
            batch_size = labels.size(0)
            valid_loss_sum = valid_loss_sum + loss.item() * batch_size
            valid_data_count = valid_data_count + batch_size
            valid_loss = valid_loss_sum / valid_data_count

            # 진행바에 전체 loss, class loss, bbox loss 표시
            valid_progress.set_postfix(loss=f"{valid_loss:.4f}",
                                       cls=f"{class_loss.item():.4f}",
                                       box=f"{bbox_loss.item():.4f}")

    # epoch 단위 valid loss 저장
    valid_loss = valid_loss_sum / valid_data_count
    valid_loss_history.append(valid_loss)

    # epoch 결과 표시
    tqdm.write(f"Epoch {epoch}/{config.NUM_EPOCHS} | "
               f"Train Loss: {train_loss:.4f} | "
               f"Valid Loss: {valid_loss:.4f}")

    # valid loss가 가장 낮아진 경우 checkpoint 저장
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss

        torch.save(
            obj={
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_valid_loss": best_valid_loss,
            },
            f=checkpoint_path,
        )

"""==============================================================
# 학습 곡선 표시
=============================================================="""
# epoch별 train/valid loss curve 표시
epoch_numbers = list(range(start_epoch, start_epoch + len(train_loss_history)))

plt.figure(figsize=(8, 5))

plt.plot(epoch_numbers, train_loss_history, label="Train loss")
plt.plot(epoch_numbers, valid_loss_history, label="Valid loss")

plt.title("Training curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.grid(True)
plt.figtext(x=0.5,
            y=0.01,
            s=f"Checkpoint: {checkpoint_path}",
            ha="center",
            fontsize=8)
plt.legend()

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()
