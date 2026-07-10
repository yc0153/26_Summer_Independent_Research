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
# 기본 설정
=============================================================="""

config = Config()
torch.manual_seed(config.RANDOM_SEED)
class_name_to_label = config.class_name_to_label

# checkpoint directory 생성
os.makedirs(name=config.CHECKPOINT_DIR, exist_ok=True)
checkpoint_path = os.path.join(config.CHECKPOINT_DIR, config.CHECKPOINT_NAME)

# GPU가 있으면 cuda, 없으면 cpu 사용
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""==============================================================
# Dataset과 DataLoader 만들기
=============================================================="""
# train/valid dataset 생성
train_dataset = DataSet(split_name="train")
valid_dataset = DataSet(split_name="valid")

# image channel 수 확인
img_ch = int(train_dataset[0][0].shape[0])

# mini-batch 학습을 위한 DataLoader 생성
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=config.BATCH_SIZE,
                          shuffle=True,
                          num_workers=config.NUM_WORKERS)
valid_loader = DataLoader(dataset=valid_dataset,
                          batch_size=config.BATCH_SIZE,
                          shuffle=False,
                          num_workers=config.NUM_WORKERS)

"""==============================================================
# model, loss, optimizer 설정
=============================================================="""
# object detection model 생성
model = CNN_Model(img_ch=img_ch)
model = model.to(device=device)

# classification loss와 bbox regression loss 정의
class_loss_fn = nn.CrossEntropyLoss()
bbox_loss_fn = nn.L1Loss()

# optimizer 정의
optimizer = torch.optim.Adam(params=model.parameters(),
                             lr=config.LEARNING_RATE)

"""==============================================================
# 함수: object detection loss 계산
=============================================================="""


def get_detection_loss(class_logits, bbox_pred, labels, bboxes):
    # class 예측 loss
    class_loss = class_loss_fn(input=class_logits, target=labels)

    # background가 아닌 sample에 대해서만 bbox loss 계산
    object_mask = labels != class_name_to_label["background"]
    bbox_loss = bbox_loss_fn(input=bbox_pred[object_mask],
                             target=bboxes[object_mask])

    # bbox loss에 weight를 곱해서 classification loss와 합산
    loss = class_loss + config.BBOX_LOSS_WEIGHT * bbox_loss

    return loss, class_loss, bbox_loss

"""==============================================================
# 학습 준비
=============================================================="""

best_valid_loss = float("inf")
train_loss_history = []
valid_loss_history = []

"""==============================================================
# model 학습
=============================================================="""

for epoch in range(1, config.NUM_EPOCHS + 1):
    """==============================================================
    ## train phase
    =============================================================="""
    model.train()

    train_loss_sum = 0.0
    train_data_count = 0
    train_progress = tqdm(train_loader,
                          desc=f"Train {epoch}/{config.NUM_EPOCHS}")

    for images, labels, bboxes in train_progress:
        # mini-batch를 device로 이동
        images = images.to(device=device)
        labels = labels.to(device=device)
        bboxes = bboxes.to(device=device)

        # forward와 loss 계산
        class_logits, bbox_pred = model(images)
        loss, class_loss, bbox_loss = get_detection_loss(
            class_logits=class_logits,
            bbox_pred=bbox_pred,
            labels=labels,
            bboxes=bboxes,
        )

        # backward와 parameter update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # epoch 평균 train loss 갱신
        batch_size = labels.size(0)
        train_loss_sum = train_loss_sum + loss.item() * batch_size
        train_data_count = train_data_count + batch_size
        train_loss = train_loss_sum / train_data_count

        train_progress.set_postfix(loss=f"{train_loss:.4f}",
                                   cls=f"{class_loss.item():.4f}",
                                   box=f"{bbox_loss.item():.4f}")

    train_loss = train_loss_sum / train_data_count
    train_loss_history.append(train_loss)

    """==============================================================
    ## valid phase
    =============================================================="""
    model.eval()

    valid_loss_sum = 0.0
    valid_data_count = 0
    valid_progress = tqdm(valid_loader,
                          desc=f"Valid {epoch}/{config.NUM_EPOCHS}")

    with torch.no_grad():
        for images, labels, bboxes in valid_progress:
            # mini-batch를 device로 이동
            images = images.to(device=device)
            labels = labels.to(device=device)
            bboxes = bboxes.to(device=device)

            # validation forward와 loss 계산
            class_logits, bbox_pred = model(images)
            loss, class_loss, bbox_loss = get_detection_loss(
                class_logits=class_logits,
                bbox_pred=bbox_pred,
                labels=labels,
                bboxes=bboxes,
            )

            # epoch 평균 valid loss 갱신
            batch_size = labels.size(0)
            valid_loss_sum = valid_loss_sum + loss.item() * batch_size
            valid_data_count = valid_data_count + batch_size
            valid_loss = valid_loss_sum / valid_data_count

            valid_progress.set_postfix(loss=f"{valid_loss:.4f}",
                                       cls=f"{class_loss.item():.4f}",
                                       box=f"{bbox_loss.item():.4f}")

    valid_loss = valid_loss_sum / valid_data_count
    valid_loss_history.append(valid_loss)

    # 현재 epoch 결과 출력
    tqdm.write(f"Epoch {epoch}/{config.NUM_EPOCHS} | "
               f"Train Loss: {train_loss:.4f} | "
               f"Valid Loss: {valid_loss:.4f}")

    # best valid loss를 갱신하면 checkpoint 저장
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
# training curve 시각화
=============================================================="""

epoch_numbers = list(range(1, config.NUM_EPOCHS + 1))

plt.figure(figsize=(8, 5))
plt.plot(epoch_numbers, train_loss_history, label="Train loss")
plt.plot(epoch_numbers, valid_loss_history, label="Valid loss")
plt.title("Training curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
