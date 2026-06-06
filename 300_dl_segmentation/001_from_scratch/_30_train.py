"""==============================================================
# 필요한 라이브러리 불러오기
=============================================================="""
import os

import matplotlib.pyplot as plt
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
# 기본 설정값 불러오기
config = Config()
num_classes = len(config.class_name_to_label)

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
# 세그멘테이션 모델 생성
=============================================================="""
# CNN_Model 객체 생성 후 학습 장치로 이동
model = CNN_Model(img_ch=img_ch)
model = model.to(device=device)

"""==============================================================
# 손실 함수와 optimizer 설정
=============================================================="""
# background pixel이 너무 많은 문제를 줄이기 위한 class weight 생성
class_weights = torch.ones(size=(num_classes,), dtype=torch.float32)
background_label = config.class_name_to_label["background"]
class_weights[background_label] = config.BACKGROUND_LOSS_WEIGHT
class_weights = class_weights.to(device=device)

# pixel-wise class prediction loss와 optimizer 설정
cross_entropy_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(params=model.parameters(),
                             lr=config.LEARNING_RATE)

"""==============================================================
# 학습 상태 초기화와 checkpoint 설정
=============================================================="""
# 학습 시작 epoch와 best valid loss 초기화
start_epoch = 1
end_epoch = config.NUM_EPOCHS
best_valid_loss = float("inf")

# checkpoint file path 생성
checkpoint_path = os.path.join(
    config.CHECKPOINT_DIR, config.TRAIN_SETTING["ckp_name"])

# 이어 학습 모드라면 checkpoint에서 모델과 optimizer 상태 복원
if config.TRAIN_SETTING["train_mode"] == "resume":
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(f=checkpoint_path, map_location=device)
    model.load_state_dict(state_dict=checkpoint["model_state_dict"])
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(state_dict=checkpoint["optimizer_state_dict"])
    else:
        print("optimizer_state_dict not found. Resume with a new optimizer.")
    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    end_epoch = start_epoch + config.NUM_EPOCHS - 1
    best_valid_loss = float(checkpoint.get("best_valid_loss", float("inf")))
elif config.TRAIN_SETTING["train_mode"] != "fresh":
    raise ValueError(
        f"Invalid train_mode: {config.TRAIN_SETTING['train_mode']}"
    )

"""==============================================================
# 모델 학습과 검증
=============================================================="""
train_loss_history = []
valid_loss_history = []
for epoch in range(start_epoch, end_epoch + 1):
    """==============================================================
    ## 학습
    =============================================================="""
    # 모델을 학습 모드로 전환
    model.train()

    # train loss 누적 변수 초기화
    train_loss_sum = 0.0
    train_data_count = 0
    train_progress = tqdm(train_loader,
                          desc=f"Train {epoch}/{end_epoch}")

    # train batch loop
    for images, masks in train_progress:
        # batch data를 학습 장치로 이동
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

        # gradient 계산과 model parameter update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 진행 중 평균 train loss 계산
        batch_size = masks.size(0)
        train_loss_sum = train_loss_sum + loss.item() * batch_size
        train_data_count = train_data_count + batch_size
        train_loss = train_loss_sum / train_data_count

        # 진행바에 전체 loss 표시
        train_progress.set_postfix(loss=f"{train_loss:.4f}")

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
                          desc=f"Valid {epoch}/{end_epoch}")

    # gradient 계산 없이 valid batch loop 실행
    with torch.no_grad():
        for images, masks in valid_progress:
            # batch data를 학습 장치로 이동
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

            # 진행 중 평균 valid loss 계산
            batch_size = masks.size(0)
            valid_loss_sum = valid_loss_sum + loss.item() * batch_size
            valid_data_count = valid_data_count + batch_size
            valid_loss = valid_loss_sum / valid_data_count

            # 진행바에 전체 loss 표시
            valid_progress.set_postfix(loss=f"{valid_loss:.4f}")

    # epoch 단위 valid loss 저장
    valid_loss = valid_loss_sum / valid_data_count
    valid_loss_history.append(valid_loss)

    # epoch 결과 표시
    tqdm.write(f"Epoch {epoch}/{end_epoch} | "
               f"Train Loss: {train_loss:.4f} | "
               f"Valid Loss: {valid_loss:.4f}")

    # 이어 학습을 위해 매 epoch의 마지막 상태를 checkpoint에 저장
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
