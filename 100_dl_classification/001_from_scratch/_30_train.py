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

# 랜덤 시드 고정 (재현성을 위함)
torch.manual_seed(config.RANDOM_SEED)

# 체크포인트 디렉도리 생성(존재하지 않는 경우)
os.makedirs(name=config.CHECKPOINT_DIR, exist_ok=True)

# 학습 장치 설정 (cpu or gpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""==============================================================
# 학습을 위한 데이터셋 생성
=============================================================="""
# 훈련 데이터셋
train_dataset = DataSet(split_name="train")

# 검증 데이터셋
valid_dataset = DataSet(split_name="valid")

# 입력 이미지 채널 수 확인
first_img_tensor = train_dataset[0][0]
img_ch = int(first_img_tensor.shape[0])

"""==============================================================
# 학습을 위한 데이터로더 생성
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
# 딥 뉴럴 네트워크 모델 생성
=============================================================="""
# 정의된 CNN_Model 클래스를 이용하여 모델 객체 생성
model = CNN_Model(img_ch=img_ch)

# 모델을 설정된 학습장치로 이동
model = model.to(device=device)

"""==============================================================
# 손실함수 및 최적화기 설정
=============================================================="""
# 손실함수 및 최적화기 설정
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    params=model.parameters(), lr=config.LEARNING_RATE)

"""==============================================================
# 학습 변수 초기화 및 이어서 학습할지 설정
=============================================================="""
# 학습 변수 초기화
start_epoch = 1
best_valid_loss = float("inf")

# 체크포인트 파일 경로
checkpoint_path = os.path.join(
    config.CHECKPOINT_DIR, config.TRAIN_SETTING["ckp_name"])

# 이어서 학습하기 위한 체크포인트 불러오기 (이어서 학습하기로 설정되어 있는 경우에 한함)
if config.TRAIN_SETTING["train_mode"] == "resume" and os.path.exists(checkpoint_path):
    # 체크포인트 불러오기
    checkpoint = torch.load(f=checkpoint_path, map_location=device)

    # 모델 웨이트 불러오기
    model.load_state_dict(state_dict=checkpoint["model_state_dict"])

    # 최적화기 상태 불러오기
    optimizer.load_state_dict(state_dict=checkpoint["optimizer_state_dict"])

    # 학습 변수 불러와서 업데이트
    start_epoch = int(checkpoint["epoch"]) + 1
    best_valid_loss = float(checkpoint["best_valid_loss"])

"""==============================================================
# 모델 학습 및 검증
=============================================================="""
train_loss_history = []
valid_loss_history = []
for epoch in range(start_epoch, config.NUM_EPOCHS + 1):
    """==============================================================
    ## 학습
    =============================================================="""
    # 모델을 학습 모드로 설정
    model.train()

    # 훈련 변수 초기화
    train_loss_sum = 0.0
    train_data_count = 0

    # 훈련 진행률 바 생성
    train_progress = tqdm(train_loader,
                          desc=f"Train {epoch}/{config.NUM_EPOCHS}")

    # 훈련 루프
    for images, labels in train_progress:
        # 훈련 데이터 준비
        images = images.to(device=device)
        labels = labels.to(device=device)

        # 예측 및 손실 계산
        logits = model(images)
        loss = loss_fn(input=logits, target=labels)

        # 모델 업데이트
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 에포크 진행 중, 평균 로스 계산을 위한 데이터 누적
        batch_size = labels.size(0)
        train_loss_sum = train_loss_sum + loss.item() * batch_size
        train_data_count = train_data_count + batch_size
        train_loss = train_loss_sum / train_data_count

    # 한 에포크 지난 후, 룬련 손실 저장
    train_loss = train_loss_sum / train_data_count
    train_loss_history.append(train_loss)

    """==============================================================
    ## 검증
    =============================================================="""
    # 모델을 검증 모드로 전환
    model.eval()

    # 검증 변수 초기화
    valid_loss_sum = 0.0
    valid_data_count = 0

    # 검증 진행률 바 생성
    valid_progress = tqdm(valid_loader,
                          desc=f"Valid {epoch}/{config.NUM_EPOCHS}")

    with torch.no_grad():
        for images, labels in valid_progress:
            # 검증 데이터 준비
            images = images.to(device=device)
            labels = labels.to(device=device)

            # 예측 및 손실 계산
            logits = model(images)
            loss = loss_fn(input=logits, target=labels)

            # 에포크 진행 중, 평균 로스 계산을 위한 데이터 누적
            batch_size = labels.size(0)
            valid_loss_sum = valid_loss_sum + loss.item() * batch_size
            valid_data_count = valid_data_count + batch_size
            valid_loss = valid_loss_sum / valid_data_count

            # 에포크 진행 중, 훈련 손실 표시
            valid_progress.set_postfix(loss=f"{valid_loss:.4f}")

    # 한 에포크 지난 후, 훈련 손실 저장
    valid_loss = valid_loss_sum / valid_data_count
    valid_loss_history.append(valid_loss)

    # 한 에포크 지난 후, 훈련 및 검증 손실 표시
    tqdm.write(f"Epoch {epoch}/{config.NUM_EPOCHS} | "
               f"Train Loss: {train_loss:.4f} | "
               f"Valid Loss: {valid_loss:.4f}")

    # 한 에포크 지난 후, 체크포인트 저장 (검증 손실 최저치를 갱신했을 경우에 한함)
    if valid_loss < best_valid_loss:

        # 검증 손실 최저치 갱신
        best_valid_loss = valid_loss

        # 체크포인트 저장
        torch.save(
            obj={
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_valid_loss": best_valid_loss,
            },
            f=checkpoint_path
        )

"""==============================================================ㅋ
# 학습 커브 표시
=============================================================="""
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
