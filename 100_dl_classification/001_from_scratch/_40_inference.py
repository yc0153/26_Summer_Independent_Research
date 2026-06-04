"""==============================================================
# 필요한 라이브러리 불러오기
=============================================================="""
import os
import cv2
import torch
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from _00_config import Config
from _20_network import CNN_Model

"""==============================================================
# 준비 작업
=============================================================="""
# 기본 설정값 불러오기
config = Config()

# 클래스 이름 목록 생성
class_items = sorted(config.class_name_to_label.items(),
                     key=lambda class_item: class_item[1])
class_names = [class_name for class_name, _ in class_items]

# 학습 장치 설정 (cpu or gpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 체크포인트 파일 경로
checkpoint_path = os.path.join(
    config.CHECKPOINT_DIR, config.INFERENCE_CKPT_NAME)

# 입력 이미지 구조 설정
img_ch = 3
img_height = config.MODEL_IMAGE_HEIGHT
img_width = config.MODEL_IMAGE_WIDTH

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
# 웹캠 설정
=============================================================="""
# 웹캠 열기
camera = cv2.VideoCapture(config.CAMERA_INDEX)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, img_width)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, img_height)

"""==============================================================
# 모델 추론
=============================================================="""
# 모델을 평가 모드로 전환
model.eval()

while camera.isOpened():
    # 웹캠 프레임 읽기
    ok, frame_bgr = camera.read()
    if not ok:
        break

    # 웹캠 프레임 크기 조정
    frame_height, frame_width = frame_bgr.shape[:2]
    if frame_width != img_width or frame_height != img_height:
        frame_bgr = cv2.resize(src=frame_bgr,
                               dsize=(img_width, img_height),
                               interpolation=cv2.INTER_AREA)

    # 추론 데이터 준비
    model_image = cv2.cvtColor(src=frame_bgr, code=cv2.COLOR_BGR2RGB)
    image_tensor = TF.to_tensor(model_image).unsqueeze(0).to(device=device)

    # 예측
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = F.softmax(input=logits, dim=1)[0]
        confidence, predicted_index = torch.max(input=probabilities, dim=0)

    # 예측 결과 정리
    predicted_name = class_names[int(predicted_index.item())]
    confidence_value = float(confidence.item())

    # 예측 결과 표시
    display_bgr = frame_bgr.copy()
    cv2.putText(img=display_bgr,
                text=f"{predicted_name}  {confidence_value:.2f}",
                org=(20, 40),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(255, 255, 255),
                thickness=5,
                lineType=cv2.LINE_AA)
    cv2.putText(img=display_bgr,
                text=f"{predicted_name}  {confidence_value:.2f}",
                org=(20, 40),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 0, 0),
                thickness=2,
                lineType=cv2.LINE_AA)

    cv2.imshow("3-class classification", display_bgr)

    # 종료 키 입력 확인
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break

"""==============================================================
# 웹캠 종료
=============================================================="""
# 웹캠 종료
camera.release()
cv2.destroyAllWindows()
