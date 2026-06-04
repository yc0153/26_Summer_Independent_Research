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

# label 순서에 맞춘 class name list 생성
class_items = sorted(config.class_name_to_label.items(),
                     key=lambda class_item: class_item[1])
class_names = [class_name for class_name, _ in class_items]

# 추론 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 추론에 사용할 checkpoint file path
checkpoint_path = os.path.join(
    config.CHECKPOINT_DIR, config.INFERENCE_CKPT_NAME)

# 모델 입력 이미지 구조 설정
img_ch = 3
img_height = config.MODEL_IMAGE_HEIGHT
img_width = config.MODEL_IMAGE_WIDTH

"""==============================================================
# 디텍션 모델 생성과 checkpoint 불러오기
=============================================================="""
# 모델 생성 후 추론 장치로 이동
model = CNN_Model(img_ch=img_ch)
model = model.to(device=device)

# 저장된 checkpoint의 모델 weight 불러오기
checkpoint = torch.load(f=checkpoint_path, map_location=device)
model.load_state_dict(state_dict=checkpoint["model_state_dict"])

"""==============================================================
# 바운딩 박스 정리 함수
=============================================================="""


def order_bbox(bbox):
    """==============================================================
    ## bbox 좌표 순서 보정
    =============================================================="""
    # x/y min/max가 뒤집혀도 화면에 그릴 수 있도록 정렬
    x_min = min(float(bbox[0]), float(bbox[2]))
    y_min = min(float(bbox[1]), float(bbox[3]))
    x_max = max(float(bbox[0]), float(bbox[2]))
    y_max = max(float(bbox[1]), float(bbox[3]))
    return x_min, y_min, x_max, y_max

"""==============================================================
# webcam 설정
=============================================================="""
# webcam 열기와 입력 크기 요청
camera = cv2.VideoCapture(config.CAMERA_INDEX)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, img_width)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, img_height)

"""==============================================================
# 모델 추론
=============================================================="""
# 모델을 평가 모드로 전환
model.eval()

# webcam frame을 반복해서 읽고 추론
while camera.isOpened():
    # webcam frame 읽기
    ok, frame_bgr = camera.read()
    if not ok:
        break

    # webcam frame 크기가 모델 입력 크기와 다르면 resize
    frame_height, frame_width = frame_bgr.shape[:2]
    if frame_width != img_width or frame_height != img_height:
        frame_bgr = cv2.resize(src=frame_bgr,
                               dsize=(img_width, img_height),
                               interpolation=cv2.INTER_AREA)

    # BGR frame을 RGB tensor로 변환
    model_image = cv2.cvtColor(src=frame_bgr, code=cv2.COLOR_BGR2RGB)
    image_tensor = TF.to_tensor(model_image).unsqueeze(0).to(device=device)

    # class logits와 bbox 예측
    with torch.no_grad():
        class_logits, bbox_pred = model(image_tensor)
        probabilities = F.softmax(input=class_logits, dim=1)[0]
        confidence, predicted_index = torch.max(input=probabilities, dim=0)

    # 예측 class name과 confidence 정리
    predicted_label = int(predicted_index.item())
    predicted_name = class_names[predicted_label]
    confidence_value = float(confidence.item())

    # 정규화 bbox를 pixel 좌표로 변환
    bbox = bbox_pred[0, predicted_label].detach().cpu()
    x_min, y_min, x_max, y_max = order_bbox(bbox)
    x_min = int(max(0.0, min(1.0, x_min)) * img_width)
    y_min = int(max(0.0, min(1.0, y_min)) * img_height)
    x_max = int(max(0.0, min(1.0, x_max)) * img_width)
    y_max = int(max(0.0, min(1.0, y_max)) * img_height)

    """==============================================================
    ## 추론 결과 그리기
    =============================================================="""
    # frame 위에 bbox 그리기
    display_bgr = frame_bgr.copy()
    cv2.rectangle(img=display_bgr,
                  pt1=(x_min, y_min),
                  pt2=(x_max, y_max),
                  color=(50, 200, 50),
                  thickness=2)

    # frame 위에 class name과 confidence 표시
    label_text = f"{predicted_name} {confidence_value:.2f}"
    text_y = max(y_min - 10, 25)
    cv2.putText(img=display_bgr,
                text=label_text,
                org=(x_min, text_y),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.7,
                color=(255, 255, 255),
                thickness=5,
                lineType=cv2.LINE_AA)
    cv2.putText(img=display_bgr,
                text=label_text,
                org=(x_min, text_y),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.7,
                color=(50, 200, 50),
                thickness=2,
                lineType=cv2.LINE_AA)

    # 추론 화면 표시
    cv2.imshow("object detection", display_bgr)

    # q 또는 ESC 입력 시 종료
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break

"""==============================================================
# webcam 종료
=============================================================="""
# camera resource와 OpenCV window 정리
camera.release()
cv2.destroyAllWindows()
