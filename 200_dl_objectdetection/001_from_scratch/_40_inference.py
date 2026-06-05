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
# 기본 설정
=============================================================="""

config = Config()
class_name_to_label = config.class_name_to_label

# label 순서대로 class name list 만들기
class_items = sorted(class_name_to_label.items(),
                     key=lambda class_item: class_item[1])
class_names = [class_name for class_name, _ in class_items]

# GPU가 있으면 cuda, 없으면 cpu 사용
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = os.path.join(config.CHECKPOINT_DIR, config.CHECKPOINT_NAME)

# webcam frame을 model 입력 크기와 맞춤
img_ch = 3
img_height = config.MODEL_IMAGE_HEIGHT
img_width = config.MODEL_IMAGE_WIDTH

"""==============================================================
# model과 checkpoint 불러오기
=============================================================="""
# object detection model 생성
model = CNN_Model(img_ch=img_ch)
model = model.to(device=device)

# 저장된 checkpoint에서 model weight 불러오기
checkpoint = torch.load(f=checkpoint_path, map_location=device)
model.load_state_dict(state_dict=checkpoint["model_state_dict"])
model.eval()

"""==============================================================
# 함수: bbox 좌표 순서 정리
=============================================================="""


def order_bbox(bbox):
    # 예측 bbox의 좌표 순서가 뒤집혀도 그릴 수 있게 정렬
    x_min = min(float(bbox[0]), float(bbox[2]))
    y_min = min(float(bbox[1]), float(bbox[3]))
    x_max = max(float(bbox[0]), float(bbox[2]))
    y_max = max(float(bbox[1]), float(bbox[3]))

    return x_min, y_min, x_max, y_max

"""==============================================================
# webcam 설정
=============================================================="""

camera = cv2.VideoCapture(config.CAMERA_INDEX)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, img_width)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, img_height)

"""==============================================================
# webcam frame 실시간 추론
=============================================================="""

while True:
    # webcam frame 읽기
    ok, frame_bgr = camera.read()
    if not ok:
        break

    # frame을 model 입력 크기로 resize
    frame_bgr = cv2.resize(src=frame_bgr,
                           dsize=(img_width, img_height),
                           interpolation=cv2.INTER_AREA)

    # OpenCV BGR 이미지를 RGB tensor로 변환
    frame_rgb = cv2.cvtColor(src=frame_bgr, code=cv2.COLOR_BGR2RGB)
    image_tensor = TF.to_tensor(frame_rgb).unsqueeze(0).to(device=device)

    # model forward 후 class 확률 계산
    with torch.no_grad():
        class_logits, bbox_pred = model(image_tensor)
        probabilities = F.softmax(input=class_logits, dim=1)[0]
        confidence, predicted_index = torch.max(input=probabilities, dim=0)

    # 예측 class label, class name, confidence 추출
    predicted_label = int(predicted_index.item())
    predicted_name = class_names[predicted_label]
    confidence_value = float(confidence.item())

    # background가 아니고 confidence가 충분할 때만 bbox 표시
    draw_bbox = (
        predicted_label != class_name_to_label["background"] and
        confidence_value >= config.INFERENCE_PROB_THRESHOLD
    )

    # 0~1 비율 bbox를 pixel 좌표로 변환
    bbox = bbox_pred[0].detach().cpu()
    x_min, y_min, x_max, y_max = order_bbox(bbox=bbox)
    x_min = int(x_min * img_width)
    y_min = int(y_min * img_height)
    x_max = int(x_max * img_width)
    y_max = int(y_max * img_height)

    display_bgr = frame_bgr.copy()

    # bbox를 그릴 경우 bbox 위에 text 표시, 아니면 좌상단에 text 표시
    if draw_bbox:
        cv2.rectangle(img=display_bgr,
                      pt1=(x_min, y_min),
                      pt2=(x_max, y_max),
                      color=(50, 200, 50),
                      thickness=2)
        text_x = x_min
        text_y = max(y_min - 10, 25)
    else:
        text_x = 10
        text_y = 25

    # class name과 confidence를 화면에 표시
    label_text = f"{predicted_name} {confidence_value:.2f}"
    cv2.putText(img=display_bgr,
                text=label_text,
                org=(text_x, text_y),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.7,
                color=(255, 255, 255),
                thickness=5,
                lineType=cv2.LINE_AA)
    cv2.putText(img=display_bgr,
                text=label_text,
                org=(text_x, text_y),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.7,
                color=(50, 200, 50),
                thickness=2,
                lineType=cv2.LINE_AA)

    cv2.imshow("object detection", display_bgr)

    # q 또는 ESC를 누르면 종료
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break

"""==============================================================
# webcam 종료
=============================================================="""

camera.release()
cv2.destroyAllWindows()
