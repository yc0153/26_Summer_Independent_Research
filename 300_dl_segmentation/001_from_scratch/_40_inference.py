"""==============================================================
# 필요한 라이브러리 불러오기
=============================================================="""
import os

import cv2
import numpy as np
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
num_classes = len(class_names)

# 추론 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 추론에 사용할 checkpoint file path
checkpoint_path = os.path.join(
    config.CHECKPOINT_DIR, config.INFERENCE_CKPT_NAME)

# 모델 입력 이미지 구조 설정
img_ch = 3
img_height = config.MODEL_IMAGE_HEIGHT
img_width = config.MODEL_IMAGE_WIDTH

# background label 확인
background_label = config.class_name_to_label.get("background", 0)
foreground_mask_alpha = 0.45
prob_threshold = config.INFERENCE_PROB_THRESHOLD

"""==============================================================
# 세그멘테이션 모델 생성과 checkpoint 불러오기
=============================================================="""
# 모델 생성 후 추론 장치로 이동
model = CNN_Model(img_ch=img_ch)
model = model.to(device=device)

# 저장된 checkpoint의 모델 weight 불러오기
checkpoint = torch.load(f=checkpoint_path, map_location=device)
model.load_state_dict(state_dict=checkpoint["model_state_dict"])

"""==============================================================
# 추론 보조 함수 정의
=============================================================="""


def make_color_map(num_classes):
    """==============================================================
    ## class별 mask 색상 생성
    =============================================================="""
    # OpenCV 표시용 BGR 색상 palette
    base_colors = [
        (0, 0, 0),
        (255, 0, 0),
        (0, 255, 0),
        (240, 180, 60),
        (220, 60, 180),
        (180, 220, 60),
    ]

    colors = []
    for class_idx in range(num_classes):
        colors.append(base_colors[class_idx % len(base_colors)])

    # 주요 class 색상은 label 기준으로 명시적으로 고정
    if "background" in config.class_name_to_label:
        colors[config.class_name_to_label["background"]] = (0, 0, 0)
    if "mouse" in config.class_name_to_label:
        colors[config.class_name_to_label["mouse"]] = (255, 0, 0)
    if "eraser" in config.class_name_to_label:
        colors[config.class_name_to_label["eraser"]] = (0, 255, 0)

    return colors


# class별 mask 색상은 추론 중 반복 생성하지 않도록 미리 생성
mask_colors = make_color_map(num_classes=num_classes)


def resize_frame_to_model_size(frame_bgr):
    """==============================================================
    ## webcam frame을 모델 입력 크기에 맞춤
    =============================================================="""
    # 카메라가 요청한 해상도를 무시할 수 있으므로, 추론 직전에 직접 크기 보정
    frame_height, frame_width = frame_bgr.shape[:2]
    if frame_width == img_width and frame_height == img_height:
        return frame_bgr

    interpolation = cv2.INTER_AREA
    if frame_width < img_width or frame_height < img_height:
        interpolation = cv2.INTER_LINEAR

    frame_bgr = cv2.resize(src=frame_bgr,
                           dsize=(img_width, img_height),
                           interpolation=interpolation)

    return frame_bgr


def get_segmentation_mask(probabilities):
    """==============================================================
    ## pixel별 class probability로 segmentation mask 생성
    =============================================================="""
    # 가장 확률이 높은 class를 각 pixel의 예측 label로 선택
    predicted_mask = np.argmax(a=probabilities, axis=0).astype(np.uint8)

    # foreground class 확률이 threshold 이상인 pixel만 표시
    max_probabilities = np.max(a=probabilities, axis=0)
    foreground_mask = predicted_mask != background_label
    confident_mask = max_probabilities >= prob_threshold
    predicted_mask[~(foreground_mask & confident_mask)] = background_label

    return predicted_mask


def get_raw_segmentation_mask(probabilities):
    """==============================================================
    ## threshold 적용 전 pixel별 예측 class mask 생성
    =============================================================="""
    predicted_mask = np.argmax(a=probabilities, axis=0).astype(np.uint8)

    return predicted_mask


def make_mask_visualization(predicted_mask):
    """==============================================================
    ## 예측 mask를 확인하기 쉬운 BGR 이미지로 변환
    =============================================================="""
    color_table = np.asarray(mask_colors, dtype=np.uint8)
    mask_bgr = color_table[predicted_mask]

    return mask_bgr


def print_mask_pixel_counts(predicted_mask):
    """==============================================================
    ## class별 예측 pixel 개수 출력
    =============================================================="""
    count_texts = []
    for class_idx, class_name in enumerate(class_names):
        pixel_count = int((predicted_mask == class_idx).sum())
        count_texts.append(f"{class_name}: {pixel_count}")

    print(" | ".join(count_texts))


def draw_legend(display_bgr):
    """==============================================================
    ## class 색상 범례 표시
    =============================================================="""
    legend_x = 10
    legend_y = 20
    box_size = 12
    row_gap = 18
    row_idx = 0

    for class_idx in range(num_classes):
        if class_idx == background_label:
            continue

        y = legend_y + row_idx * row_gap
        cv2.rectangle(img=display_bgr,
                      pt1=(legend_x, y - box_size + 2),
                      pt2=(legend_x + box_size, y + 2),
                      color=mask_colors[class_idx],
                      thickness=-1)
        cv2.putText(img=display_bgr,
                    text=class_names[class_idx],
                    org=(legend_x + box_size + 6, y + 2),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.45,
                    color=(255, 255, 255),
                    thickness=3,
                    lineType=cv2.LINE_AA)
        cv2.putText(img=display_bgr,
                    text=class_names[class_idx],
                    org=(legend_x + box_size + 6, y + 2),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.45,
                    color=mask_colors[class_idx],
                    thickness=1,
                    lineType=cv2.LINE_AA)
        row_idx = row_idx + 1


def draw_segmentation_result(frame_bgr, predicted_mask):
    """==============================================================
    ## mask와 class 정보를 frame 위에 표시
    =============================================================="""
    # mask class별 색상 이미지 생성
    color_mask = np.zeros_like(frame_bgr)
    for class_idx in range(num_classes):
        if class_idx == background_label:
            continue

        color_mask[predicted_mask == class_idx] = mask_colors[class_idx]

    # background는 색칠하지 않고, foreground는 class 색상으로 overlay
    display_bgr = frame_bgr.copy()
    foreground_mask = predicted_mask != background_label
    foreground_blended_bgr = cv2.addWeighted(
        src1=frame_bgr,
        alpha=1.0 - foreground_mask_alpha,
        src2=color_mask,
        beta=foreground_mask_alpha,
        gamma=0,
    )
    display_bgr[foreground_mask] = foreground_blended_bgr[foreground_mask]

    # class별 외곽선 표시
    for class_idx in range(num_classes):
        if class_idx == background_label:
            continue

        class_pixels = predicted_mask == class_idx
        if not class_pixels.any():
            continue

        class_mask = class_pixels.astype(np.uint8) * 255
        contours, _ = cv2.findContours(image=class_mask,
                                       mode=cv2.RETR_EXTERNAL,
                                       method=cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 20:
                continue

            cv2.drawContours(image=display_bgr,
                             contours=[contour],
                             contourIdx=-1,
                             color=mask_colors[class_idx],
                             thickness=2)

    # 가장 넓게 예측된 foreground class를 대표 class로 표시
    draw_legend(display_bgr=display_bgr)

    return display_bgr


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
frame_idx = 0
while camera.isOpened():
    # webcam frame 읽기
    ok, frame_bgr = camera.read()
    if not ok:
        break

    # webcam frame을 모델 입력 크기로 먼저 맞춘 뒤 추론 시작
    camera_frame_height, camera_frame_width = frame_bgr.shape[:2]
    frame_bgr = resize_frame_to_model_size(frame_bgr=frame_bgr)
    if frame_idx == 0:
        print(
            f"camera frame: {camera_frame_width}x{camera_frame_height} "
            f"-> model input: {img_width}x{img_height}"
        )

    # BGR frame을 RGB tensor로 변환
    model_image = cv2.cvtColor(src=frame_bgr, code=cv2.COLOR_BGR2RGB)
    image_tensor = TF.to_tensor(model_image).unsqueeze(0).to(device=device)

    # pixel별 class logits 추론
    with torch.no_grad():
        mask_logits = model(image_tensor)
        probabilities = F.softmax(input=mask_logits, dim=1)[0]

    # pixel별 class 확률을 numpy 배열로 변환 후 segmentation mask 생성
    probabilities = probabilities.detach().cpu().numpy()
    raw_predicted_mask = get_raw_segmentation_mask(probabilities=probabilities)
    predicted_mask = get_segmentation_mask(probabilities=probabilities)
    mask_bgr = make_mask_visualization(predicted_mask=predicted_mask)

    frame_idx = frame_idx + 1
    if frame_idx % 30 == 1:
        print("raw prediction")
        print_mask_pixel_counts(predicted_mask=raw_predicted_mask)
        print("thresholded prediction")
        print_mask_pixel_counts(predicted_mask=predicted_mask)

    """==============================================================
    ## 추론 결과 그리기
    =============================================================="""
    # frame 위에 mask 표시
    display_bgr = draw_segmentation_result(frame_bgr=frame_bgr,
                                           predicted_mask=predicted_mask)

    # 추론 화면 표시
    cv2.imshow("semantic segmentation", display_bgr)
    cv2.imshow("predicted mask", mask_bgr)

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
