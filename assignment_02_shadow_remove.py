import os
import cv2
import numpy as np


def add_shadow_noise(img):
    """
    A4 이미지에 인위적인 그림자 노이즈 추가
    """
    h, w = img.shape[:2]

    # 0~1 그림자 마스크 생성
    mask = np.ones((h, w), dtype=np.float32)

    # 사선 방향 그림자 영역
    polygon = np.array([
        [int(w * 0.00), int(h * 0.10)],
        [int(w * 0.55), int(h * 0.00)],
        [int(w * 1.00), int(h * 1.00)],
        [int(w * 0.25), int(h * 1.00)]
    ], dtype=np.int32)

    shadow = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(shadow, [polygon], 255)

    # 그림자 경계를 부드럽게
    shadow = cv2.GaussianBlur(shadow, (151, 151), 0)

    # 그림자 강도
    shadow_strength = 0.45
    mask = 1.0 - (shadow.astype(np.float32) / 255.0) * shadow_strength

    shadow_img = img.astype(np.float32)

    for c in range(3):
        shadow_img[:, :, c] = shadow_img[:, :, c] * mask

    shadow_img = np.clip(shadow_img, 0, 255).astype(np.uint8)

    return shadow_img


def remove_shadow_gray(img):
    """
    그림자 제거 핵심 함수
    배경 조명 성분을 추정한 뒤 나누기 연산으로 보정
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 배경 조명 성분 추정
    # 커널이 클수록 넓은 그림자 제거에 유리함
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=45, sigmaY=45)

    # 0 나누기 방지
    background = np.where(background == 0, 1, background)

    # 원본 / 배경 = 조명 보정
    corrected = cv2.divide(gray, background, scale=255)

    # 대비 향상
    corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)

    return corrected.astype(np.uint8), background.astype(np.uint8)


def make_binary_document(corrected):
    """
    그림자 제거 후 문서처럼 흑백화
    """
    binary = cv2.adaptiveThreshold(
        corrected,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10
    )

    return binary


def remove_shadow_color(img):
    """
    컬러 이미지에서 채널별 그림자 제거
    """
    channels = cv2.split(img)
    result_channels = []

    for ch in channels:
        background = cv2.GaussianBlur(ch, (0, 0), sigmaX=45, sigmaY=45)
        background = np.where(background == 0, 1, background)

        corrected = cv2.divide(ch, background, scale=255)
        corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)

        result_channels.append(corrected.astype(np.uint8))

    result = cv2.merge(result_channels)
    return result


# ============================
# 경로 설정
# ============================
dir_base = os.path.dirname(os.path.abspath(__file__))
dir_output = os.path.join(dir_base, "output")
os.makedirs(dir_output, exist_ok=True)

# 1번 과제 결과 이미지 사용
# manual 코드 기준 파일명
img_path = os.path.join(dir_output, "02_a4_warped.png")

# 만약 네 결과 파일명이 03_a4_warped.png라면 위 줄 대신 아래 줄 사용
# img_path = os.path.join(dir_output, "03_a4_warped.png")

print("Input image:", img_path)

img = cv2.imread(img_path)

if img is None:
    print("Error: Could not open warped A4 image")
    print("Check this path:", img_path)
    exit()

# ============================
# 1. 그림자 노이즈 추가
# ============================
shadow_added = add_shadow_noise(img)

# ============================
# 2. 그림자 제거
# ============================
removed_gray, background = remove_shadow_gray(shadow_added)
removed_binary = make_binary_document(removed_gray)
removed_color = remove_shadow_color(shadow_added)

# ============================
# 결과 저장
# ============================
cv2.imwrite(os.path.join(dir_output, "04_original_warped.png"), img)
cv2.imwrite(os.path.join(dir_output, "05_shadow_added.png"), shadow_added)
cv2.imwrite(os.path.join(dir_output, "06_background_estimate.png"), background)
cv2.imwrite(os.path.join(dir_output, "07_shadow_removed_gray.png"), removed_gray)
cv2.imwrite(os.path.join(dir_output, "08_shadow_removed_binary.png"), removed_binary)
cv2.imwrite(os.path.join(dir_output, "09_shadow_removed_color.png"), removed_color)

print("저장 완료:")
print("04_original_warped.png")
print("05_shadow_added.png")
print("06_background_estimate.png")
print("07_shadow_removed_gray.png")
print("08_shadow_removed_binary.png")
print("09_shadow_removed_color.png")

# ============================
# 화면 출력
# ============================
cv2.imshow("Original Warped A4", img)
cv2.imshow("Shadow Added", shadow_added)
cv2.imshow("Background Estimate", background)
cv2.imshow("Shadow Removed Gray", removed_gray)
cv2.imshow("Shadow Removed Binary", removed_binary)
cv2.imshow("Shadow Removed Color", removed_color)

cv2.waitKey(0)
cv2.destroyAllWindows()