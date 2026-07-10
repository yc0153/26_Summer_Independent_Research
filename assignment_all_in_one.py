import os
import cv2
import numpy as np

points = []
display_img = None
scale = 1.0


def imread_unicode(path):
    """
    한글 경로에서도 이미지가 읽히도록 하는 함수
    """
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def imwrite_unicode(path, img):
    """
    한글 경로에서도 이미지가 저장되도록 하는 함수
    """
    ext = os.path.splitext(path)[1]
    result, encoded_img = cv2.imencode(ext, img)

    if result:
        encoded_img.tofile(path)
        return True

    return False


def mouse_callback(event, x, y, flags, param):
    """
    마우스로 A4 꼭짓점 4개 선택
    """
    global points, display_img, scale

    window_name = param

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) >= 4:
            print("이미 꼭짓점 4개를 모두 선택했습니다. 아무 키나 누르세요.")
            return

        original_x = int(x / scale)
        original_y = int(y / scale)

        points.append([original_x, original_y])

        cv2.circle(display_img, (x, y), 8, (0, 0, 255), -1)
        cv2.putText(
            display_img,
            str(len(points)),
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        cv2.imshow(window_name, display_img)
        print(f"{len(points)}번 점 선택: 원본 좌표 ({original_x}, {original_y})")


def order_points(pts):
    """
    네 점을 왼쪽 위, 오른쪽 위, 오른쪽 아래, 왼쪽 아래 순서로 정렬
    """
    pts = np.array(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]       # top-left
    rect[2] = pts[np.argmax(s)]       # bottom-right
    rect[1] = pts[np.argmin(diff)]    # top-right
    rect[3] = pts[np.argmax(diff)]    # bottom-left

    return rect


def warp_a4(img, points):
    """
    클릭한 네 꼭짓점을 이용해 A4를 정면으로 보정
    """
    src = order_points(points)

    output_width = 700
    output_height = int(output_width * 1.414)

    dst = np.float32([
        [0, 0],
        [output_width - 1, 0],
        [output_width - 1, output_height - 1],
        [0, output_height - 1]
    ])

    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, matrix, (output_width, output_height))

    debug = img.copy()

    for i, p in enumerate(src):
        x, y = p.astype(int)
        cv2.circle(debug, (x, y), 15, (0, 0, 255), -1)
        cv2.putText(
            debug,
            str(i + 1),
            (x + 15, y - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 255),
            4
        )

    cv2.polylines(debug, [src.astype(int)], True, (0, 0, 255), 5)

    return warped, debug


def remove_shadow_gray(img):
    """
    펴진 A4 이미지에서 실제 그림자 제거
    그레이스케일 기반 방식
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 조명/그림자 배경 성분 추정
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=65, sigmaY=65)
    background = np.where(background == 0, 1, background)

    # 원본을 배경으로 나누어 조명 보정
    corrected = cv2.divide(gray, background, scale=255)

    # 대비 정규화
    corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)

    return corrected.astype(np.uint8), background.astype(np.uint8)


def remove_shadow_color(img):
    """
    컬러 이미지에서 실제 그림자 제거
    각 채널별로 배경 조명을 추정해서 보정
    """
    channels = cv2.split(img)
    result_channels = []

    for ch in channels:
        background = cv2.GaussianBlur(ch, (0, 0), sigmaX=65, sigmaY=65)
        background = np.where(background == 0, 1, background)

        corrected = cv2.divide(ch, background, scale=255)
        corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)

        result_channels.append(corrected.astype(np.uint8))

    result = cv2.merge(result_channels)

    return result


def make_binary_document(corrected_gray):
    """
    그림자 제거 후 문서처럼 흑백화
    """
    binary = cv2.adaptiveThreshold(
        corrected_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10
    )

    return binary


# ============================
# 경로 설정
# ============================
dir_base = os.path.dirname(os.path.abspath(__file__))
dir_input = os.path.join(dir_base, "input")
dir_output = os.path.join(dir_base, "output")
os.makedirs(dir_output, exist_ok=True)

# 사용할 이미지 선택
img_name = input("사용할 이미지 파일명 입력 예: img1.jpg, img2.jpg, img3.jpg : ").strip()

if img_name == "":
    img_name = "img1.jpg"

img_path = os.path.join(dir_input, img_name)
base_name = os.path.splitext(img_name)[0]

# 이미지별 결과 폴더 생성
save_dir = os.path.join(dir_output, base_name)
os.makedirs(save_dir, exist_ok=True)

print("================================")
print("실행 중인 파일:", __file__)
print("현재 사용하는 이미지:", img_name)
print("전체 이미지 경로:", img_path)
print("결과 저장 폴더:", save_dir)
print("================================")

img = imread_unicode(img_path)

if img is None:
    print("Error: Could not open image")
    print("Check this path:", img_path)
    exit()

# 실제 읽은 원본 저장
imwrite_unicode(os.path.join(save_dir, "00_loaded_input.png"), img)

# ============================
# 1. A4 꼭짓점 직접 선택
# ============================
h, w = img.shape[:2]

max_width = 900
max_height = 900

scale = min(max_width / w, max_height / h, 1.0)

display_w = int(w * scale)
display_h = int(h * scale)

display_img = cv2.resize(img, (display_w, display_h))

cv2.putText(
    display_img,
    img_name,
    (30, 50),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.5,
    (0, 0, 255),
    3
)

print()
print(f"{img_name}의 A4 네 꼭짓점을 클릭하세요.")
print("추천 순서: 왼쪽 위 → 오른쪽 위 → 오른쪽 아래 → 왼쪽 아래")
print("4개 클릭 후 아무 키나 누르면 A4 보정과 그림자 제거가 실행됩니다.")
print()

window_name = f"{img_name} - Select 4 A4 corners"

cv2.imshow(window_name, display_img)
cv2.setMouseCallback(window_name, mouse_callback, window_name)
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(points) != 4:
    print("Error: 꼭짓점 4개를 정확히 클릭해야 합니다.")
    print("현재 클릭한 점 개수:", len(points))
    exit()

# ============================
# 2. A4 정면 보정
# ============================
warped, selected_debug = warp_a4(img, points)

# ============================
# 3. 펴진 A4 이미지에서 실제 그림자 제거
# ============================
removed_gray, background = remove_shadow_gray(warped)
removed_binary = make_binary_document(removed_gray)
removed_color = remove_shadow_color(warped)

# ============================
# 4. 결과 저장
# ============================
imwrite_unicode(os.path.join(save_dir, "01_selected_points.png"), selected_debug)
imwrite_unicode(os.path.join(save_dir, "02_a4_warped.png"), warped)
imwrite_unicode(os.path.join(save_dir, "03_background_estimate.png"), background)
imwrite_unicode(os.path.join(save_dir, "04_shadow_removed_gray.png"), removed_gray)
imwrite_unicode(os.path.join(save_dir, "05_shadow_removed_binary.png"), removed_binary)
imwrite_unicode(os.path.join(save_dir, "06_shadow_removed_color.png"), removed_color)

print()
print("저장 완료:")
print(os.path.join(save_dir, "00_loaded_input.png"))
print(os.path.join(save_dir, "01_selected_points.png"))
print(os.path.join(save_dir, "02_a4_warped.png"))
print(os.path.join(save_dir, "03_background_estimate.png"))
print(os.path.join(save_dir, "04_shadow_removed_gray.png"))
print(os.path.join(save_dir, "05_shadow_removed_binary.png"))
print(os.path.join(save_dir, "06_shadow_removed_color.png"))

# ============================
# 5. 결과 화면 출력
# ============================
cv2.imshow(f"{img_name} - 01 Selected Points", cv2.resize(selected_debug, (display_w, display_h)))
cv2.imshow(f"{img_name} - 02 Warped A4", cv2.resize(warped, (490, 690)))
cv2.imshow(f"{img_name} - 03 Shadow Removed Gray", cv2.resize(removed_gray, (490, 690)))
cv2.imshow(f"{img_name} - 04 Shadow Removed Binary", cv2.resize(removed_binary, (490, 690)))
cv2.imshow(f"{img_name} - 05 Shadow Removed Color", cv2.resize(removed_color, (490, 690)))

cv2.waitKey(0)
cv2.destroyAllWindows()