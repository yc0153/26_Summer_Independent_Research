import os
import cv2
import numpy as np

points = []
display_img = None
scale = 1.0


def mouse_callback(event, x, y, flags, param):
    global points, display_img, scale

    if event == cv2.EVENT_LBUTTONDOWN:
        # 화면에 축소되어 표시된 좌표를 원본 이미지 좌표로 변환
        original_x = int(x / scale)
        original_y = int(y / scale)

        points.append([original_x, original_y])

        # 표시용 이미지에는 클릭한 위치 표시
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

        cv2.imshow("Select 4 A4 corners", display_img)

        print(f"{len(points)}번 점 선택: 원본 좌표 ({original_x}, {original_y})")


def order_points(pts):
    """
    4개의 점을 왼쪽 위, 오른쪽 위, 오른쪽 아래, 왼쪽 아래 순서로 정렬
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


# ============================
# 경로 설정
# ============================
dir_base = os.path.dirname(os.path.abspath(__file__))
dir_input = os.path.join(dir_base, "input")
dir_output = os.path.join(dir_base, "output")
os.makedirs(dir_output, exist_ok=True)

# 사용할 이미지 파일 이름
img_name = "img3.jpg"
img_path = os.path.join(dir_input, img_name)

print("Base path:", dir_base)
print("Input image path:", img_path)
print("Output path:", dir_output)

img = cv2.imread(img_path)

if img is None:
    print("Error: Could not open image")
    print("Check this path:", img_path)
    exit()

# ============================
# 이미지 화면 크기에 맞게 축소 표시
# ============================
h, w = img.shape[:2]

max_width = 900
max_height = 900

scale = min(max_width / w, max_height / h, 1.0)

display_w = int(w * scale)
display_h = int(h * scale)

display_img = cv2.resize(img, (display_w, display_h))

print()
print("A4의 네 꼭짓점을 클릭하세요.")
print("추천 순서: 왼쪽 위 → 오른쪽 위 → 오른쪽 아래 → 왼쪽 아래")
print("4개를 클릭한 뒤 아무 키나 누르면 변환됩니다.")
print()

cv2.imshow("Select 4 A4 corners", display_img)
cv2.setMouseCallback("Select 4 A4 corners", mouse_callback)

cv2.waitKey(0)
cv2.destroyAllWindows()

if len(points) != 4:
    print("Error: 꼭짓점 4개를 정확히 클릭해야 합니다.")
    print("현재 클릭한 점 개수:", len(points))
    exit()

# ============================
# Perspective Transform
# ============================
src = order_points(points)

# A4 비율: 1 : 1.414
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

# ============================
# 결과 저장
# ============================
debug = img.copy()

# 선택한 점 표시
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

# 선택된 네 점을 선으로 연결
src_int = src.astype(int)
cv2.polylines(debug, [src_int], True, (0, 0, 255), 5)

cv2.imwrite(os.path.join(dir_output, "01_selected_points.png"), debug)
cv2.imwrite(os.path.join(dir_output, "02_a4_warped.png"), warped)

print("저장 완료:")
print(os.path.join(dir_output, "01_selected_points.png"))
print(os.path.join(dir_output, "02_a4_warped.png"))

# ============================
# 결과 확인
# ============================
debug_show = cv2.resize(debug, (display_w, display_h))
warped_show = cv2.resize(warped, (int(output_width * 0.7), int(output_height * 0.7)))

cv2.imshow("Selected Points", debug_show)
cv2.imshow("Warped A4", warped_show)
cv2.waitKey(0)
cv2.destroyAllWindows()