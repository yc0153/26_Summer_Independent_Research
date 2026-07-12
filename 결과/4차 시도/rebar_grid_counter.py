import argparse
import itertools
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np


# ------------------------------------------------------------
# 4차 시도 핵심 설정
# - Hough 직선 조합을 자동 탐지의 1순위로 사용
# - GrabCut은 보조 수단으로만 남겨 둔다
# - 특정 이미지의 꼭짓점 좌표나 정답 카운트를 사용하지 않는다
# - 이미지 크기에 비례한 값과 후보 사각형의 기하학적 점수만 사용한다
# ------------------------------------------------------------
AUTO_GRABCUT_RECTS = [
    (0.05, 0.12, 0.90, 0.76),
    (0.05, 0.10, 0.90, 0.80),
    (0.03, 0.12, 0.92, 0.76),
    (0.03, 0.10, 0.92, 0.80),
]

AUTO_APPROX_EPSILONS = [0.015, 0.02, 0.025, 0.03, 0.035, 0.04]

# 아래 값들은 특정 사진의 좌표가 아니라, 해상도가 달라도 유지되는
# 일반적인 탐지 규칙이다.
AUTO_LINE_MIN_LENGTH_FRACTION = 0.10
AUTO_FRAME_MARGIN_FRACTION = 0.05
AUTO_LINE_DEDUP_ANGLE_DEG = 3.0
AUTO_LINE_DEDUP_DISTANCE = 8.0
AUTO_MAX_CANDIDATE_LINES = 48
AUTO_POOL_LIMIT = 20
AUTO_MIN_AREA_FRACTION = 0.12
AUTO_MAX_AREA_FRACTION = 0.95
AUTO_MIN_SIDE_FRACTION = 0.08

clicked_points = []
display_image = None
display_scale = 1.0


# ------------------------------------------------------------
# 파일 입출력 / 좌표 처리 / 수동 클릭용 유틸리티
# ------------------------------------------------------------
def imread_unicode(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"Could not encode image: {path}")
    encoded.tofile(str(path))


def order_points(points):
    points = np.asarray(points, dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)

    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1)

    rect[0] = points[np.argmin(sums)]
    rect[1] = points[np.argmin(diffs)]
    rect[2] = points[np.argmax(sums)]
    rect[3] = points[np.argmax(diffs)]
    return rect


def parse_points(raw_points):
    cleaned = raw_points.replace(";", " ").replace("|", " ")
    pairs = cleaned.split()
    points = []

    for pair in pairs:
        if "," not in pair:
            raise ValueError("Each point must look like x,y")
        x_text, y_text = pair.split(",", 1)
        points.append([float(x_text), float(y_text)])

    if len(points) != 4:
        raise ValueError("Exactly four points are required")

    return np.asarray(points, dtype=np.float32)


def mouse_callback(event, x, y, flags, window_name):
    del flags
    global clicked_points, display_image, display_scale

    if event != cv2.EVENT_LBUTTONDOWN or len(clicked_points) >= 4:
        return

    original_x = int(x / display_scale)
    original_y = int(y / display_scale)
    clicked_points.append([original_x, original_y])

    cv2.circle(display_image, (x, y), 6, (0, 0, 255), -1)
    cv2.putText(
        display_image,
        str(len(clicked_points)),
        (x + 8, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
    )
    cv2.imshow(window_name, display_image)


def select_points_manually(image):
    global clicked_points, display_image, display_scale
    clicked_points = []

    height, width = image.shape[:2]
    display_scale = min(1000 / width, 800 / height, 1.0)
    display_size = (int(width * display_scale), int(height * display_scale))
    display_image = cv2.resize(image, display_size)

    window_name = "Select 4 grid corners"
    cv2.imshow(window_name, display_image)
    cv2.setMouseCallback(window_name, mouse_callback, window_name)

    print("Click 4 rebar-grid outer corners, then press any key.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(clicked_points) != 4:
        raise RuntimeError(f"Expected 4 points, got {len(clicked_points)}")

    return np.asarray(clicked_points, dtype=np.float32)


def draw_points(image, points):
    result = image.copy()
    ordered = order_points(points).astype(int)
    cv2.polylines(result, [ordered], True, (0, 0, 255), 2)

    for index, point in enumerate(ordered, start=1):
        x, y = point
        cv2.circle(result, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(
            result,
            str(index),
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
    return result


# ------------------------------------------------------------
# perspective warp: 4개의 꼭짓점을 기준으로 이미지를 정면에 가깝게 펴는 기능
# ------------------------------------------------------------
def warp_by_four_points(image, points, output_width=1100):
    source = order_points(points)
    tl, tr, br, bl = source

    top_width = np.linalg.norm(tr - tl)
    bottom_width = np.linalg.norm(br - bl)
    left_height = np.linalg.norm(bl - tl)
    right_height = np.linalg.norm(br - tr)

    source_width = max(top_width, bottom_width)
    source_height = max(left_height, right_height)
    output_height = int(round(output_width * source_height / max(source_width, 1)))
    output_height = max(output_height, 500)

    destination = np.array(
        [
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(source, destination)
    warped = cv2.warpPerspective(image, matrix, (output_width, output_height))
    return warped, matrix, source


# ------------------------------------------------------------
# GrabCut 기반 보조 분할
# - Hough 직선 탐지가 실패했을 때만 사용하는 안전장치
# - 큰 사각형을 초기값으로 주고 foreground mask를 만든다
# ------------------------------------------------------------
def run_grabcut_foreground_mask(image, rect, iterations=5):
    height, width = image.shape[:2]
    mask = np.zeros((height, width), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(
        image,
        mask,
        rect,
        bgd_model,
        fgd_model,
        iterations,
        cv2.GC_INIT_WITH_RECT,
    )

    foreground = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
    foreground = cv2.morphologyEx(
        foreground * 255,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=2,
    )
    return foreground


# ------------------------------------------------------------
# 윤곽선을 4개의 꼭짓점 사각형으로 바꾸는 보조 함수
# ------------------------------------------------------------
def contour_to_quad(contour):
    if contour is None or len(contour) < 4:
        return None

    perimeter = cv2.arcLength(contour, True)
    sources = [contour, cv2.convexHull(contour)]

    for source in sources:
        for epsilon in AUTO_APPROX_EPSILONS:
            approx = cv2.approxPolyDP(source, epsilon * perimeter, True)
            if len(approx) == 4:
                return approx.reshape(-1, 2).astype(np.float32)

    rect = cv2.minAreaRect(contour)
    return cv2.boxPoints(rect).astype(np.float32)


# ------------------------------------------------------------
# 최종 사각형과 윤곽선을 원본 이미지 위에 표시하는 함수
# ------------------------------------------------------------
def draw_quad_overlay(image, contour, quad):
    result = image.copy()

    if contour is not None:
        cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)

    quad_ordered = order_points(quad).astype(int)
    cv2.polylines(result, [quad_ordered], True, (0, 0, 255), 3)

    for index, point in enumerate(quad_ordered, start=1):
        x, y = point
        cv2.circle(result, (x, y), 8, (0, 0, 255), -1)
        cv2.putText(
            result,
            str(index),
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
        )

    return result


# ------------------------------------------------------------
# Hough 직선 기반 자동 사각형 탐지
# 1) Canny + HoughLinesP로 직선 후보를 모은다
# 2) 이미지 크기에 맞춰 너무 짧거나 테두리인 선을 제거한다
# 3) 거의 같은 Hough 선을 하나로 합친다
# 4) 넓은 위치 조건으로 상/하/좌/우 후보를 만든다
# 5) 네 직선의 교차점으로 사각형을 만들고 경계선 지지도로 평가한다
# ------------------------------------------------------------
def detect_hough_segments(image):
    height, width = image.shape[:2]
    min_dimension = min(height, width)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(35, int(min_dimension * 0.14)),
        minLineLength=max(40, int(min_dimension * AUTO_LINE_MIN_LENGTH_FRACTION)),
        maxLineGap=max(8, int(min_dimension * 0.03)),
    )

    segments = []
    if lines is None:
        return edges, segments

    # OpenCV versions return HoughLinesP as either (N, 1, 4) or (N, 4).
    # Normalize both forms so the web app and CLI behave identically.
    for line in np.asarray(lines).reshape(-1, 4):
        x1, y1, x2, y2 = map(float, line)
        length = math.hypot(x2 - x1, y2 - y1)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if angle < 0:
            angle += 180

        # 입력 이미지 자체의 테두리는 모든 사진에서 생길 수 있는
        # 인공적인 직선이므로, 화면 가장자리의 선은 후보에서 제외한다.
        frame_margin = min_dimension * AUTO_FRAME_MARGIN_FRACTION
        if (
            max(x1, x2) < frame_margin
            or min(x1, x2) > width - frame_margin
            or max(y1, y2) < frame_margin
            or min(y1, y2) > height - frame_margin
        ):
            continue

        segments.append(
            {
                "line": (x1, y1, x2, y2),
                "length": length,
                "angle": angle,
                "mx": (x1 + x2) / 2.0,
                "my": (y1 + y2) / 2.0,
            }
        )

    return edges, deduplicate_hough_segments(segments)


def line_from_segment(segment):
    x1, y1, x2, y2 = segment["line"]
    return np.array([y1 - y2, x2 - x1, x1 * y2 - x2 * y1], dtype=np.float64)


def intersect_lines(line_a, line_b):
    point = np.cross(line_a, line_b)
    if abs(point[2]) < 1e-8:
        return None
    return np.array([point[0] / point[2], point[1] / point[2]], dtype=np.float32)


# 거의 같은 Hough 선을 하나로 줄인다.
# 하나의 철근이나 외곽선에서 양쪽 edge가 여러 번 검출되는 현상을
# 후보 조합 단계에서 줄이기 위한 전처리다.
def deduplicate_hough_segments(segments):
    ordered = sorted(segments, key=lambda segment: segment["length"], reverse=True)
    unique = []

    for segment in ordered:
        line = line_from_segment(segment)
        norm = math.hypot(line[0], line[1])
        if norm < 1e-8:
            continue

        normalized = line / norm
        if normalized[2] < 0:
            normalized = -normalized

        rho = -float(normalized[2])
        is_duplicate = any(
            angle_distance(segment["angle"], saved["angle"])
            <= AUTO_LINE_DEDUP_ANGLE_DEG
            and abs(rho - saved["rho"]) <= AUTO_LINE_DEDUP_DISTANCE
            for saved in unique
        )
        if is_duplicate:
            continue

        saved = dict(segment)
        saved["rho"] = rho
        unique.append(saved)

        if len(unique) >= AUTO_MAX_CANDIDATE_LINES:
            break

    return unique


# Hough 직선 후보를 상/하/좌/우 네 방향으로 좁힌다.
# 픽셀 좌표나 특정 각도 구간 대신 이미지 크기에 대한 상대 위치만 사용한다.
def build_hough_pools(segments, image_shape):
    height, width = image_shape[:2]

    def is_horizontalish(segment):
        angle = math.radians(segment["angle"])
        return abs(math.sin(angle)) <= 0.72

    pools = {
        "top": [
            segment
            for segment in segments
            if segment["my"] < height * 0.60 and is_horizontalish(segment)
        ],
        "bottom": [
            segment
            for segment in segments
            if segment["my"] > height * 0.40 and is_horizontalish(segment)
        ],
        "left": [segment for segment in segments if segment["mx"] < width * 0.60],
        "right": [segment for segment in segments if segment["mx"] > width * 0.40],
    }

    for key in pools:
        pools[key] = sorted(
            pools[key],
            key=lambda segment: segment["length"],
            reverse=True,
        )[:AUTO_POOL_LIMIT]

    return pools


def quad_geometry(quad, width, height):
    ordered = order_points(quad)
    if not np.all(np.isfinite(ordered)):
        return None

    bounds = np.array([width, height], dtype=np.float32)
    if np.any(ordered < -0.15 * bounds) or np.any(ordered > 1.15 * bounds):
        return None

    cross_products = []
    for index in range(4):
        a = ordered[index]
        b = ordered[(index + 1) % 4]
        c = ordered[(index + 2) % 4]
        cross_products.append(
            float((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
        )

    if min(cross_products) * max(cross_products) <= 0:
        return None

    area = abs(float(cv2.contourArea(ordered.astype(np.float32))))
    area_fraction = area / max(width * height, 1)
    if not AUTO_MIN_AREA_FRACTION <= area_fraction <= AUTO_MAX_AREA_FRACTION:
        return None

    side_lengths = [
        float(np.linalg.norm(ordered[(index + 1) % 4] - ordered[index]))
        for index in range(4)
    ]
    diagonal = math.hypot(width, height)
    if min(side_lengths) < diagonal * AUTO_MIN_SIDE_FRACTION:
        return None

    return ordered, area, area_fraction, side_lengths


# 사각형의 네 변이 실제 edge를 얼마나 잘 따라가는지 계산한다.
def quad_edge_support(edge_band, quad):
    supports = []

    for index in range(4):
        start = quad[index]
        end = quad[(index + 1) % 4]
        sample_count = 40
        xs = np.linspace(start[0], end[0], sample_count).round().astype(int)
        ys = np.linspace(start[1], end[1], sample_count).round().astype(int)
        valid = (xs >= 0) & (xs < edge_band.shape[1]) & (ys >= 0) & (ys < edge_band.shape[0])
        if not np.any(valid):
            supports.append(0.0)
            continue
        supports.append(float(edge_band[ys[valid], xs[valid]].mean() / 255.0))

    return float(np.mean(supports)), float(np.min(supports))


def quad_interior_line_score(segments, quad, width, height):
    total_length = 0.0
    for segment in segments:
        point = (float(segment["mx"]), float(segment["my"]))
        if cv2.pointPolygonTest(quad.astype(np.float32), point, False) >= 0:
            total_length += segment["length"]

    diagonal = math.hypot(width, height)
    return min(1.0, total_length / max(diagonal * 18.0, 1.0))


# 네 변의 edge 지지도, 사각형 면적, 내부 직선 구조를 조합한다.
# 특정 사진의 목표 꼭짓점과의 거리는 전혀 사용하지 않는다.
def score_opening_quad(quad, edge_band, segments, width, height):
    geometry = quad_geometry(quad, width, height)
    if geometry is None:
        return None

    ordered, area, area_fraction, side_lengths = geometry
    edge_mean, edge_min = quad_edge_support(edge_band, ordered)
    interior_line_score = quad_interior_line_score(segments, ordered, width, height)

    center = ordered.mean(axis=0)
    image_center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
    center_penalty = float(np.linalg.norm(center - image_center) / max(math.hypot(width, height), 1.0))

    score = (
        edge_mean * 4.0
        + edge_min * 0.8
        # 같은 정도의 edge 지지도를 가진 후보라면, 내부 격자를 더 넓게
        # 포함하는 후보를 우선한다. 이 값은 특정 이미지의 면적이 아니라
        # 모든 입력에 공통으로 적용되는 일반적인 구조 점수다.
        + area_fraction * 3.0
        + interior_line_score * 0.7
        - center_penalty * 0.15
    )
    return {
        "quad": ordered,
        "score": float(score),
        "area": area,
        "area_fraction": area_fraction,
        "side_lengths": side_lengths,
        "edge_support_mean": edge_mean,
        "edge_support_min": edge_min,
        "interior_line_score": interior_line_score,
        "center_penalty": center_penalty,
    }


# 상/하/좌/우 직선 조합으로 후보 사각형을 만든 뒤
# 경계선 지지도와 기하학적 조건으로 가장 자연스러운 후보를 고른다.
def detect_opening_quad_from_hough(image):
    height, width = image.shape[:2]
    edges, segments = detect_hough_segments(image)

    if not segments:
        raise RuntimeError("No Hough line segments found")

    pools = build_hough_pools(segments, image.shape)
    if min(len(pool) for pool in pools.values()) == 0:
        raise RuntimeError("Not enough Hough candidates for opening detection")

    edge_band = cv2.dilate(edges, np.ones((5, 5), dtype=np.uint8), iterations=1)
    candidates = []
    for top, bottom, left, right in itertools.product(
        pools["top"], pools["bottom"], pools["left"], pools["right"]
    ):
        top_line = line_from_segment(top)
        bottom_line = line_from_segment(bottom)
        left_line = line_from_segment(left)
        right_line = line_from_segment(right)

        points = [
            intersect_lines(top_line, left_line),
            intersect_lines(top_line, right_line),
            intersect_lines(bottom_line, right_line),
            intersect_lines(bottom_line, left_line),
        ]
        if any(point is None for point in points):
            continue

        scored = score_opening_quad(
            np.array(points, dtype=np.float32),
            edge_band,
            segments,
            width,
            height,
        )
        if scored is None:
            continue

        scored.update(
            {
                "top": top,
                "bottom": bottom,
                "left": left,
                "right": right,
            }
        )
        candidates.append(scored)

    if not candidates:
        raise RuntimeError("Hough opening detection failed")

    best = max(candidates, key=lambda candidate: candidate["score"])
    return best["quad"], edges, {
        "quad_detection_method": "hough",
        "quad_detection_score": best["score"],
        "quad_score_components": {
            key: best[key]
            for key in [
                "area_fraction",
                "edge_support_mean",
                "edge_support_min",
                "interior_line_score",
                "center_penalty",
            ]
        },
        "selected_lines": {
            "top": [float(v) for v in best["top"]["line"]],
            "bottom": [float(v) for v in best["bottom"]["line"]],
            "left": [float(v) for v in best["left"]["line"]],
            "right": [float(v) for v in best["right"]["line"]],
        },
    }


# Hough 방식이 실패하면 GrabCut으로 한 번 더 시도한다.
# 완전 자동을 유지하면서도 실패 확률을 낮추기 위한 안전장치다.
def auto_detect_outer_quad(image):
    try:
        quad, debug_image, debug_info = detect_opening_quad_from_hough(image)
        return quad, debug_image, debug_info
    except RuntimeError:
        pass

    height, width = image.shape[:2]
    best = None

    for left_frac, top_frac, right_frac, bottom_frac in AUTO_GRABCUT_RECTS:
        rect = (
            int(width * left_frac),
            int(height * top_frac),
            int(width * right_frac),
            int(height * bottom_frac),
        )

        mask = run_grabcut_foreground_mask(image, rect)
        contours, _ = cv2.findContours(
            image=mask,
            mode=cv2.RETR_EXTERNAL,
            method=cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        quad = contour_to_quad(contour)
        if quad is None or len(quad) != 4:
            continue

        ordered = order_points(quad)
        area = cv2.contourArea(ordered.astype(np.float32))
        score = area

        if best is None or score > best["score"]:
            best = {
                "quad": ordered,
                "debug_image": mask,
                "rect": rect,
                "score": score,
            }

    if best is None:
        raise RuntimeError("Automatic quad detection failed")

    return best["quad"], best["debug_image"], {
        "quad_detection_method": "grabcut",
        "quad_detection_score": best["score"],
        "grabcut_rect": list(best["rect"]),
    }


# perspective warp 뒤 검은 영역이 생기면,
# 실제 유효한 화면만 남기기 위해 테두리를 잘라낸다.
def trim_valid_region(image, border=18, threshold=8):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    valid = gray > threshold
    ys, xs = np.where(valid)

    if len(xs) == 0 or len(ys) == 0:
        h, w = image.shape[:2]
        return image.copy(), (0, 0, w, h)

    x0 = max(0, int(xs.min()) - border)
    y0 = max(0, int(ys.min()) - border)
    x1 = min(image.shape[1] - 1, int(xs.max()) + border)
    y1 = min(image.shape[0] - 1, int(ys.max()) + border)

    cropped = image[y0 : y1 + 1, x0 : x1 + 1].copy()
    return cropped, (x0, y0, x1 - x0 + 1, y1 - y0 + 1)


# 워핑된 이미지에서 철근 방향을 다시 찾기 위한 각도 관련 유틸리티
def line_angle(line):
    x1, y1, x2, y2 = line
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    if angle < 0:
        angle += 180
    return angle


def angle_distance(angle_a, angle_b):
    return abs((angle_a - angle_b + 90) % 180 - 90)


def weighted_mean_angle(lines, mode, tolerance=15):
    x_sum = 0.0
    y_sum = 0.0

    for line in lines:
        angle = line_angle(line)
        if angle_distance(angle, mode) > tolerance:
            continue

        x1, y1, x2, y2 = line
        weight = math.hypot(x2 - x1, y2 - y1)
        radians = math.radians(angle * 2)
        x_sum += weight * math.cos(radians)
        y_sum += weight * math.sin(radians)

    mean = 0.5 * math.degrees(math.atan2(y_sum, x_sum))
    if mean < 0:
        mean += 180
    return mean


# 워핑 후 Hough 선들의 각도 히스토그램을 보고
# x축 방향 / y축 방향의 평균 각도를 추정한다.
def estimate_grid_axes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blur = cv2.GaussianBlur(enhanced, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)

    min_line_length = max(60, min(image.shape[:2]) // 8)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=70,
        minLineLength=min_line_length,
        maxLineGap=20,
    )

    if lines is None:
        raise RuntimeError("Could not find enough grid lines for axis estimation")

    # OpenCV 4/5 differ in whether the singleton dimension is present.
    lines = [line.astype(np.float32) for line in np.asarray(lines).reshape(-1, 4)]
    histogram = np.zeros(180, dtype=np.float64)

    for line in lines:
        angle = line_angle(line)
        x1, y1, x2, y2 = line
        histogram[int(round(angle)) % 180] += math.hypot(x2 - x1, y2 - y1)

    kernel = np.ones(11, dtype=np.float64) / 11
    smooth = np.convolve(
        np.r_[histogram[-5:], histogram, histogram[:5]],
        kernel,
        mode="same",
    )[5:-5]

    modes = []
    working = smooth.copy()
    for _ in range(4):
        mode = int(np.argmax(working))
        modes.append(mode)
        for angle in range(180):
            if angle_distance(angle, mode) < 25:
                working[angle] = 0

    mean_angles = [weighted_mean_angle(lines, mode) for mode in modes]
    x_index = min(
        range(len(mean_angles)),
        key=lambda i: min(mean_angles[i], 180 - mean_angles[i]),
    )

    candidates = [
        i
        for i in range(len(mean_angles))
        if i != x_index and angle_distance(mean_angles[i], mean_angles[x_index]) > 30
    ]
    if not candidates:
        raise RuntimeError("Could not separate x/y grid directions")

    y_index = candidates[0]
    return mean_angles[x_index], mean_angles[y_index], edges


# 두 축이 완전히 직교하지 않더라도,
# affine 변환으로 최대한 수평/수직에 가깝게 맞춰 준다.
def affine_rectify_axes(image, x_angle, y_angle):
    x_direction = np.array(
        [math.cos(math.radians(x_angle)), math.sin(math.radians(x_angle))],
        dtype=np.float32,
    )
    y_direction = np.array(
        [math.cos(math.radians(y_angle)), math.sin(math.radians(y_angle))],
        dtype=np.float32,
    )

    basis = np.column_stack([x_direction, y_direction]).astype(np.float32)
    transform = np.linalg.inv(basis).astype(np.float32)

    height, width = image.shape[:2]
    corners = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    transformed = (transform @ corners.T).T
    min_xy = transformed.min(axis=0)
    max_xy = transformed.max(axis=0)

    output_width = int(math.ceil(max_xy[0] - min_xy[0]))
    output_height = int(math.ceil(max_xy[1] - min_xy[1]))
    matrix = np.hstack([transform, -min_xy.reshape(2, 1)]).astype(np.float32)

    rectified = cv2.warpAffine(
        image,
        matrix,
        (output_width, output_height),
        flags=cv2.INTER_LINEAR,
        borderValue=(0, 0, 0),
    )
    return rectified, matrix


# profile에서 주변보다 확실히 강한 봉우리를 골라낸다.
# peak끼리 너무 가까우면 같은 철근으로 보고 중복을 막는다.
def find_profile_peaks(profile, min_distance, percentile=65, relative=0.25):
    values = profile[profile > 0]
    if len(values) == 0:
        return [], 0.0

    threshold = max(np.percentile(values, percentile), values.max() * relative)
    candidates = []

    for index in range(1, len(profile) - 1):
        current = profile[index]
        if current < threshold:
            continue
        if current >= profile[index - 1] and current >= profile[index + 1]:
            candidates.append((float(current), index))

    candidates.sort(reverse=True)
    kept = []

    for strength, index in candidates:
        if all(abs(index - kept_index) >= min_distance for _, kept_index in kept):
            kept.append((strength, index))

    kept.sort(key=lambda item: item[1])
    return kept, float(threshold)


# 행/열 방향 gradient를 누적해서 철근 위치가 두드러지는 profile을 만든다.
def make_profiles(rectified):
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    valid_mask = (gray > 8).astype(np.uint8) * 255
    valid_mask = cv2.erode(
        valid_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)),
        iterations=1,
    )
    valid = valid_mask > 0

    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

    gradient_x = np.abs(cv2.Sobel(enhanced, cv2.CV_32F, 1, 0, ksize=3))
    gradient_y = np.abs(cv2.Sobel(enhanced, cv2.CV_32F, 0, 1, ksize=3))

    row_counts = valid.sum(axis=1).astype(np.float32)
    col_counts = valid.sum(axis=0).astype(np.float32)

    x_profile = (gradient_y * valid).sum(axis=1) / np.maximum(row_counts, 1)
    y_profile = (gradient_x * valid).sum(axis=0) / np.maximum(col_counts, 1)

    x_profile = cv2.GaussianBlur(
        x_profile.reshape(-1, 1).astype(np.float32), (1, 25), 0
    ).ravel()
    y_profile = cv2.GaussianBlur(
        y_profile.reshape(-1, 1).astype(np.float32), (1, 25), 0
    ).ravel()

    x_profile[row_counts < row_counts.max() * 0.15] = 0
    y_profile[col_counts < col_counts.max() * 0.15] = 0
    return x_profile, y_profile


# 검출된 peak를 실제 이미지 위에 그려서
# 어떤 위치가 카운트되었는지 한눈에 확인한다.
def draw_count_overlay(rectified, x_peaks, y_peaks):
    result = rectified.copy()
    height, width = result.shape[:2]

    for _, row in x_peaks:
        cv2.line(result, (0, row), (width - 1, row), (0, 0, 255), 2)

    for _, col in y_peaks:
        cv2.line(result, (col, 0), (col, height - 1), (255, 0, 0), 2)

    cv2.putText(
        result,
        f"X-axis direction: {len(x_peaks)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2,
    )
    cv2.putText(
        result,
        f"Y-axis direction: {len(y_peaks)}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 0, 0),
        2,
    )
    return result


# profile 모양과 threshold선을 함께 보여주는 디버그용 그래프
def draw_profiles(x_profile, y_profile, x_threshold, y_threshold):
    height = 220
    width = max(len(x_profile), len(y_profile), 1)
    canvas = np.full((height * 2, width, 3), 255, dtype=np.uint8)

    for profile, offset, color, threshold in [
        (x_profile, 0, (0, 0, 255), x_threshold),
        (y_profile, height, (255, 0, 0), y_threshold),
    ]:
        max_value = float(profile.max()) + 1e-6
        scaled = profile / max_value * (height - 20)

        for index in range(1, len(profile)):
            cv2.line(
                canvas,
                (index - 1, offset + height - 10 - int(scaled[index - 1])),
                (index, offset + height - 10 - int(scaled[index])),
                color,
                1,
            )

        threshold_y = offset + height - 10 - int(threshold / max_value * (height - 20))
        cv2.line(canvas, (0, threshold_y), (len(profile) - 1, threshold_y), (0, 150, 0), 1)

    return canvas


def build_parser():
    script_dir = Path(__file__).resolve().parent
    default_image = script_dir.parent.parent.parent / "input.jpg"
    default_output = script_dir / "output"

    parser = argparse.ArgumentParser(description="Rectify and count a rebar grid image.")
    parser.add_argument("--image", default=str(default_image), help="Input image path")
    parser.add_argument("--output", default=str(default_output), help="Output directory")
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Click four rebar-grid corners instead of using the target preset",
    )
    parser.add_argument(
        "--points",
        default=None,
        help='Four points, e.g. "145,142 390,135 550,386 32,373"',
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1100,
        help="Perspective-warp output width",
    )
    parser.add_argument(
        "--crop-border",
        type=int,
        default=18,
        help="Extra border to keep when trimming black regions after warp",
    )
    parser.add_argument(
        "--x-min-distance",
        type=int,
        default=None,
        help="Minimum pixel distance between x-axis-direction bars",
    )
    parser.add_argument(
        "--y-min-distance",
        type=int,
        default=None,
        help="Minimum pixel distance between y-axis-direction bars",
    )
    parser.add_argument(
        "--peak-percentile",
        type=float,
        default=65.0,
        help="Peak threshold percentile",
    )
    return parser


# ------------------------------------------------------------
# 전체 실행 순서
# 1) 입력 이미지 로드
# 2) 외곽 사각형 자동 탐지
# 3) perspective warp
# 4) 축 정렬
# 5) profile 기반 카운트
# 6) 결과 이미지와 JSON 저장
# ------------------------------------------------------------
def main():
    args = build_parser().parse_args()
    image_path = Path(args.image).resolve()
    output_root = Path(args.output).resolve()
    save_dir = output_root / image_path.stem
    save_dir.mkdir(parents=True, exist_ok=True)

    image = imread_unicode(image_path)
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    # 원본 이미지를 가장 먼저 저장해 두면, 나중에 비교하기 쉽다.
    imwrite_unicode(save_dir / "00_loaded_input.png", image)

    detection_mode = "auto"
    detection_debug_image = None
    detection_info = {}

    # 수동 입력, 좌표 문자열 입력, 완전 자동 중에서 실행 방식을 고른다.
    if args.points:
        points = parse_points(args.points)
        detection_mode = "manual_points"
    elif args.manual:
        points = select_points_manually(image)
        detection_mode = "manual"
    else:
        points, detection_debug_image, detection_info = auto_detect_outer_quad(image)

    # 선택된 4점을 원본 이미지 위에 표시한다.
    selected = draw_points(image, points)

    if detection_mode == "auto":
        # 자동 모드에서는 Hough/GrabCut 탐지 결과 디버그 이미지를 따로 남긴다.
        if detection_debug_image is not None:
            imwrite_unicode(save_dir / "01_detection_debug.png", detection_debug_image)
        selected_save_name = "02_selected_quad.png"
        warp_save_name = "03_perspective_warp.png"
        trimmed_warp_save_name = "04_trimmed_warp.png"
        hough_edges_save_name = "05_hough_edges.png"
        rectified_save_name = "06_axis_rectified.png"
        trimmed_rectified_save_name = "07_trimmed_rectified.png"
        profiles_save_name = "08_profiles.png"
        overlay_save_name = "09_count_overlay.png"
        result_save_name = "10_result.json"
    else:
        selected_save_name = "01_selected_corners.png"
        warp_save_name = "02_perspective_warp.png"
        trimmed_warp_save_name = "03_trimmed_warp.png"
        hough_edges_save_name = "04_hough_edges.png"
        rectified_save_name = "05_axis_rectified.png"
        trimmed_rectified_save_name = "06_trimmed_rectified.png"
        profiles_save_name = "07_profiles.png"
        overlay_save_name = "08_count_overlay.png"
        result_save_name = "09_result.json"

    imwrite_unicode(save_dir / selected_save_name, selected)

    # 선택된 4점을 기준으로 이미지를 정면에 가깝게 펴기
    warped_full, perspective_matrix, ordered_points = warp_by_four_points(
        image, points, output_width=args.width
    )
    warped, warp_bbox = trim_valid_region(
        warped_full,
        border=args.crop_border,
    )

    # 워핑된 화면에서 철근 방향을 다시 추정
    try:
        x_angle, y_angle, edges = estimate_grid_axes(warped)
    except RuntimeError:
        x_angle, y_angle, edges = estimate_grid_axes(warped_full)

    # 축이 아주 조금 기울어져 있어도 카운트가 안정되도록 보정
    rectified_full, affine_matrix = affine_rectify_axes(warped, x_angle, y_angle)
    rectified, rect_bbox = trim_valid_region(
        rectified_full,
        border=args.crop_border,
    )

    # 행/열 profile을 만든 뒤 peak를 찾아 철근 개수를 센다.
    x_profile, y_profile = make_profiles(rectified)
    x_min_distance = args.x_min_distance or max(45, int(rectified.shape[0] * 0.09))
    y_min_distance = args.y_min_distance or max(45, int(rectified.shape[1] * 0.05))

    x_peaks, x_threshold = find_profile_peaks(
        x_profile,
        min_distance=x_min_distance,
        percentile=args.peak_percentile,
    )
    y_peaks, y_threshold = find_profile_peaks(
        y_profile,
        min_distance=y_min_distance,
        percentile=args.peak_percentile,
    )

    overlay = draw_count_overlay(rectified, x_peaks, y_peaks)
    profile_plot = draw_profiles(x_profile, y_profile, x_threshold, y_threshold)

    # 중간 결과를 순서대로 저장
    imwrite_unicode(save_dir / warp_save_name, warped_full)
    imwrite_unicode(save_dir / trimmed_warp_save_name, warped)
    imwrite_unicode(save_dir / hough_edges_save_name, edges)
    imwrite_unicode(save_dir / rectified_save_name, rectified_full)
    imwrite_unicode(save_dir / trimmed_rectified_save_name, rectified)
    imwrite_unicode(save_dir / profiles_save_name, profile_plot)
    imwrite_unicode(save_dir / overlay_save_name, overlay)

    result = {
        "image": str(image_path),
        "detection_mode": detection_mode,
        "quad_detection_method": detection_info.get("quad_detection_method"),
        "quad_detection_score": detection_info.get("quad_detection_score"),
        "quad_score_components": detection_info.get("quad_score_components"),
        "x_axis_direction_count": len(x_peaks),
        "y_axis_direction_count": len(y_peaks),
        "ordered_points": ordered_points.tolist(),
        "warp_bbox": list(warp_bbox),
        "rectified_bbox": list(rect_bbox),
        "x_axis_angle_after_warp": x_angle,
        "y_axis_angle_after_warp": y_angle,
        "x_min_distance": x_min_distance,
        "y_min_distance": y_min_distance,
        "crop_border": args.crop_border,
        "peak_percentile": args.peak_percentile,
        "perspective_matrix": perspective_matrix.tolist(),
        "affine_matrix": affine_matrix.tolist(),
        "grabcut_rect": detection_info.get("grabcut_rect"),
        "selected_lines": detection_info.get("selected_lines"),
        "warped_shape": list(warped_full.shape[:2]),
        "trimmed_warp_shape": list(warped.shape[:2]),
        "rectified_shape": list(rectified_full.shape[:2]),
        "trimmed_rectified_shape": list(rectified.shape[:2]),
        "x_peak_rows": [int(index) for _, index in x_peaks],
        "y_peak_cols": [int(index) for _, index in y_peaks],
    }

    # 나중에 실험을 다시 보거나 비교하기 쉽도록 핵심 수치도 JSON에 넣는다.
    with open(save_dir / result_save_name, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    print(f"Input: {image_path}")
    print(f"Output: {save_dir}")
    print(f"Detection mode: {detection_mode}")
    if detection_info.get("quad_detection_method"):
        print(f"Quad detection method: {detection_info['quad_detection_method']}")
    if detection_info.get("quad_detection_score") is not None:
        print(f"Quad detection score: {detection_info['quad_detection_score']:.2f}")
    print(f"Outer quad: {ordered_points.tolist()}")
    print(f"X-axis direction count: {len(x_peaks)}")
    print(f"Y-axis direction count: {len(y_peaks)}")


if __name__ == "__main__":
    main()
