import argparse
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np


DEFAULT_TARGET_POINTS = np.array(
    [
        [134, 145],
        [424, 133],
        [546, 340],
        [45, 338],
    ],
    dtype=np.float32,
)

clicked_points = []
display_image = None
display_scale = 1.0


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

    lines = [line[0].astype(np.float32) for line in lines]
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
    default_image = script_dir.parent.parent.parent / "target_image.jpg"
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


def main():
    args = build_parser().parse_args()
    image_path = Path(args.image).resolve()
    output_root = Path(args.output).resolve()
    save_dir = output_root / image_path.stem
    save_dir.mkdir(parents=True, exist_ok=True)

    image = imread_unicode(image_path)
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    imwrite_unicode(save_dir / "00_loaded_input.png", image)

    if args.points:
        points = parse_points(args.points)
    elif args.manual:
        points = select_points_manually(image)
    else:
        points = DEFAULT_TARGET_POINTS.copy()

    selected = draw_points(image, points)
    warped_full, perspective_matrix, ordered_points = warp_by_four_points(
        image, points, output_width=args.width
    )
    warped, warp_bbox = trim_valid_region(
        warped_full,
        border=args.crop_border,
    )

    try:
        x_angle, y_angle, edges = estimate_grid_axes(warped)
    except RuntimeError:
        x_angle, y_angle, edges = estimate_grid_axes(warped_full)
    rectified_full, affine_matrix = affine_rectify_axes(warped, x_angle, y_angle)
    rectified, rect_bbox = trim_valid_region(
        rectified_full,
        border=args.crop_border,
    )

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

    imwrite_unicode(save_dir / "01_selected_corners.png", selected)
    imwrite_unicode(save_dir / "02_perspective_warp.png", warped_full)
    imwrite_unicode(save_dir / "03_trimmed_warp.png", warped)
    imwrite_unicode(save_dir / "04_hough_edges.png", edges)
    imwrite_unicode(save_dir / "05_axis_rectified.png", rectified_full)
    imwrite_unicode(save_dir / "06_trimmed_rectified.png", rectified)
    imwrite_unicode(save_dir / "07_profiles.png", profile_plot)
    imwrite_unicode(save_dir / "08_count_overlay.png", overlay)

    result = {
        "image": str(image_path),
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
        "warped_shape": list(warped_full.shape[:2]),
        "trimmed_warp_shape": list(warped.shape[:2]),
        "rectified_shape": list(rectified_full.shape[:2]),
        "trimmed_rectified_shape": list(rectified.shape[:2]),
        "x_peak_rows": [int(index) for _, index in x_peaks],
        "y_peak_cols": [int(index) for _, index in y_peaks],
    }

    with open(save_dir / "09_result.json", "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    print(f"Input: {image_path}")
    print(f"Output: {save_dir}")
    print(f"X-axis direction count: {len(x_peaks)}")
    print(f"Y-axis direction count: {len(y_peaks)}")


if __name__ == "__main__":
    main()
