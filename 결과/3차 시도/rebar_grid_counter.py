import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


DEFAULT_TARGET_POINTS = np.array(
    [
        [104.0, 154.0],
        [371.0, 135.0],
        [453.0, 306.0],
        [124.0, 349.0],
    ],
    dtype=np.float32,
)

HORIZONTAL_COLOR = (0, 0, 255)
VERTICAL_COLOR = (255, 0, 0)
GRID_LINE_THICKNESS = 2

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


def normalize_hough_lines(lines):
    normalized = []
    for line in lines:
        flat = np.asarray(line, dtype=np.float32).reshape(-1)
        if flat.size != 4:
            continue
        normalized.append(flat)
    return normalized


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
    weight_sum = 0.0

    for line in lines:
        angle = line_angle(line)
        if angle_distance(angle, mode) > tolerance:
            continue

        x1, y1, x2, y2 = line
        weight = math.hypot(x2 - x1, y2 - y1)
        radians = math.radians(angle * 2)
        x_sum += weight * math.cos(radians)
        y_sum += weight * math.sin(radians)
        weight_sum += weight

    if weight_sum == 0:
        return float(mode)

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

    lines = normalize_hough_lines(lines)
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


def summarize_clusters(clusters):
    return [
        {
            "center": float(cluster["center"]),
            "support": float(cluster["support"]),
            "segment_count": int(cluster["segment_count"]),
            "position_min": float(cluster["position_min"]),
            "position_max": float(cluster["position_max"]),
        }
        for cluster in clusters
    ]


def smooth_profile(profile, radius=4):
    if radius <= 0:
        return np.asarray(profile, dtype=np.float64)

    kernel = np.ones(2 * radius + 1, dtype=np.float64)
    kernel /= kernel.sum()
    return np.convolve(np.asarray(profile, dtype=np.float64), kernel, mode="same")


def find_profile_peaks(profile, min_distance, max_peaks=20):
    """Keep strong, separated responses as one center per grid bar."""
    profile = smooth_profile(profile, radius=4)
    if len(profile) < 3:
        return []

    baseline = float(np.percentile(profile, 25))
    spread = float(np.percentile(profile, 90) - baseline)
    threshold = baseline + spread * 0.12
    candidates = []

    for index in range(1, len(profile) - 1):
        if profile[index] < threshold:
            continue
        if profile[index] >= profile[index - 1] and profile[index] >= profile[index + 1]:
            candidates.append(index)

    selected = []
    for index in sorted(candidates, key=lambda item: profile[item], reverse=True):
        if all(abs(index - other) >= min_distance for other in selected):
            selected.append(index)
        if len(selected) >= max_peaks:
            break

    selected.sort()
    return selected


def profile_clusters(profile, min_distance, max_peaks=20):
    positions = find_profile_peaks(profile, min_distance, max_peaks=max_peaks)
    return [
        {
            "center": float(position),
            "support": float(profile[position]),
            "segment_count": 1,
            "position_min": float(position),
            "position_max": float(position),
        }
        for position in positions
    ]


def masked_profile(response, mask, axis):
    weights = mask.astype(np.float64)
    response = np.asarray(response, dtype=np.float64)

    if axis == 0:
        numerator = (response * weights).sum(axis=0)
        denominator = weights.sum(axis=0)
    else:
        numerator = (response * weights).sum(axis=1)
        denominator = weights.sum(axis=1)

    return numerator / np.maximum(denominator, 1.0)


def grid_polygon_in_rectified(warped_shape, affine_matrix):
    height, width = warped_shape[:2]
    corners = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    return cv2.transform(corners.reshape(1, -1, 2), affine_matrix)[0]


def build_profile_clusters(rectified, grid_polygon):
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(grid_polygon).astype(np.int32), 1)

    gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    horizontal_response = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (51, 3)),
    )
    vertical_response = np.abs(
        cv2.Sobel(gray_blurred, cv2.CV_32F, 1, 0, ksize=3)
    )

    horizontal_profile = masked_profile(horizontal_response, mask, axis=1)
    vertical_profile = masked_profile(vertical_response, mask, axis=0)

    horizontal_clusters = profile_clusters(
        horizontal_profile,
        min_distance=max(45, rectified.shape[0] // 10),
        max_peaks=12,
    )
    vertical_clusters = profile_clusters(
        vertical_profile,
        min_distance=max(55, rectified.shape[1] // 14),
        max_peaks=14,
    )

    if len(horizontal_clusters) < 2 or len(vertical_clusters) < 2:
        raise RuntimeError("Could not identify both directions of the rebar grid")

    return (
        horizontal_clusters,
        vertical_clusters,
        horizontal_profile,
        vertical_profile,
        mask,
    )


def line_polygon_endpoints(polygon, orientation, position):
    intersections = []
    polygon = np.asarray(polygon, dtype=np.float32)

    for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
        if orientation == "horizontal":
            coordinate_a, coordinate_b = start[1], end[1]
            if abs(float(coordinate_b - coordinate_a)) < 1e-6:
                continue
            if min(coordinate_a, coordinate_b) <= position <= max(coordinate_a, coordinate_b):
                ratio = (position - coordinate_a) / (coordinate_b - coordinate_a)
                intersections.append(float(start[0] + ratio * (end[0] - start[0])))
        else:
            coordinate_a, coordinate_b = start[0], end[0]
            if abs(float(coordinate_b - coordinate_a)) < 1e-6:
                continue
            if min(coordinate_a, coordinate_b) <= position <= max(coordinate_a, coordinate_b):
                ratio = (position - coordinate_a) / (coordinate_b - coordinate_a)
                intersections.append(float(start[1] + ratio * (end[1] - start[1])))

    if len(intersections) < 2:
        return None

    intersections.sort()
    if orientation == "horizontal":
        return np.array(
            [[intersections[0], position], [intersections[-1], position]],
            dtype=np.float32,
        )
    return np.array(
        [[position, intersections[0]], [position, intersections[-1]]],
        dtype=np.float32,
    )


def draw_profile_overlay(
    image,
    horizontal_clusters,
    vertical_clusters,
    grid_polygon,
    annotate=True,
):
    result = image.copy()

    for index, cluster in enumerate(horizontal_clusters, start=1):
        endpoints = line_polygon_endpoints(
            grid_polygon,
            "horizontal",
            cluster["center"],
        )
        if endpoints is None:
            continue
        start, end = np.round(endpoints).astype(np.int32)
        cv2.line(
            result,
            tuple(start),
            tuple(end),
            HORIZONTAL_COLOR,
            GRID_LINE_THICKNESS,
            cv2.LINE_AA,
        )
        if annotate:
            cv2.putText(
                result,
                str(index),
                (int(start[0]) + 5, int(start[1]) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                HORIZONTAL_COLOR,
                2,
            )

    for index, cluster in enumerate(vertical_clusters, start=1):
        endpoints = line_polygon_endpoints(
            grid_polygon,
            "vertical",
            cluster["center"],
        )
        if endpoints is None:
            continue
        start, end = np.round(endpoints).astype(np.int32)
        cv2.line(
            result,
            tuple(start),
            tuple(end),
            VERTICAL_COLOR,
            GRID_LINE_THICKNESS,
            cv2.LINE_AA,
        )
        if annotate:
            cv2.putText(
                result,
                str(index),
                (int(start[0]) + 5, int(start[1]) + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                VERTICAL_COLOR,
                2,
            )

    if annotate:
        cv2.putText(
            result,
            f"Horizontal bars: {len(horizontal_clusters)}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            HORIZONTAL_COLOR,
            2,
        )
        cv2.putText(
            result,
            f"Vertical bars: {len(vertical_clusters)}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            VERTICAL_COLOR,
            2,
        )

    return result


def transform_rectified_lines_to_original(
    original,
    horizontal_clusters,
    vertical_clusters,
    grid_polygon,
    affine_matrix,
    perspective_matrix,
):
    result = original.copy()
    inverse_affine = cv2.invertAffineTransform(affine_matrix)
    inverse_perspective = np.linalg.inv(perspective_matrix).astype(np.float32)

    def draw_transformed_line(orientation, position):
        endpoints = line_polygon_endpoints(grid_polygon, orientation, position)
        if endpoints is None:
            return
        points = endpoints.reshape(1, -1, 2)
        warped_points = cv2.transform(points, inverse_affine)
        original_points = cv2.perspectiveTransform(
            warped_points,
            inverse_perspective,
        )[0]
        start, end = np.round(original_points).astype(np.int32)
        color = HORIZONTAL_COLOR if orientation == "horizontal" else VERTICAL_COLOR
        cv2.line(
            result,
            tuple(start),
            tuple(end),
            color,
            GRID_LINE_THICKNESS,
            cv2.LINE_AA,
        )

    for cluster in horizontal_clusters:
        draw_transformed_line("horizontal", cluster["center"])
    for cluster in vertical_clusters:
        draw_transformed_line("vertical", cluster["center"])

    return result


def build_parser():
    script_dir = Path(__file__).resolve().parent
    default_image = script_dir.parent.parent / "target_image.jpg"
    default_output = script_dir / "output"

    parser = argparse.ArgumentParser(
        description="Rectify and mark a rebar grid using directional profiles."
    )
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
    warped = warped_full
    warp_bbox = (0, 0, warped.shape[1], warped.shape[0])

    x_angle, y_angle, edges = estimate_grid_axes(warped)
    rectified_full, affine_matrix = affine_rectify_axes(warped, x_angle, y_angle)
    rectified = rectified_full
    rect_bbox = (0, 0, rectified.shape[1], rectified.shape[0])
    grid_polygon = grid_polygon_in_rectified(warped.shape, affine_matrix)

    (
        horizontal_clusters,
        vertical_clusters,
        horizontal_profile,
        vertical_profile,
        grid_mask,
    ) = build_profile_clusters(rectified, grid_polygon)
    profile_overlay = draw_profile_overlay(
        rectified,
        horizontal_clusters,
        vertical_clusters,
        grid_polygon,
        annotate=True,
    )
    primary_overlay = transform_rectified_lines_to_original(
        image,
        horizontal_clusters,
        vertical_clusters,
        grid_polygon,
        affine_matrix,
        perspective_matrix,
    )

    primary_method = "directional_profile"
    primary_x_count = len(horizontal_clusters)
    primary_y_count = len(vertical_clusters)

    imwrite_unicode(save_dir / "01_selected_corners.png", selected)
    imwrite_unicode(save_dir / "02_perspective_warp.png", warped_full)
    imwrite_unicode(save_dir / "03_trimmed_warp.png", warped)
    imwrite_unicode(save_dir / "04_hough_edges.png", edges)
    imwrite_unicode(save_dir / "05_axis_rectified.png", rectified_full)
    imwrite_unicode(save_dir / "06_trimmed_rectified.png", rectified)
    imwrite_unicode(save_dir / "07_profile_overlay.png", profile_overlay)
    imwrite_unicode(save_dir / "08_primary_overlay.png", primary_overlay)

    result = {
        "image": str(image_path),
        "primary_method": primary_method,
        "x_axis_direction_count": int(primary_x_count),
        "y_axis_direction_count": int(primary_y_count),
        "ordered_points": ordered_points.tolist(),
        "warp_bbox": list(warp_bbox),
        "rectified_bbox": list(rect_bbox),
        "x_axis_angle_after_warp": float(x_angle),
        "y_axis_angle_after_warp": float(y_angle),
        "perspective_matrix": perspective_matrix.tolist(),
        "affine_matrix": affine_matrix.tolist(),
        "warped_shape": list(warped_full.shape[:2]),
        "trimmed_warp_shape": list(warped.shape[:2]),
        "rectified_shape": list(rectified_full.shape[:2]),
        "trimmed_rectified_shape": list(rectified.shape[:2]),
        "profile": {
            "horizontal_cluster_count": len(horizontal_clusters),
            "vertical_cluster_count": len(vertical_clusters),
            "horizontal_profile_peak_max": float(np.max(horizontal_profile)),
            "vertical_profile_peak_max": float(np.max(vertical_profile)),
            "grid_mask_pixels": int(np.count_nonzero(grid_mask)),
            "horizontal_clusters": summarize_clusters(horizontal_clusters),
            "vertical_clusters": summarize_clusters(vertical_clusters),
        },
    }

    with open(save_dir / "09_result.json", "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    print(f"Input: {image_path}")
    print(f"Output: {save_dir}")
    print(f"Primary method: {primary_method}")
    print(
        f"Grid count: horizontal {len(horizontal_clusters)}, vertical {len(vertical_clusters)}"
    )


if __name__ == "__main__":
    main()
