from __future__ import annotations
import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import cv2
import numpy as np


MAX_STEERING_ANGLE_DEG = 25.0
STEERING_EMA_ALPHA = 0.35
_prev_steering_angle_deg: Optional[float] = None

# 基于你当前样例图的黄线范围，留出一定冗余提升光照鲁棒性
HSV_YELLOW_LOWER_A = np.array([15, 80, 80], dtype=np.uint8)
HSV_YELLOW_UPPER_A = np.array([40, 255, 255], dtype=np.uint8)
HSV_YELLOW_LOWER_B = np.array([18, 40, 160], dtype=np.uint8)
HSV_YELLOW_UPPER_B = np.array([38, 180, 255], dtype=np.uint8)


@dataclass
class LaneLine:
    """用两点表示一条车道边界线（用于可视化）。"""

    x1: int
    y1: int
    x2: int
    y2: int
    slope: float


@dataclass
class RoadAngleResult:
    """道路转弯角估计结果。"""

    steering_angle_deg: float
    raw_steering_angle_deg: float
    lane_center_x: Optional[float]
    lane_center_bottom_x: Optional[float]
    image_center_x: float
    left_lane: Optional[LaneLine]
    right_lane: Optional[LaneLine]
    lane_center_path: list[tuple[int, int]]
    predicted_path: list[tuple[int, int]]
    actual_straight_path: list[tuple[int, int]]
    left_lane_curve: list[tuple[int, int]]
    right_lane_curve: list[tuple[int, int]]
    lane_status: str
    lane_confidence: float


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def smooth_steering_angle(current_angle: float) -> float:
    """EMA平滑，抑制帧间抖动（单图像时等价于首次值）。"""
    global _prev_steering_angle_deg
    if _prev_steering_angle_deg is None:
        _prev_steering_angle_deg = current_angle
        return current_angle

    smoothed = STEERING_EMA_ALPHA * current_angle + (1.0 - STEERING_EMA_ALPHA) * _prev_steering_angle_deg
    _prev_steering_angle_deg = smoothed
    return smoothed


def build_road_roi_mask(height: int, width: int) -> np.ndarray:
    """固定视场下使用梯形ROI，仅保留前方路面。"""
    mask = np.zeros((height, width), dtype=np.uint8)
    polygon = np.array(
        [[
            (int(0.01 * width), height),
            (int(0.04 * width), int(0.06 * height)),
            (int(0.96 * width), int(0.06 * height)),
            (int(0.99 * width), height),
        ]],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, polygon, 255)
    return mask


def filter_lane_components(binary: np.ndarray) -> np.ndarray:
    """连通域筛选，去除与车道几何不符的白色噪声。"""
    h, w = binary.shape
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return binary

    min_area = max(120, int(0.00018 * h * w))
    kept: list[tuple[float, int]] = []

    for label in range(1, num_labels):
        x, y, bw, bh, area = stats[label]
        if area < min_area:
            continue

        # 仅在图像上方且未延伸到中下区域的斑块，视为噪声
        if (y + bh) < int(0.58 * h):
            continue

        cx = x + 0.5 * bw
        side_bonus = 1.2 if (cx < 0.48 * w or cx > 0.52 * w) else 0.8
        aspect = bh / max(1.0, float(bw))
        elongation_bonus = 1.0 + min(1.2, aspect)
        area_score = min(3.0, area / max(1.0, 0.01 * h * w))

        score = side_bonus * elongation_bonus + area_score
        kept.append((score, label))

    if not kept:
        return binary

    kept.sort(reverse=True, key=lambda item: item[0])
    max_keep = min(4, len(kept))
    selected = {label for _, label in kept[:max_keep]}

    filtered = np.zeros_like(binary)
    for label in selected:
        filtered[labels == label] = 255

    return filtered


def extract_lane_binary(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """颜色主导、梯度辅助的车道二值图。"""
    h, w = image.shape[:2]

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    yellow_a = cv2.inRange(hsv, HSV_YELLOW_LOWER_A, HSV_YELLOW_UPPER_A)
    yellow_b = cv2.inRange(hsv, HSV_YELLOW_LOWER_B, HSV_YELLOW_UPPER_B)
    yellow_hsv = cv2.bitwise_or(yellow_a, yellow_b)

    # HLS在远处低饱和和高反光区域更稳，和HSV互补
    yellow_hls = cv2.inRange(
        hls,
        np.array([12, 60, 40], dtype=np.uint8),
        np.array([46, 255, 255], dtype=np.uint8),
    )
    yellow = cv2.bitwise_or(yellow_hsv, yellow_hls)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobelx = np.absolute(sobelx)
    scaled = np.uint8(255 * abs_sobelx / (np.max(abs_sobelx) + 1e-6))
    grad_binary = np.zeros_like(scaled)
    grad_binary[(scaled >= 45) & (scaled <= 255)] = 255

    roi_mask = build_road_roi_mask(h, w)
    yellow = cv2.bitwise_and(yellow, roi_mask)
    grad_binary = cv2.bitwise_and(grad_binary, roi_mask)

    # 用膨胀后的颜色先验约束梯度，避免把纹理噪声大量引入
    yellow_dilated = cv2.dilate(yellow, np.ones((5, 5), dtype=np.uint8), iterations=1)
    grad_guided = cv2.bitwise_and(grad_binary, yellow_dilated)
    lane_binary = cv2.bitwise_or(yellow, grad_guided)

    # 白像素太少时，启用更宽松阈值兜底，优先保证线不断
    if cv2.countNonZero(lane_binary) < 1500:
        yellow_hsv_loose = cv2.inRange(
            hsv,
            np.array([10, 25, 55], dtype=np.uint8),
            np.array([50, 255, 255], dtype=np.uint8),
        )
        yellow_hls_loose = cv2.inRange(
            hls,
            np.array([8, 45, 20], dtype=np.uint8),
            np.array([52, 255, 255], dtype=np.uint8),
        )
        loose = cv2.bitwise_or(yellow_hsv_loose, yellow_hls_loose)
        loose = cv2.bitwise_and(loose, roi_mask)
        lane_binary = cv2.bitwise_or(lane_binary, loose)

    # 竖向闭运算优先连接细长车道线
    lane_binary = cv2.morphologyEx(
        lane_binary,
        cv2.MORPH_CLOSE,
        np.ones((5, 11), dtype=np.uint8),
        iterations=1,
    )
    lane_binary = cv2.morphologyEx(
        lane_binary,
        cv2.MORPH_OPEN,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    )
    lane_binary = filter_lane_components(lane_binary)
    lane_binary = cv2.bitwise_and(lane_binary, roi_mask)

    return lane_binary, roi_mask


def perspective_matrices(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """构建透视变换矩阵，将路面映射到俯视图。"""
    src = np.float32(
        [
            [0.01 * width, 0.98 * height],
            [0.33 * width, 0.42 * height],
            [0.67 * width, 0.42 * height],
            [0.99 * width, 0.98 * height],
        ]
    )
    dst = np.float32(
        [
            [0.20 * width, 1.00 * height],
            [0.20 * width, 0.00 * height],
            [0.80 * width, 0.00 * height],
            [0.80 * width, 1.00 * height],
        ]
    )
    mat = cv2.getPerspectiveTransform(src, dst)
    mat_inv = cv2.getPerspectiveTransform(dst, src)
    return mat, mat_inv


def sliding_window_polyfit(binary_warped: np.ndarray) -> tuple[Optional[np.ndarray], Optional[np.ndarray], str, float]:
    """在BEV中使用滑窗搜索左右车道，并拟合二次曲线 x=f(y)。"""
    h, w = binary_warped.shape
    histogram = np.sum(binary_warped[h // 2 :, :], axis=0)

    midpoint = w // 2
    leftx_base = int(np.argmax(histogram[:midpoint]))
    rightx_base = int(np.argmax(histogram[midpoint:]) + midpoint)

    nwindows = 9
    margin = 60
    minpix = 30
    window_height = h // nwindows

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base
    left_lane_inds: list[np.ndarray] = []
    right_lane_inds: list[np.ndarray] = []

    for window in range(nwindows):
        win_y_low = h - (window + 1) * window_height
        win_y_high = h - window * window_height

        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        good_left_inds = (
            (nonzeroy >= win_y_low)
            & (nonzeroy < win_y_high)
            & (nonzerox >= win_xleft_low)
            & (nonzerox < win_xleft_high)
        ).nonzero()[0]
        good_right_inds = (
            (nonzeroy >= win_y_low)
            & (nonzeroy < win_y_high)
            & (nonzerox >= win_xright_low)
            & (nonzerox < win_xright_high)
        ).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds) if left_lane_inds else np.array([], dtype=int)
    right_lane_inds = np.concatenate(right_lane_inds) if right_lane_inds else np.array([], dtype=int)

    left_fit: Optional[np.ndarray] = None
    right_fit: Optional[np.ndarray] = None

    if len(left_lane_inds) > 300:
        left_fit = np.polyfit(nonzeroy[left_lane_inds], nonzerox[left_lane_inds], 2)
    if len(right_lane_inds) > 300:
        right_fit = np.polyfit(nonzeroy[right_lane_inds], nonzerox[right_lane_inds], 2)

    base_lane_width = rightx_base - leftx_base
    if 0.25 * w <= base_lane_width <= 0.92 * w:
        lane_width_px = float(base_lane_width)
    else:
        lane_width_px = 0.50 * w

    if left_fit is not None and right_fit is None:
        right_fit = left_fit.copy()
        right_fit[2] += lane_width_px
    elif right_fit is not None and left_fit is None:
        left_fit = right_fit.copy()
        left_fit[2] -= lane_width_px

    left_found = len(left_lane_inds) > 300
    right_found = len(right_lane_inds) > 300
    if left_found and right_found:
        lane_status = "both"
    elif left_found or right_found:
        lane_status = "single"
    else:
        lane_status = "none"

    total_pixels = max(1, cv2.countNonZero(binary_warped))
    used_pixels = min(total_pixels, len(left_lane_inds) + len(right_lane_inds))
    confidence = clamp(used_pixels / total_pixels, 0.0, 1.0)
    if lane_status == "single":
        confidence *= 0.6
    if lane_status == "none":
        confidence = 0.0

    return left_fit, right_fit, lane_status, float(confidence)


def poly_x_at_y(poly: np.ndarray, y: np.ndarray | float) -> np.ndarray | float:
    return poly[0] * y * y + poly[1] * y + poly[2]


def lane_line_from_poly(poly: Optional[np.ndarray], width: int, height: int, top_ratio: float = 0.62) -> Optional[LaneLine]:
    """将二次曲线在可视范围采样成一条直线用于叠加显示。"""
    if poly is None:
        return None

    y_bottom = float(height - 1)
    y_top = float(int(top_ratio * height))
    x_bottom = int(clamp(float(poly_x_at_y(poly, y_bottom)), 0.0, width - 1.0))
    x_top = int(clamp(float(poly_x_at_y(poly, y_top)), 0.0, width - 1.0))

    dy = y_top - y_bottom
    dx = x_top - x_bottom
    slope = float(dy / dx) if dx != 0 else float("inf")

    return LaneLine(
        x1=x_bottom,
        y1=int(y_bottom),
        x2=x_top,
        y2=int(y_top),
        slope=slope,
    )


def transform_points(points: list[tuple[int, int]], transform: np.ndarray) -> list[tuple[int, int]]:
    """将点集从一个视角映射到另一个视角。"""
    if not points:
        return []

    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(pts, transform)
    mapped = mapped.reshape(-1, 2)
    return [(int(p[0]), int(p[1])) for p in mapped]


def trim_curve_with_binary(
    curve: list[tuple[int, int]],
    lane_binary: np.ndarray,
    radius: int = 7,
    min_keep: int = 8,
) -> list[tuple[int, int]]:
    """按二值车道图裁剪曲线，移除跑出白线区域的端点与离群段。"""
    if len(curve) < min_keep:
        return curve

    h, w = lane_binary.shape
    keep_flags: list[bool] = []
    for x, y in curve:
        x0 = max(0, x - radius)
        x1 = min(w, x + radius + 1)
        y0 = max(0, y - radius)
        y1 = min(h, y + radius + 1)
        keep_flags.append(cv2.countNonZero(lane_binary[y0:y1, x0:x1]) >= 5)

    segments: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []
    prev_pt: Optional[tuple[int, int]] = None
    for pt, keep in zip(curve, keep_flags):
        if prev_pt is not None:
            if math.hypot(pt[0] - prev_pt[0], pt[1] - prev_pt[1]) > 80:
                if current:
                    segments.append(current)
                    current = []
        if keep:
            current.append(pt)
        elif current:
            segments.append(current)
            current = []
        prev_pt = pt
    if current:
        segments.append(current)

    if not segments:
        return curve

    best = max(segments, key=len)
    if len(best) < min_keep:
        return []
    return best


def build_paths_from_poly(
    width: int,
    height: int,
    left_fit: Optional[np.ndarray],
    right_fit: Optional[np.ndarray],
    mat_inv: np.ndarray,
    num_points: int = 24,
    top_ratio: float = 0.38,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """构建中心线与车辆目标行驶线。"""
    image_center_x = width / 2.0
    lane_center_path: list[tuple[int, int]] = []
    predicted_path: list[tuple[int, int]] = []

    if left_fit is None or right_fit is None:
        return lane_center_path, predicted_path

    y_bev = np.linspace(height - 1, int(top_ratio * height), num_points)
    x_left = poly_x_at_y(left_fit, y_bev)
    x_right = poly_x_at_y(right_fit, y_bev)
    x_center = (x_left + x_right) / 2.0

    center_bev_points = [(int(xc), int(yc)) for xc, yc in zip(x_center, y_bev)]
    center_img_points = transform_points(center_bev_points, mat_inv)

    for i, (cx, cy) in enumerate(center_img_points):
        travel_t = i / max(1, len(center_img_points) - 1)
        blend = travel_t ** 0.65
        pred_x = int((1.0 - blend) * image_center_x + blend * cx)
        predicted_path.append((pred_x, cy))
        lane_center_path.append((cx, cy))

    return lane_center_path, predicted_path


def build_lane_curves_from_poly(
    width: int,
    height: int,
    left_fit: Optional[np.ndarray],
    right_fit: Optional[np.ndarray],
    mat_inv: np.ndarray,
    num_points: int = 36,
    top_ratio: float = 0.38,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """将BEV中的左右二次曲线采样后映射回原图，得到可视化曲线点。"""
    y_bev = np.linspace(height - 1, int(top_ratio * height), num_points)
    left_curve: list[tuple[int, int]] = []
    right_curve: list[tuple[int, int]] = []

    if left_fit is not None:
        x_left = poly_x_at_y(left_fit, y_bev)
        left_bev_points = [(int(x), int(y)) for x, y in zip(x_left, y_bev)]
        left_curve = transform_points(left_bev_points, mat_inv)
        left_curve = [
            (int(clamp(px, 0.0, width - 1.0)), int(clamp(py, 0.0, height - 1.0)))
            for px, py in left_curve
        ]

    if right_fit is not None:
        x_right = poly_x_at_y(right_fit, y_bev)
        right_bev_points = [(int(x), int(y)) for x, y in zip(x_right, y_bev)]
        right_curve = transform_points(right_bev_points, mat_inv)
        right_curve = [
            (int(clamp(px, 0.0, width - 1.0)), int(clamp(py, 0.0, height - 1.0)))
            for px, py in right_curve
        ]

    return left_curve, right_curve


def pure_pursuit_angle(
    predicted_path: list[tuple[int, int]],
    image_center_x: float,
    height: int,
    wheelbase_px: float,
) -> float:
    """按Pure Pursuit几何关系计算转角。"""
    if len(predicted_path) < 3:
        return 0.0

    lookahead_idx = int(0.6 * (len(predicted_path) - 1))
    target_x, target_y = predicted_path[lookahead_idx]

    dx = float(target_x) - image_center_x
    dy = float(height - target_y)
    ld = max(1.0, math.hypot(dx, dy))
    alpha = math.atan2(dx, max(1.0, dy))

    steer_rad = math.atan2(2.0 * wheelbase_px * math.sin(alpha), ld)
    return math.degrees(steer_rad)


def estimate_steering_angle(
    width: int,
    height: int,
    left_fit: Optional[np.ndarray],
    right_fit: Optional[np.ndarray],
    mat_inv: np.ndarray,
    lane_status: str,
    lane_confidence: float,
) -> RoadAngleResult:
    """估计道路转角并构建可跟踪路径。"""
    image_center_x = width / 2.0

    lane_center_path, predicted_path = build_paths_from_poly(
        width=width,
        height=height,
        left_fit=left_fit,
        right_fit=right_fit,
        mat_inv=mat_inv,
    )

    lane_center_x: Optional[float] = None
    lane_center_bottom_x: Optional[float] = None
    if lane_center_path:
        lane_center_bottom_x = float(lane_center_path[0][0])
        lane_center_x = float(lane_center_path[-1][0])

    if lane_status == "none" or not predicted_path:
        raw_steering_angle = 0.0
        # 丢线时给直行兜底路径，避免输出空路径
        y_samples = np.linspace(height - 1, int(0.45 * height), 14).astype(int)
        lane_center_path = [(int(image_center_x), int(y)) for y in y_samples]
        predicted_path = [(int(image_center_x), int(y)) for y in y_samples]
        target_x, target_y = predicted_path[-1]
    else:
        target_x, target_y = predicted_path[-1]
        dx = float(target_x) - image_center_x
        dy = float(height - 1 - target_y)
        raw_steering_angle = math.degrees(math.atan2(dx, max(1.0, dy)))

    actual_straight_path = [(int(image_center_x), int(height - 1)), (int(target_x), int(target_y))]

    # 低置信度时降低控制激进程度
    raw_steering_angle *= (0.45 + 0.55 * lane_confidence)
    raw_steering_angle = clamp(raw_steering_angle, -MAX_STEERING_ANGLE_DEG, MAX_STEERING_ANGLE_DEG)
    steering_angle = smooth_steering_angle(raw_steering_angle)

    left_lane = lane_line_from_poly(left_fit, width=width, height=height)
    right_lane = lane_line_from_poly(right_fit, width=width, height=height)
    left_curve, right_curve = build_lane_curves_from_poly(
        width=width,
        height=height,
        left_fit=left_fit,
        right_fit=right_fit,
        mat_inv=mat_inv,
    )

    return RoadAngleResult(
        steering_angle_deg=float(steering_angle),
        raw_steering_angle_deg=float(raw_steering_angle),
        lane_center_x=lane_center_x,
        lane_center_bottom_x=lane_center_bottom_x,
        image_center_x=float(image_center_x),
        left_lane=left_lane,
        right_lane=right_lane,
        lane_center_path=lane_center_path,
        predicted_path=predicted_path,
        actual_straight_path=actual_straight_path,
        left_lane_curve=left_curve,
        right_lane_curve=right_curve,
        lane_status=lane_status,
        lane_confidence=float(lane_confidence),
    )


def draw_result(image: np.ndarray, result: RoadAngleResult, lane_binary: np.ndarray) -> np.ndarray:
    """绘制车道线、中心线、路径与状态信息。"""
    vis = image.copy()
    h, _ = vis.shape[:2]

    left_curve = trim_curve_with_binary(result.left_lane_curve, lane_binary)
    right_curve = trim_curve_with_binary(result.right_lane_curve, lane_binary)

    if len(left_curve) >= 2:
        left_pts = np.array(left_curve, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [left_pts], isClosed=False, color=(0, 255, 0), thickness=4)
    elif result.left_lane:
        cv2.line(
            vis,
            (result.left_lane.x1, result.left_lane.y1),
            (result.left_lane.x2, result.left_lane.y2),
            (0, 255, 0),
            4,
        )

    if len(right_curve) >= 2:
        right_pts = np.array(right_curve, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [right_pts], isClosed=False, color=(0, 255, 0), thickness=4)
    elif result.right_lane:
        cv2.line(
            vis,
            (result.right_lane.x1, result.right_lane.y1),
            (result.right_lane.x2, result.right_lane.y2),
            (0, 255, 0),
            4,
        )

    img_center = (int(result.image_center_x), h - 1)
    cv2.circle(vis, img_center, 6, (255, 0, 0), -1)

    if result.lane_center_path:
        center_pts = np.array(result.lane_center_path, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [center_pts], isClosed=False, color=(255, 255, 255), thickness=2)

    if result.predicted_path:
        route_pts = np.array(result.predicted_path, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [route_pts], isClosed=False, color=(255, 0, 0), thickness=4)

    if len(result.actual_straight_path) == 2:
        cv2.line(
            vis,
            result.actual_straight_path[0],
            result.actual_straight_path[1],
            (0, 0, 255),
            4,
        )

    direction_text = "straight"
    if result.steering_angle_deg > 2:
        direction_text = "turn right"
    elif result.steering_angle_deg < -2:
        direction_text = "turn left"

    cv2.putText(
        vis,
        f"Turn Angle: {result.steering_angle_deg:.2f} deg",
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        vis,
        f"Raw: {result.raw_steering_angle_deg:.2f} deg",
        (20, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        vis,
        f"Lane: {result.lane_status}  Conf: {result.lane_confidence:.2f}",
        (20, 106),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        vis,
        f"Decision: {direction_text}",
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    lane_binary_bgr = cv2.cvtColor(lane_binary, cv2.COLOR_GRAY2BGR)
    lane_binary_bgr = cv2.resize(lane_binary_bgr, (vis.shape[1], vis.shape[0]))
    panel = np.hstack((vis, lane_binary_bgr))
    return panel


def compute_road_angle(image: np.ndarray) -> tuple[RoadAngleResult, np.ndarray, np.ndarray]:
    """完整管线：HSV黄线 -> ROI -> BEV -> 滑窗拟合 -> 中心路径 -> Pure Pursuit。"""
    h, w = image.shape[:2]

    lane_binary, roi_mask = extract_lane_binary(image)

    mat, mat_inv = perspective_matrices(width=w, height=h)
    warped = cv2.warpPerspective(lane_binary, mat, (w, h), flags=cv2.INTER_LINEAR)

    left_fit, right_fit, lane_status, lane_confidence = sliding_window_polyfit(warped)

    result = estimate_steering_angle(
        width=w,
        height=h,
        left_fit=left_fit,
        right_fit=right_fit,
        mat_inv=mat_inv,
        lane_status=lane_status,
        lane_confidence=lane_confidence,
    )
    vis = draw_result(image, result, lane_binary)

    return result, vis, roi_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="规则式道路转角估计（HSV+BEV+PurePursuit）")
    parser.add_argument(
        "--image",
        type=str,
        default=str(Path("Resources") / "road_detect_1.png"),#########
        #######################################
        help="输入道路图像路径",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"未找到输入图像: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"OpenCV无法读取图像: {image_path}")

    result, vis, _ = compute_road_angle(image)

    print("=== Road Turn Angle Estimation ===")
    print(f"Image: {image_path}")
    print(f"Steering angle (deg): {result.steering_angle_deg:.2f}")
    print(f"Raw steering angle (deg): {result.raw_steering_angle_deg:.2f}")
    print(f"Lane status: {result.lane_status}")
    print(f"Lane confidence: {result.lane_confidence:.2f}")
    print(f"Image center x: {result.image_center_x:.2f}")
    if result.lane_center_x is None:
        print("Lane center x: N/A (未检测到有效车道线)")
    else:
        print(f"Lane center x: {result.lane_center_x:.2f}")
    print(f"Predicted path points: {len(result.predicted_path)}")

    cv2.imshow("road_angle", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
