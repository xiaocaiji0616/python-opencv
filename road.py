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
TARGET_Y_RATIOS: tuple[float, ...] = (0.74, 0.66, 0.58)


@dataclass
class LaneLine:
	"""用两点表示一条拟合后的车道线。"""

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


def region_of_interest(edges: np.ndarray) -> np.ndarray:
	"""只保留道路区域（近似梯形），减少误检。"""
	h, w = edges.shape
	mask = np.zeros_like(edges)

	# 量产常见思路：固定视场下使用路面ROI，忽略天空/车头等区域
	polygon = np.array(
		[[
			(int(0.08 * w), h),
			(int(0.42 * w), int(0.62 * h)),
			(int(0.58 * w), int(0.62 * h)),
			(int(0.92 * w), h),
		]],
		dtype=np.int32,
	)

	cv2.fillPoly(mask, polygon, 255)
	return cv2.bitwise_and(edges, mask)


def detect_line_segments(roi_edges: np.ndarray) -> Optional[np.ndarray]:
	"""HoughLinesP提取线段。"""
	return cv2.HoughLinesP(
		roi_edges,
		rho=1,
		theta=np.pi / 180,
		threshold=30,
		minLineLength=30,
		maxLineGap=80,
	)


def fit_lane_line(points: list[tuple[int, int]], y_bottom: int, y_top: int) -> Optional[LaneLine]:
	"""将同侧线段点集拟合为一条稳定车道线。"""
	if len(points) < 2:
		return None

	xs = np.array([p[0] for p in points], dtype=np.float64)
	ys = np.array([p[1] for p in points], dtype=np.float64)

	# x = m*y + b，比 y = kx + b 在垂直线场景更稳定
	m, b = np.polyfit(ys, xs, deg=1)
	x1 = int(m * y_bottom + b)
	x2 = int(m * y_top + b)

	# 转成传统斜率（便于调试展示）
	dy = y_top - y_bottom
	dx = x2 - x1
	slope = float(dy / dx) if dx != 0 else float("inf")

	return LaneLine(x1=x1, y1=y_bottom, x2=x2, y2=y_top, slope=slope)


def classify_and_average_lanes(
	line_segments: Optional[np.ndarray],
	width: int,
	height: int,
) -> tuple[Optional[LaneLine], Optional[LaneLine]]:
	"""按斜率分类左右车道线段，并拟合为左右两条边线。"""
	if line_segments is None:
		return None, None

	left_points: list[tuple[int, int]] = []
	right_points: list[tuple[int, int]] = []

	y_top = int(0.62 * height)
	y_bottom = height

	for seg in line_segments:
		x1, y1, x2, y2 = seg[0]
		if x1 == x2:
			continue

		slope = (y2 - y1) / (x2 - x1)
		if abs(slope) < 0.35:  # 过滤近水平噪声
			continue

		# 使用线段在车辆前方底部(y_bottom)处的交点来判定左右，更适合弯道
		x_bottom = x1 + (y_bottom - y1) * (x2 - x1) / (y2 - y1)
		if x_bottom < width / 2:
			left_points.extend([(x1, y1), (x2, y2)])
		else:
			right_points.extend([(x1, y1), (x2, y2)])

	left_lane = fit_lane_line(left_points, y_bottom=y_bottom, y_top=y_top)
	right_lane = fit_lane_line(right_points, y_bottom=y_bottom, y_top=y_top)
	return left_lane, right_lane


def lane_x_at_y(line: LaneLine, y: int) -> float:
	"""根据LaneLine在指定y处求x（线性插值/外推）。"""
	dy = line.y2 - line.y1
	if dy == 0:
		return float(line.x1)
	t = (y - line.y1) / dy
	return float(line.x1 + t * (line.x2 - line.x1))


def half_lane_width_at_y(width: int, height: int, y: int) -> float:
	"""单侧车道可见时的半车道宽度估计（近处更宽，远处更窄）。"""
	top_y = int(0.58 * height)
	bottom_y = height
	if bottom_y == top_y:
		return 0.25 * width
	t = (y - top_y) / (bottom_y - top_y)
	t = clamp(t, 0.0, 1.0)
	return (0.22 + 0.06 * t) * width


def estimate_lane_boundaries_at_y(
	width: int,
	height: int,
	y: int,
	left_lane: Optional[LaneLine],
	right_lane: Optional[LaneLine],
) -> tuple[float, float]:
	"""在指定y处估计左右边界x，缺边时做对称补全。"""
	half_width = half_lane_width_at_y(width, height, y)

	if left_lane and right_lane:
		x_left = lane_x_at_y(left_lane, y)
		x_right = lane_x_at_y(right_lane, y)
		if x_left > x_right:
			x_left, x_right = x_right, x_left
		return x_left, x_right

	if left_lane:
		x_left = lane_x_at_y(left_lane, y)
		x_right = x_left + 2.0 * half_width
		return x_left, x_right

	if right_lane:
		x_right = lane_x_at_y(right_lane, y)
		x_left = x_right - 2.0 * half_width
		return x_left, x_right

	# 双侧都没有时：以图像中心为默认车道中心
	image_center_x = width / 2
	return image_center_x - half_width, image_center_x + half_width


def build_paths(
	width: int,
	height: int,
	image_center_x: float,
	left_lane: Optional[LaneLine],
	right_lane: Optional[LaneLine],
	num_points: int = 18,
	top_ratio: float = 0.58,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
	"""
	返回两条路径：
	- lane_center_path: 道路中心线（由左右实线中点拟合）
	- predicted_path: 车辆行驶路线（从车头平滑收敛到中心线）
	"""
	top_y = int(top_ratio * height)
	if num_points < 2:
		num_points = 2

	y_samples = np.linspace(height, top_y, num_points).astype(int)
	lane_center_path: list[tuple[int, int]] = []
	predicted_path: list[tuple[int, int]] = []

	for y in y_samples:
		center_x: Optional[float] = None
		if left_lane and right_lane:
			x_left = lane_x_at_y(left_lane, int(y))
			x_right = lane_x_at_y(right_lane, int(y))
			center_x = (x_left + x_right) / 2.0
		elif left_lane:
			center_x = lane_x_at_y(left_lane, int(y)) + half_lane_width_at_y(width, height, int(y))
		elif right_lane:
			center_x = lane_x_at_y(right_lane, int(y)) - half_lane_width_at_y(width, height, int(y))

		if center_x is None:
			continue

		# 车辆路线从车头中心出发，向前逐步贴合道路中心线
		travel_t = (height - y) / max(1, (height - top_y))
		travel_t = clamp(travel_t, 0.0, 1.0)
		blend = travel_t ** 0.7
		pred_x = (1.0 - blend) * image_center_x + blend * center_x

		lane_center_path.append((int(center_x), int(y)))
		predicted_path.append((int(pred_x), int(y)))

	if not predicted_path:
		# 兜底：至少给一条直行路线
		for y in y_samples:
			predicted_path.append((int(image_center_x), int(y)))
			lane_center_path.append((int(image_center_x), int(y)))

	return lane_center_path, predicted_path


def estimate_steering_angle(
	width: int,
	height: int,
	left_lane: Optional[LaneLine],
	right_lane: Optional[LaneLine],
) -> RoadAngleResult:
	"""
	估计道路转角：
	- 以图像中心作为车辆当前位置；
	- 以车道中心偏移作为横向误差；
	- 将横向误差换算为转向角（度）。
	"""
	image_center_x = width / 2
	y_bottom = height
	lane_center_path, predicted_path = build_paths(width, height, image_center_x, left_lane, right_lane)

	lane_center_x: Optional[float] = None
	lane_center_bottom_x: Optional[float] = None

	if lane_center_path:
		lane_center_bottom_x = float(lane_center_path[0][0])
		lane_center_x = float(lane_center_path[-1][0])

	if lane_center_x is None or lane_center_bottom_x is None or len(predicted_path) < 2:
		raw_steering_angle = 0.0
	else:
		near_idx = min(3, len(predicted_path) - 1)
		far_idx = len(predicted_path) - 1

		x_near, y_near = predicted_path[near_idx]
		x_far, y_far = predicted_path[far_idx]

		# 1) 行驶路线切线角
		dx_heading = x_far - x_near
		dy_heading = y_near - y_far
		heading_angle = math.degrees(math.atan2(dx_heading, dy_heading))

		# 2) 近处横向偏差修正
		dx_offset = x_near - image_center_x
		offset_angle = math.degrees(math.atan2(dx_offset, max(1, dy_heading)))

		# 行驶方向优先，偏差次之
		raw_steering_angle = 0.65 * heading_angle + 0.35 * offset_angle

	raw_steering_angle = clamp(raw_steering_angle, -MAX_STEERING_ANGLE_DEG, MAX_STEERING_ANGLE_DEG)
	steering_angle = smooth_steering_angle(raw_steering_angle)

	return RoadAngleResult(
		steering_angle_deg=float(steering_angle),
		raw_steering_angle_deg=float(raw_steering_angle),
		lane_center_x=lane_center_x,
		lane_center_bottom_x=lane_center_bottom_x,
		image_center_x=image_center_x,
		left_lane=left_lane,
		right_lane=right_lane,
		lane_center_path=lane_center_path,
		predicted_path=predicted_path,
	)


def draw_result(image: np.ndarray, result: RoadAngleResult) -> np.ndarray:
	"""绘制车道线、中心线、目标点和角度文本。"""
	vis = image.copy()
	h, w = vis.shape[:2]

	if result.left_lane:
		cv2.line(
			vis,
			(result.left_lane.x1, result.left_lane.y1),
			(result.left_lane.x2, result.left_lane.y2),
			(0, 255, 0),
			5,
		)
	if result.right_lane:
		cv2.line(
			vis,
			(result.right_lane.x1, result.right_lane.y1),
			(result.right_lane.x2, result.right_lane.y2),
			(0, 255, 0),
			5,
		)

	img_center = (int(result.image_center_x), h)
	cv2.circle(vis, img_center, 6, (255, 0, 0), -1)

	if result.lane_center_path:
		center_pts = np.array(result.lane_center_path, dtype=np.int32).reshape(-1, 1, 2)
		cv2.polylines(vis, [center_pts], isClosed=False, color=(255, 255, 255), thickness=2)

	if result.predicted_path:
		pred_pts = np.array(result.predicted_path, dtype=np.int32).reshape(-1, 1, 2)
		# 青线：预计行驶路线（保持中线并连续收敛）
		cv2.polylines(vis, [pred_pts], isClosed=False, color=(255, 255, 0), thickness=4)

	if result.predicted_path:
		route_pts = np.array(result.predicted_path, dtype=np.int32).reshape(-1, 1, 2)
		# 蓝色：车辆行驶路线
		cv2.polylines(vis, [route_pts], isClosed=False, color=(255, 0, 0), thickness=4)

	direction_text = "straight"
	if result.steering_angle_deg > 2:
		direction_text = "turn right"
	elif result.steering_angle_deg < -2:
		direction_text = "turn left"

	cv2.putText(
		vis,
		f"Steering Angle: {result.steering_angle_deg:.2f} deg",
		(20, 40),
		cv2.FONT_HERSHEY_SIMPLEX,
		1.0,
		(0, 255, 255),
		2,
		cv2.LINE_AA,
	)
	cv2.putText(
		vis,
		f"Route Angle(raw): {result.raw_steering_angle_deg:.2f} deg",
		(20, 80),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.8,
		(0, 255, 255),
		2,
		cv2.LINE_AA,
	)
	cv2.putText(
		vis,
		f"Decision: {direction_text}",
		(20, 115),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.9,
		(0, 255, 255),
		2,
		cv2.LINE_AA,
	)
	return vis


def compute_road_angle(image: np.ndarray) -> tuple[RoadAngleResult, np.ndarray, np.ndarray]:
	"""完整管线：预处理 -> 边缘 -> ROI -> 线段 -> 车道拟合 -> 转角估计。"""
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (5, 5), 0)
	edges = cv2.Canny(blur, 80, 160)

	roi_edges = region_of_interest(edges)
	line_segments = detect_line_segments(roi_edges)

	h, w = image.shape[:2]
	left_lane, right_lane = classify_and_average_lanes(line_segments, w, h)
	result = estimate_steering_angle(w, h, left_lane, right_lane)
	vis = draw_result(image, result)

	return result, vis, roi_edges


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="规则式道路转角估计（无神经网络）")
	parser.add_argument(
		"--image",
		type=str,
		default=str(Path("Resources") / "road_detect.png"),
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

	result, vis, roi_edges = compute_road_angle(image)

	print("=== Road Turn Angle Estimation ===")
	print(f"Image: {image_path}")
	print(f"Steering angle (deg): {result.steering_angle_deg:.2f}")
	print(f"Image center x: {result.image_center_x:.2f}")
	if result.lane_center_x is None:
		print("Lane center x: N/A (未检测到有效车道线)")
	else:
		print(f"Lane center x: {result.lane_center_x:.2f}")
	print(f"Predicted path points: {len(result.predicted_path)}")

	# 按用户需求：不新建输出图片，直接弹窗显示
	cv2.imshow("road_angle", vis)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
