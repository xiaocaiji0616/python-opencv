import cv2
import numpy as np
original_img = cv2.imread("Resources/chapter9_pics/road.jpg")  # 替换为你的图像路径
if original_img is None:
    raise RuntimeError("图像读取失败，请检查路径 Resources/chapter9_pics/road.jpg")

img = cv2.resize(original_img, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_CUBIC)  # 调整图像大小，减小计算量
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Canny 前先转灰度
blur = cv2.GaussianBlur(gray, (5, 5), 0)  # 轻度高斯模糊，抑制噪声
edges = cv2.Canny(blur, 50, 150, apertureSize=3)  # Canny边缘检测

# 全图检测：使用标准霍夫变换，返回 (rho, theta)
# 阈值过高时容易一条都检测不到，这里做分级回退
lines = None
for hough_threshold in [140, 120, 100, 80]:
    lines = cv2.HoughLines(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold   
    )
    if lines is not None:
        print("Use Hough threshold:", hough_threshold)
        break

result = img.copy()
line_count = 0
if lines is not None:
    # 不再直接截取前 N 条；先按 (rho, theta) 去重，减少重复强线把弱线挤掉
    lines_rt = [(float(line[0][0]), float(line[0][1])) for line in lines]
    lines_rt.sort(key=lambda x: x[0])

    unique_lines = []
    for rho, theta in lines_rt:
        is_duplicate = False
        for urho, utheta in unique_lines:
            # rho/theta 都很接近时，认为是同一条线的重复检测
            if abs(rho - urho) < 12 and abs(theta - utheta) < np.deg2rad(2):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_lines.append((rho, theta))

    # 绘制去重后的线：同一逻辑分类着色，不做“单独补线”
    for rho, theta in unique_lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho

        # 按方向向两侧延长，得到贯穿整图的长直线
        x1 = int(x0 + 2000 * (-b))
        y1 = int(y0 + 2000 * (a))
        x2 = int(x0 - 2000 * (-b))
        y2 = int(y0 - 2000 * (a))

        # theta 接近 pi/2 时更偏向水平线，用红色；其余线用绿色
        color = (0, 0, 255) if abs(theta - np.pi / 2) < np.deg2rad(10) else (0, 255, 0)
        cv2.line(result, (x1, y1), (x2, y2), color, 2)
        line_count += 1

print("Line Num:", line_count)

cv2.imshow("Original Image", img)
cv2.imshow("Edges", edges)
cv2.imshow("Detected Lines", result)
cv2.waitKey(0)
cv2.destroyAllWindows()