import cv2
import numpy as np
img = cv2.imread("Resources/yingbi.jpg")
if img is None:
    raise RuntimeError("图像读取失败，请检查路径 Resources/yingbi.jpg")
result = img.copy()
# 1. 灰度化和高斯模糊，减少背景噪声对圆检测的影响
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
# 2. Canny边缘图，用来观察硬币边缘特征
edges = cv2.Canny(blur, 50, 150)
# 3. 霍夫圆检测硬币边缘
circles = cv2.HoughCircles(
    blur,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=35,
    param1=80,
    param2=30,
    minRadius=25,
    maxRadius=42
)
if circles is None:
    print("未检测到硬币")
else:
    circles = np.around(circles[0]).astype(int)
    circles = sorted(circles, key=lambda c: (c[1], c[0]))

    print("硬币个数：", len(circles))

    for num, (x, y, r) in enumerate(circles, start=1):
        area = np.pi * r * r

        cv2.circle(result, (x, y), r, (0, 255, 0), 2)
        cv2.circle(result, (x, y), 2, (0, 255, 0), 3)
        cv2.putText(result, str(num), (x - 10, y + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        print("第{}个硬币面积：{:.2f}".format(num, area))

cv2.imshow("Original Image", img)
cv2.imshow("Coin Edges", edges)
cv2.imshow("Detected Coins", result)
cv2.waitKey(0)
cv2.destroyAllWindows()