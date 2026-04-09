import cv2
import numpy as np
original_img = cv2.imread("Resources/chapter9_pics/road.jpg")  # 替换为你的图像路径
img = cv2.resize(original_img, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_CUBIC)  # 调整图像大小，减小计算量
img = cv2.GaussianBlur(img, (3, 3), 0)  # 高斯模糊，降噪
edges = cv2.Canny(img, 50, 150, apertureSize=3)  # Canny边缘检测
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 150)  # 霍夫
print("Line Num:", len(lines))
result = img.copy()

for line in lines:
    rho = line[0][0]
    theta = line[0][1]
    if theta < np.pi / 4 or theta > 3 * np.pi / 4:  # 垂直线
        pt1 = (int(rho / np.cos(theta)), 0)
        pt2 = (int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta)), result.shape[0])
    else:  # 水平线
        pt1 = (0, int(rho / np.sin(theta)))
        pt2 = (result.shape[1], int((rho - result.shape[1] * np.cos(theta)) / np.sin(theta)))
    cv2.line(result, pt1, pt2, (0, 0,255), 2)
cv2.imshow("Original Image", img)
cv2.imshow("Edges", edges)
cv2.imshow("Detected Lines", result)
cv2.waitKey(0)
cv2.destroyAllWindows()