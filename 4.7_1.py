import cv2
import numpy as np
from skimage import util
from skimage.filters import roberts
import matplotlib.pyplot as plt
img = cv2.imread("Resources/paper.jpg") 
noise_sp_img = util.random_noise(img, mode='s&p', amount=0.2)
noise_sp_img = np.array(255*noise_sp_img, dtype = 'uint8')#将浮点数图像转换为 uint8 类型
imgnew = cv2.medianBlur(noise_sp_img, 5)
# 对中值滤波后的图像进行边缘检测
gray_imgnew = cv2.cvtColor(imgnew, cv2.COLOR_BGR2GRAY)
imgCanny = cv2.Canny(gray_imgnew, 100, 200)
imgRoberts = roberts(gray_imgnew)
# OpenCV 默认是 BGR，使用 plt 显示前先转换为 RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
noise_rgb = cv2.cvtColor(noise_sp_img, cv2.COLOR_BGR2RGB)
imgnew_rgb = cv2.cvtColor(imgnew, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(14, 8))
plt.subplot(2, 3, 1)
plt.imshow(img_rgb)
plt.title("Original")
plt.axis("off")
plt.subplot(2, 3, 2)
plt.imshow(noise_rgb)
plt.title("Salt and Pepper Noise")
plt.axis("off")
plt.subplot(2, 3, 3)
plt.imshow(imgnew_rgb)
plt.title("Median Filter Result")
plt.axis("off")
plt.subplot(2, 3, 4)
plt.imshow(imgCanny, cmap="gray")
plt.title("Canny Edge Detection")
plt.axis("off")
plt.subplot(2, 3, 5)
plt.imshow(imgRoberts, cmap="gray")
plt.title("Roberts Edge Detection")
plt.axis("off")
plt.tight_layout()
plt.show()