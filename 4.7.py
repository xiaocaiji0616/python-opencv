import cv2
import numpy as np
from skimage import util
import matplotlib.pyplot as plt

img = cv2.imread("Resources/lena.png")

# 复制一份图像，避免直接修改原图
noisy_sp_img = img.copy()

noise_sp_img = util.random_noise(img, mode='s&p', amount=0.02)
noise_sp_img = np.array(255*noise_sp_img, dtype = 'uint8')

# 用中值滤波去除椒盐噪声，结果存入新的图像变量 imgnew
imgnew = cv2.medianBlur(noise_sp_img, 5)

# OpenCV 读入的是 BGR，需要转换为 RGB 再用 plt 正确显示颜色
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
noisy_rgb = cv2.cvtColor(noise_sp_img, cv2.COLOR_BGR2RGB)
imgnew_rgb = cv2.cvtColor(imgnew, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 6))#创建一个 12x6 英寸的图像窗口

plt.subplot(1, 3, 1)
plt.imshow(img_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(noisy_rgb)
plt.title("Salt and Pepper Noise")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(imgnew_rgb)
plt.title("Median Filter Result")
plt.axis("off")

plt.tight_layout()
plt.show()
