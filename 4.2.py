import cv2
from matplotlib import pyplot as plt
import numpy as np
img = cv2.imread('Resources/lena.png')
img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
mask = np.zeros(img.shape[:2], np.uint8)
mask[200:400, 200:400] = 255
masked_img = cv2.bitwise_and(img_RGB, img_RGB, mask=mask)
hist_full = cv2.calcHist([img_RGB], [0], None, [256], [0, 256])
hist_mask = cv2.calcHist([img_RGB], [0], mask, [256], [0, 256])
plt.subplot(221), plt.imshow(img_RGB, 'gray'), plt.title('Original Image')
plt.subplot(222), plt.imshow(masked_img, 'gray'), plt.title('Mask')
plt.subplot(223), plt.plot(hist_full, color='r')
plt.subplot(224), plt.plot(hist_mask, color='b')
plt.xlim([0, 256])
plt.show()