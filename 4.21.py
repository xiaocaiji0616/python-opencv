import cv2
import numpy as np
from matplotlib import pyplot as plt

#img = cv2.imread("Resources/chapter8_pics/j_noise.png",0) 
# kernel = np.ones((5,5),np.uint8)
# opening = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
# plt.figure('MORPH_OPEN',figsize=(8,8))
# plt.subplot(1,2,1),plt.imshow(img,'gray')
# plt.title('Original Image'),plt.xticks([]),plt.yticks([])
# plt.subplot(1,2,2),plt.imshow(opening,'gray')
# plt.title('MORPH_OPEN'),plt.xticks([]),plt.yticks([])
# plt.show()


# img = cv2.imread("resources/heighthat.png")
# kernel = np.ones((9,9),np.uint8)
# tophat = cv2.morphologyEx(img,cv2.MORPH_TOPHAT,kernel)
# plt.figure('MORPH_TOPHAT',figsize=(8,8))
# plt.subplot(1,2,1),plt.imshow(img,'gray')
# plt.title('Original Image'),plt.xticks([]),plt.yticks([])
# plt.subplot(1,2,2),plt.imshow(tophat,'gray')
# plt.title('MORPH_TOPHAT'),plt.xticks([]),plt.yticks([])
# plt.show()


img = cv2.imread("resources/exam.jpg",0)
kernel = np.ones((11,11),np.uint8)
blackhat = cv2.morphologyEx(img,cv2.MORPH_BLACKHAT,kernel)
#取反

plt.figure('MORPH_BLACKHAT',figsize=(8,8))
plt.subplot(1,2,1),plt.imshow(img,'gray')
plt.title('Original Image'),plt.xticks([]),plt.yticks([])
plt.subplot(1,2,2),plt.imshow(blackhat,'gray')
plt.title('MORPH_BLACKHAT'),plt.xticks([]),plt.yticks([])
plt.show()