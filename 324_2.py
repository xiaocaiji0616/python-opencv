import cv2 as cv
import numpy as np
img = cv.imread("Resources/lena.png")
cv.imshow("Image",img)
img2=np.zeros(img.shape, np.uint8)
img2[50:350, 50:300] = 255
cv.imshow("Image2",img2)
img3=cv.bitwise_and(img,img2)
cv.imshow("Image3",img3)
cv.waitKey(0)