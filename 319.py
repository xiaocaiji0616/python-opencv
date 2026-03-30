import cv2 as cv
import numpy as np
img = cv.imread("Resources/lena.png")
img_gray = cv.imread("Resources/lena.png", cv.IMREAD_GRAYSCALE)

cv.imwrite("Resources/lena_gray.png", img_gray)
num = img[100,100]
print(num)
region = img[0:300, 0:300]

b, g, r = cv.split(img)



cv.imshow("Image",img)
cv.imshow("Gray Image", img_gray)
cv.imshow("Region", region)
cv.imshow("Blue",b)
cv.imshow("Green",g)
cv.imshow("Red",r)

print(img.shape)
print(img.size)
print(img.dtype)
print(img_gray.shape)
print(img_gray.size)
print(img_gray.dtype)



cv.waitKey(0)